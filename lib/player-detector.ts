import type {Box} from './tracking';
import {shirtFeature} from './track-vision';
import type {TrackDetection} from './persistent-tracker';
import {cropsFor,refinementCrops,decodeYolox,suppress,toTensor,INPUT_SIZE,type Detection} from './detection-core';
import type {InferenceSession,Tensor} from 'onnxruntime-web';
import {detectPitch} from './pitch-detector';
import {cropQuality,PlayerReID} from './player-reid';
import {JerseyGuide,type JerseyFilter} from './jersey-filter';
const reid=new PlayerReID();
let appearanceUnavailable=false;
export const appearanceMatchingAvailable=()=>!appearanceUnavailable;
let sessionPromise:Promise<InferenceSession>|undefined;
async function session(){if(!sessionPromise)sessionPromise=(async()=>{const ort=await import('onnxruntime-web/wasm');ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('/ort/',window.location.origin).href;const response=await fetch('/models/yolox-tiny.onnx',{signal:AbortSignal.timeout(60000)});if(!response.ok)throw Error('Detector download failed. Reload and try again.');const bytes=await response.arrayBuffer();return ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});})().catch(e=>{sessionPromise=undefined;throw e;});return sessionPromise;}
export async function detectFrame(source:HTMLVideoElement|HTMLCanvasElement,{detailed=true,minScore=.18,pitchOnly=true,jerseys,jerseyGuide,focus,refine=false,embeddings=false,onProgress=()=>{},cancelled=()=>false}:{detailed?:boolean;minScore?:number;pitchOnly?:boolean;jerseys?:JerseyFilter;jerseyGuide?:JerseyGuide;focus?:Box;refine?:boolean;embeddings?:boolean;onProgress?:(message:string)=>void;cancelled?:()=>boolean}={}):Promise<TrackDetection[]>{
 onProgress('Loading detector (first use downloads about 30 MB)…');const model=await session();if(cancelled())throw Error('Detection stopped.');const ort=await import('onnxruntime-web/wasm');const width='videoWidth'in source?source.videoWidth:source.width,height='videoHeight'in source?source.videoHeight:source.height;if(!width||!height)throw Error('Wait for the video frame to load.');const snapshot=document.createElement('canvas');const shrink=Math.min(1,1920/Math.max(width,height));snapshot.width=Math.round(width*shrink);snapshot.height=Math.round(height*shrink);snapshot.getContext('2d')!.drawImage(source,0,0,snapshot.width,snapshot.height);const scanWidth=snapshot.width,scanHeight=snapshot.height;const input=document.createElement('canvas');input.width=INPUT_SIZE;input.height=INPUT_SIZE;const ctx=input.getContext('2d',{willReadFrequently:true})!;const size=Math.min(256,scanWidth,scanHeight);const crops=focus?[{x:Math.max(0,Math.min(scanWidth-size,(focus.x+focus.w/2)*scanWidth-size/2)),y:Math.max(0,Math.min(scanHeight-size,(focus.y+focus.h/2)*scanHeight-size/2)),w:size,h:size}]:cropsFor(scanWidth,scanHeight,detailed),all:Detection[]=[];const baseCropCount=crops.length;
 for(let i=0;i<crops.length;i++){if(cancelled())throw Error('Detection stopped.');onProgress(`Finding people and ball · area ${i+1}/${crops.length}`);const crop=crops[i],ratio=Math.min(INPUT_SIZE/crop.w,INPUT_SIZE/crop.h);ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);ctx.drawImage(snapshot,crop.x,crop.y,crop.w,crop.h,0,0,Math.floor(crop.w*ratio),Math.floor(crop.h*ratio));const tensor=new ort.Tensor('float32',toTensor(ctx.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data),[1,3,INPUT_SIZE,INPUT_SIZE]);let output:Record<string,Tensor>|undefined;try{output=await model.run({[model.inputNames[0]]:tensor});const result=output[model.outputNames[0]];if(result.dims.join(',')!=='1,3549,85')throw Error('Unexpected detector output.');all.push(...decodeYolox(result.data as Float32Array,crop,scanWidth,scanHeight,minScore));}finally{tensor.dispose();if(output)Object.values(output).forEach(t=>t.dispose());}if(refine&&!focus&&i===baseCropCount-1){const coarse=suppress(all),pitch=detectPitch(snapshot.getContext('2d')!.getImageData(0,0,scanWidth,scanHeight));crops.push(...refinementCrops(coarse.filter(d=>!pitchOnly||!pitch.reliable||pitch.score(d.box)>=.2),scanWidth,scanHeight));}await new Promise(resolve=>setTimeout(resolve,0));}
 const frame=snapshot.getContext('2d')!.getImageData(0,0,scanWidth,scanHeight);
 const pitch=detectPitch(frame);
 if(pitchOnly&&!pitch.reliable)onProgress('Pitch uncertain. Showing people for review; automatic roster admission is paused.');
 const raw=suppress(all).filter(d=>!focus||d.kind==='ball');
 const guide=jerseyGuide||new JerseyGuide();
 const guided=jerseys?.enabled&&!focus?guide.filter(frame,raw,pitch):raw;
 if(jerseys?.enabled&&!focus)onProgress(guide.status);
 const detections:TrackDetection[]=guided.filter(d=>d.kind==='ball'||!pitchOnly||!pitch.reliable||pitch.score(d.box)>=.2).map((d,i)=>{
  const appearance=d.kind==='person'?shirtFeature(frame,d.box):[];
  return {...d,id:'candidate-'+(i+1),appearance,fieldScore:pitch.score(d.box),fieldReliable:pitch.reliable,
   kit:d.kind==='person'?[...appearance,...shirtFeature(frame,{...d.box,y:d.box.y+d.box.h*.42,h:d.box.h*.4}),...shirtFeature(frame,{...d.box,y:d.box.y+d.box.h*.7,h:d.box.h*.3})]:[]};
 });
 const people=detections.filter(d=>d.kind==='person');
 for(const d of people)d.quality=cropQuality(frame,d.box,people.map(p=>p.box));
 if(embeddings&&!focus&&!appearanceUnavailable){
  const eligible=people.filter(d=>(d.quality||0)>.35).sort((a,b)=>(b.quality||0)-(a.quality||0)).slice(0,32);
  if(eligible.length){onProgress('Comparing player appearances…');try{const vectors=await reid.embed(snapshot,eligible.map(d=>d.box),cancelled);eligible.forEach((d,i)=>{d.embedding=vectors[i];});}catch(error){if(cancelled())throw error;appearanceUnavailable=true;console.warn('Re-ID unavailable; retaining detected boxes for review.',error);onProgress('People detected. Appearance matching unavailable; identities need confirmation.');}}
 }
 return detections;
}
