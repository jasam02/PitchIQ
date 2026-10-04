import type {InferenceSession} from 'onnxruntime-web';
import type {Box} from './tracking';
import type {Frame} from './patch-tracker';
import {iou} from './detection-core';

export function cosine(a:number[],b:number[]) {
 if(a.length!==512||b.length!==512)return 0;
 let dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}
 return aa&&bb?Math.max(0,Math.min(1,dot/Math.sqrt(aa*bb))):0;
}
export function quantize(embedding:number[]){const max=Math.max(...embedding.map(Math.abs),1e-9);return embedding.map(v=>Math.round(v/max*127));}
export function cropQuality(frame:Frame,box:Box,others:Box[]) {
 if(box.h*frame.height<48||box.w*frame.width<16||box.x<.002||box.y<.002||box.x+box.w>.998||box.y+box.h>.998)return 0;
 const occlusion=Math.max(0,...others.filter(b=>b!==box).map(b=>iou(box,b)));
 if(occlusion>.25)return 0;
 let edges=0,n=0;
 const grey=(x:number,y:number)=>{const i=(y*frame.width+x)*4;return .299*frame.data[i]+.587*frame.data[i+1]+.114*frame.data[i+2];};
 for(let y=Math.ceil((box.y+box.h*.1)*frame.height);y<Math.min(frame.height-1,(box.y+box.h*.9)*frame.height);y+=3)
  for(let x=Math.ceil((box.x+box.w*.2)*frame.width);x<Math.min(frame.width-1,(box.x+box.w*.8)*frame.width);x+=3){edges+=Math.abs(grey(x,y)-grey(x+1,y))+Math.abs(grey(x,y)-grey(x,y+1));n++;}
 return Math.min(1,box.h*frame.height/100)*Math.min(1,edges/Math.max(1,n)/12)*(1-occlusion);
}
let loading:Promise<InferenceSession>|undefined;
async function model(){
 if(!loading)loading=(async()=>{
  const ort=await import('onnxruntime-web/wasm');ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths=new URL('/ort/',location.origin).href;
  const response=await fetch('/models/osnet-x025.onnx',{signal:AbortSignal.timeout(60000)});
  if(!response.ok)throw Error('Player appearance model could not load.');
  return ort.InferenceSession.create(await response.arrayBuffer(),{executionProviders:['wasm'],graphOptimizationLevel:'all'});
 })().catch(e=>{loading=undefined;throw e;});
 return loading;
}
// OSNet x0.25 MSMT17, RGB 256x128, ImageNet mean/std, L2-normalized 512-D
// descriptor. The model runs locally; crops never leave the browser.
export class PlayerReID {
 private canvas?:HTMLCanvasElement;
 async embed(source:HTMLVideoElement|HTMLCanvasElement,boxes:Box[],cancelled=()=>false){
  const session=await model(),ort=await import('onnxruntime-web/wasm');
  this.canvas??=document.createElement('canvas');this.canvas.width=128;this.canvas.height=256;
  const ctx=this.canvas.getContext('2d',{willReadFrequently:true})!,width='videoWidth'in source?source.videoWidth:source.width,height='videoHeight'in source?source.videoHeight:source.height;
  const result:number[][]=[];
  // This pinned export has a fixed batch of 16. Pad the last batch and
  // discard padded outputs; never run 16 copies of the network per person.
  const plane=128*256;
  for(let offset=0;offset<boxes.length;offset+=16){
   if(cancelled())throw Error('Tracking stopped.');
   const batch=boxes.slice(offset,offset+16),input=new Float32Array(16*3*plane),mean=[.485,.456,.406],std=[.229,.224,.225];
   for(let n=0;n<batch.length;n++){
    const box=batch[n];ctx.drawImage(source,box.x*width,box.y*height,box.w*width,box.h*height,0,0,128,256);
    const pixels=ctx.getImageData(0,0,128,256).data;
    for(let i=0;i<plane;i++)for(let c=0;c<3;c++)input[n*3*plane+c*plane+i]=(pixels[i*4+c]/255-mean[c])/std[c];
   }
   const tensor=new ort.Tensor('float32',input,[16,3,256,128]);
   try{const outputs=await session.run({[session.inputNames[0]]:tensor});try{
    const values=outputs[session.outputNames[0]].data as Float32Array;
    if(values.length!==16*512)throw Error('Unexpected Re-ID model output.');
    for(let n=0;n<batch.length;n++){
     const row=Array.from(values.slice(n*512,(n+1)*512));
     if(row.some(v=>!Number.isFinite(v)))throw Error('Invalid Re-ID embedding.');
     const norm=Math.hypot(...row)||1;result.push(row.map(v=>v/norm));
    }
   }finally{Object.values(outputs).forEach(t=>t.dispose());}}finally{tensor.dispose();}
   await new Promise(resolve=>setTimeout(resolve,0));
  }
  return result;
 }
}
