// Part-based colour appearance for soccer re-identification.
// jersey (upper torso), shorts, socks: soft HSV histograms with achromatic (black/grey/white) bins;
// layout: coarse 2x4 grid of mean colours of the whole body. Grass pixels are never sampled.
// Colour cannot separate same-kit teammates; spatial/temporal evidence does that (identity.ts).
// descriptorDistance is the single comparison seam: a learned re-ID embedding can be blended in there.
import type {Box,Descriptor,Frame,GallerySample,PartDistance} from './types';

type Spec={hues:number;levels:number;greys:number[]};
const JERSEY:Spec={hues:10,levels:2,greys:[.1,.38,.65,.92]}; // 4 achromatic + 10 hues x 2 brightness = 24
const SHORTS:Spec={hues:9,levels:1,greys:[.1,.45,.85]}; // 3 + 9 = 12
const SOCKS:Spec={hues:5,levels:1,greys:[.1,.45,.85]}; // 3 + 5 = 8
const bins=(s:Spec)=>s.greys.length+s.hues*s.levels;
const LAYOUT_COLS=2,LAYOUT_ROWS=4,LAYOUT_LEN=LAYOUT_COLS*LAYOUT_ROWS*3; // mean (L, r-g, r+g-2b) per cell; -1 = no body pixels
export const PART_LENGTHS={jersey:bins(JERSEY),shorts:bins(SHORTS),socks:bins(SOCKS),layout:LAYOUT_LEN};
// [quality, jersey(24), shorts(12), socks(8), layout(24)]
export const DESCRIPTOR_LENGTH=69; // length of encodeDescriptor output (<=80)
export const MIN_GALLERY_QUALITY=.45;
export const GALLERY_SPACING=.8; // seconds between gallery samples
export const PART_WEIGHTS={jersey:.45,shorts:.2,socks:.1,layout:.25};

const clamp=(v:number,a=0,b=1)=>v<a?a:v>b?b:v;
const r3=(v:number)=>Math.round(v*1000)/1000;
const isGrass=(r:number,g:number,b:number)=>g>r*1.08&&g>b*1.15&&g>35;
const sameBox=(a:Box,b:Box)=>a===b||(a.x===b.x&&a.y===b.y&&a.w===b.w&&a.h===b.h);
const overlapArea=(a:Box,b:Box)=>Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));

// Soft binning so small hue/lighting changes move mass between neighbouring bins instead of flipping bins.
function addColour(h:number[],s:Spec,r:number,g:number,b:number){
 const max=Math.max(r,g,b),min=Math.min(r,g,b),c=max-min;
 const chroma=clamp((c-.06)/.12)*clamp((max-.1)/.12),grey=1-chroma,gs=s.greys;
 if(grey>0){
  const v=(max+min)/2;let k=0;while(k<gs.length-1&&v>gs[k+1])k++;
  if(v<=gs[0]||k===gs.length-1)h[k]+=grey;else{const t=(v-gs[k])/(gs[k+1]-gs[k]);h[k]+=grey*(1-t);h[k+1]+=grey*t;}
 }
 if(chroma>0){
  let hue=max===r?(g-b)/c:max===g?2+(b-r)/c:4+(r-g)/c;hue=(hue+6)%6;
  const p=hue/6*s.hues-.5,i0=Math.floor(p),t=p-i0,a=(i0+s.hues)%s.hues,n=(i0+1)%s.hues,off=gs.length;
  if(s.levels===1){h[off+a]+=chroma*(1-t);h[off+n]+=chroma*t;return;}
  const lv=clamp((max-.2)/.6);
  h[off+a*2]+=chroma*(1-t)*(1-lv);h[off+a*2+1]+=chroma*(1-t)*lv;h[off+n*2]+=chroma*t*(1-lv);h[off+n*2+1]+=chroma*t*lv;
 }
}
const normalized=(h:number[])=>{const s=h.reduce((a,v)=>a+v,0);return s>0?h.map(v=>r3(v/s)):h.map(()=>0);};
const emptyDescriptor=():Descriptor=>({jersey:Array(PART_LENGTHS.jersey).fill(0),shorts:Array(PART_LENGTHS.shorts).fill(0),socks:Array(PART_LENGTHS.socks).fill(0),layout:Array(LAYOUT_LEN).fill(-1),quality:0});

// `others` are neighbouring person boxes used to estimate occlusion.
// Samples a bounded grid per part (<= 16x20 jersey, 16x10 shorts, 16x8 socks, 8x16 layout, 12x16 sharpness).
export function describe(frame:Frame,box:Box,others:Box[]=[]):Descriptor{
 const W=frame.width,H=frame.height,data=frame.data;
 if(!W||!H||!(box.w>0)||!(box.h>0))return emptyDescriptor();
 const bx=box.x*W,by=box.y*H,bw=box.w*W,bh=box.h*H;
 if(bx+bw<=0||by+bh<=0||bx>=W||by>=H)return emptyDescriptor();
 // People whose feet are lower are in front: their pixels are skipped and their overlap costs quality.
 const near=others.filter(o=>!sameBox(o,box)&&overlapArea(o,box)>0);
 const front=near.filter(o=>o.y+o.h>box.y+box.h+1e-6);
 const hidden=(px:number,py:number)=>front.some(o=>px>=o.x*W&&px<(o.x+o.w)*W&&py>=o.y*H&&py<(o.y+o.h)*H);
 const at=(px:number,py:number)=>(Math.min(H-1,Math.max(0,Math.floor(py)))*W+Math.min(W-1,Math.max(0,Math.floor(px))))*4;
 const part=(spec:Spec,u0:number,u1:number,v0:number,v1:number,maxX:number,maxY:number)=>{
  const h=Array(bins(spec)).fill(0),nx=clamp(Math.round((u1-u0)*bw),3,maxX),ny=clamp(Math.round((v1-v0)*bh),3,maxY);let used=0,visible=0;
  for(let y=0;y<ny;y++){const py=by+(v0+(v1-v0)*(y+.5)/ny)*bh;if(py<0||py>=H)continue;
   for(let x=0;x<nx;x++){const px=bx+(u0+(u1-u0)*(x+.5)/nx)*bw;if(px<0||px>=W||hidden(px,py))continue;visible++;
    const i=at(px,py),r=data[i],g=data[i+1],b=data[i+2];if(isGrass(r,g,b))continue;addColour(h,spec,r/255,g/255,b/255);used++;}}
  return {h:used>=Math.max(3,visible*.12)?normalized(h):h.map(()=>0),frac:visible?used/visible:0};
 };
 const jersey=part(JERSEY,.25,.75,.18,.45,16,20),shorts=part(SHORTS,.2,.8,.5,.68,16,10),socks=part(SOCKS,.12,.88,.75,.92,16,8);
 // Coarse layout of the body (whole box minus a thin border).
 const layout:number[]=[],sx=clamp(Math.round(bw*.8/LAYOUT_COLS),1,4),sy=clamp(Math.round(bh*.96/LAYOUT_ROWS),1,4);let bodyUsed=0,bodyVisible=0;
 for(let row=0;row<LAYOUT_ROWS;row++)for(let col=0;col<LAYOUT_COLS;col++){
  let n=0,L=0,o1=0,o2=0,seen=0;
  for(let j=0;j<sy;j++)for(let i=0;i<sx;i++){
   const px=bx+(.1+.8*(col+(i+.5)/sx)/LAYOUT_COLS)*bw,py=by+(.02+.96*(row+(j+.5)/sy)/LAYOUT_ROWS)*bh;
   if(px<0||px>=W||py<0||py>=H||hidden(px,py))continue;seen++;
   const k=at(px,py),r=data[k]/255,g=data[k+1]/255,b=data[k+2]/255;if(isGrass(data[k],data[k+1],data[k+2]))continue;
   L+=(r+g+b)/3;o1+=(r-g)/2+.5;o2+=(r+g-2*b)/4+.5;n++;
  }
  bodyUsed+=n;bodyVisible+=seen;
  if(n>=Math.max(1,Math.ceil(seen*.25)))layout.push(r3(L/n),r3(o1/n),r3(o2/n));else layout.push(-1,-1,-1);
 }
 // Sharpness: largest 1-px step vs the 3-px step over the same span; ~1 for crisp edges, ~1/3 for blurred ramps.
 const lum=(px:number,py:number)=>{const i=at(px,py);return data[i]*.299+data[i+1]*.587+data[i+2]*.114;};
 const gx=clamp(Math.round(bw*.6),2,12),gy=clamp(Math.round(bh*.8),2,16);let fine=0,coarse=0,pts=0;
 for(let y=0;y<gy;y++)for(let x=0;x<gx;x++){
  const px=Math.floor(bx+(.2+.6*(x+.5)/gx)*bw),py=Math.floor(by+(.1+.8*(y+.5)/gy)*bh);if(px<0||px>=W-3||py<0||py>=H-3)continue;
  const h=[0,1,2,3].map(k=>lum(px+k,py)),v=[0,1,2,3].map(k=>lum(px,py+k));
  fine+=Math.max(Math.abs(h[1]-h[0]),Math.abs(h[2]-h[1]),Math.abs(h[3]-h[2]))+Math.max(Math.abs(v[1]-v[0]),Math.abs(v[2]-v[1]),Math.abs(v[3]-v[2]));
  coarse+=Math.abs(h[3]-h[0])+Math.abs(v[3]-v[0]);pts++;
 }
 const sharpF=!pts||coarse<pts*2?.9:.75+.25*clamp((fine/coarse-.5)/.45);
 // Quality: size, occlusion, truncation, enough body pixels, aspect ratio, sharpness.
 const visH=Math.min(H,by+bh)-Math.max(0,by),visW=Math.min(W,bx+bw)-Math.max(0,bx);
 const sizeF=clamp((visH-16)/56)**.6;
 const area=box.w*box.h;let occ=0;for(const o of near)occ+=overlapArea(o,box)/area*(front.includes(o)?1:.5);
 const occF=clamp(1-1.4*Math.min(1,occ));
 const e=.002,truncF=(box.x<=e?.55:1)*(box.x+box.w>=1-e?.55:1)*(box.y<=e?.75:1)*(box.y+box.h>=1-e?.5:1);
 const pixF=clamp((jersey.frac-.2)/.3)*clamp((bodyVisible?bodyUsed/bodyVisible:0)/.25);
 const ratio=visH/Math.max(1,visW),aspF=ratio<1.3?clamp((ratio-.6)/.7):ratio>5.5?clamp((9-ratio)/3.5):1;
 const quality=r3(clamp(sizeF*occF*truncF*pixF*aspF*sharpF));
 return {jersey:jersey.h,shorts:shorts.h,socks:socks.h,layout,quality};
}

export function encodeDescriptor(d:Descriptor):number[]{
 const fit=(v:number[],n:number,fill:number)=>Array.from({length:n},(_,i)=>Number.isFinite(v[i])?r3(v[i]):fill);
 return [r3(clamp(d.quality)),...fit(d.jersey,PART_LENGTHS.jersey,0),...fit(d.shorts,PART_LENGTHS.shorts,0),...fit(d.socks,PART_LENGTHS.socks,0),...fit(d.layout,LAYOUT_LEN,-1)];
}
export function decodeDescriptor(v:number[]):Descriptor|undefined{
 if(!Array.isArray(v)||v.length!==DESCRIPTOR_LENGTH||!v.every(Number.isFinite))return undefined;
 const j=1+PART_LENGTHS.jersey,s=j+PART_LENGTHS.shorts,k=s+PART_LENGTHS.socks;
 return {quality:clamp(v[0]),jersey:v.slice(1,j),shorts:v.slice(j,s),socks:v.slice(s,k),layout:v.slice(k)};
}

// Bhattacharyya (Hellinger) distance of two histograms, 0 identical .. 1 disjoint; undefined when either is empty.
export function bhattacharyya(a:number[],b:number[]):number|undefined{
 let sa=0,sb=0,bc=0;const n=Math.max(a.length,b.length);
 for(let i=0;i<n;i++){const x=Math.max(0,a[i]||0),y=Math.max(0,b[i]||0);sa+=x;sb+=y;bc+=Math.sqrt(x*y);}
 if(sa<=1e-9||sb<=1e-9)return undefined;
 return Math.sqrt(clamp(1-bc/Math.sqrt(sa*sb)));
}
function layoutDistance(a:number[],b:number[]){
 let sum=0,n=0;
 for(let i=0;i+2<Math.min(a.length,b.length);i+=3){if(a[i]<0||b[i]<0)continue;sum+=Math.min(1,Math.hypot(a[i]-b[i],a[i+1]-b[i+1],a[i+2]-b[i+2])/.5);n++;}
 return n?sum/n:undefined;
}
// Missing parts (no usable pixels) report .5 and are left out of the weighted total.
export function descriptorDistance(a:Descriptor,b:Descriptor):PartDistance{
 const parts={jersey:bhattacharyya(a.jersey,b.jersey),shorts:bhattacharyya(a.shorts,b.shorts),socks:bhattacharyya(a.socks,b.socks),layout:layoutDistance(a.layout,b.layout)};
 let total=0,weight=0;
 for(const k of Object.keys(PART_WEIGHTS) as (keyof typeof PART_WEIGHTS)[]){const v=parts[k];if(v!==undefined){total+=PART_WEIGHTS[k]*v;weight+=PART_WEIGHTS[k];}}
 const or=(v:number|undefined)=>v===undefined?.5:r3(v);
 return {total:weight?r3(total/weight):.5,jersey:or(parts.jersey),shorts:or(parts.shorts),socks:or(parts.socks),layout:or(parts.layout)};
}

// Quality- and diversity-aware gallery update. Low-quality samples are ignored.
// Samples stay >= GALLERY_SPACING apart; a much better sample may replace a close one. When full, the sample
// with the lowest quality + diversity value is dropped (diversity capped so one odd sample cannot dominate).
export function addToGallery(gallery:GallerySample[],d:Descriptor,time:number,max=5):GallerySample[]{
 let g=gallery.slice();
 if(d.quality>=MIN_GALLERY_QUALITY&&Number.isFinite(time)){
  const close=g.filter(s=>Math.abs(s.time-time)<GALLERY_SPACING);
  if(close.length===1&&d.quality>=close[0].d.quality+.1)g=g.filter(s=>s!==close[0]).concat({d,time});
  else if(!close.length)g.push({d,time});
 }
 while(g.length>Math.max(0,max)){
  const value=g.map((s,i)=>{let m=1;g.forEach((o,j)=>{if(j!==i)m=Math.min(m,descriptorDistance(s.d,o.d).total);});return s.d.quality+1.5*Math.min(m,.3);});
  let worst=0;for(let i=1;i<g.length;i++)if(value[i]<value[worst]-1e-9||(Math.abs(value[i]-value[worst])<=1e-9&&g[i].time<g[worst].time))worst=i;
  g.splice(worst,1);
 }
 return g.sort((a,b)=>a.time-b.time);
}

// Robust comparison of several probe descriptors against several gallery samples:
// mean of the best two pair distances (per part as well). Poor probes are skipped when better ones exist.
export function galleryDistance(gallery:GallerySample[],probes:Descriptor[]):PartDistance|undefined{
 const good=probes.filter(p=>p.quality>=.25),use=good.length?good:probes;
 const pairs:PartDistance[]=[];for(const s of gallery)for(const p of use)pairs.push(descriptorDistance(s.d,p));
 if(!pairs.length)return undefined;
 const best=(k:keyof PartDistance)=>{const v=pairs.map(p=>p[k]).sort((a,b)=>a-b).slice(0,2);return r3(v.reduce((a,b)=>a+b,0)/v.length);};
 return {total:best('total'),jersey:best('jersey'),shorts:best('shorts'),socks:best('socks'),layout:best('layout')};
}
