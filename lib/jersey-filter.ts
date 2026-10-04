import type {Box} from './tracking';
import type {Frame} from './patch-tracker';
import type {Detection} from './detection-core';
import type {PitchRegion} from './pitch-detector';

export type JerseyFilter={enabled:boolean};
export const defaultJerseyFilter: JerseyFilter={enabled:true};
type Shirt={hist:number[];bands:number[][]};
type Sample={shirt:Shirt;weight:number};
const clamp=(n:number)=>Math.max(0,Math.min(1,n));
const distance=(a:number[],b:number[])=>clamp(1-a.reduce((sum,v,i)=>sum+Math.sqrt(v*b[i]),0));

// Soft, circular hue bins avoid hard color boundaries. Dark navy and black
// share shadow evidence; pale/white shirts retain low-saturation evidence.
// No grass-color exclusion: either team may actually wear green.
function shirt(frame:Frame,box:Box):Shirt {
 const bands=Array.from({length:3},()=>Array<number>(14).fill(0));
 for(let y=0;y<12;y++)for(let x=0;x<8;x++){
  const px=Math.max(0,Math.min(frame.width-1,Math.floor((box.x+box.w*(.28+.44*(x+.5)/8))*frame.width)));
  const py=Math.max(0,Math.min(frame.height-1,Math.floor((box.y+box.h*(.18+.32*(y+.5)/12))*frame.height)));
  const i=(py*frame.width+px)*4,r=frame.data[i]/255,g=frame.data[i+1]/255,b=frame.data[i+2]/255;
  const hi=Math.max(r,g,b),lo=Math.min(r,g,b),delta=hi-lo,s=hi?delta/hi:0;
  const dark=clamp((.46-hi)/.24),neutral=(1-dark)*clamp((.3-s)/.22),chromatic=1-dark-neutral;
  let hue=delta?(hi===r?(g-b)/delta:hi===g?2+(b-r)/delta:4+(r-g)/delta):0;
  hue=((hue+6)%6)*2;
  const bin=Math.floor(hue),mix=hue-bin,hist=bands[Math.floor(y/4)];
  hist[bin]+=chromatic*(1-mix)/32;hist[(bin+1)%12]+=chromatic*mix/32;
  hist[12]+=dark/32;hist[13]+=neutral/32;
 }
 return {bands,hist:bands[0].map((_,i)=>bands.reduce((sum,b)=>sum+b[i],0)/3)};
}

function mean(samples:Sample[]){
 const weight=samples.reduce((sum,s)=>sum+s.weight,0);
 return Array.from({length:14},(_,i)=>samples.reduce((sum,s)=>sum+s.shirt.hist[i]*s.weight,0)/weight);
}

// One guide per video. Learn from several separate people in a frame, rather
// than counting the same person over successive frames as a whole team.
export class JerseyGuide {
 private kits:number[][]=[];
 reset(){this.kits=[];}
 get status(){return this.kits.length===2?'Two team colors learned; used as a loose guide.':'Learning both team colors from players on the pitch; keeping uncertain shirts visible.';}

 filter(frame:Frame,detections:Detection[],pitch:PitchRegion):Detection[]{
  const people=detections.filter(d=>d.kind==='person');
  const shirts=new Map(people.map(d=>[d,shirt(frame,d.box)]));
  const largest=Math.max(0,...people.filter(d=>d.score>=.3&&pitch.score(d.box)>=.65).map(d=>d.box.h));
  const samples:Sample[]=pitch.reliable?people.filter(d=>
   d.score>=.3&&d.box.h>=Math.max(.035,largest*.3)&&d.box.h*frame.height>=24&&
   d.box.w*frame.width>=8&&pitch.score(d.box)>=.65&&
   // Audience at the grass boundary must not dominate team-color learning.
   pitch.score({...d.box,y:d.box.y-d.box.h*.5})>=.45
  ).map(d=>({shirt:shirts.get(d)!,weight:d.score*Math.min(2,d.box.h/.06)}))
   .sort((a,b)=>b.weight-a.weight).slice(0,32):[];
  this.learn(samples);
  if(this.kits.length!==2)return detections;
  return detections.filter(d=>{
   if(d.kind==='ball')return true;
   const feature=shirts.get(d)!,field=pitch.score(d.box);
   // Keep clear nearby players even when numbers, folds or shadow change
   // their apparent kit. Color is never an identity assignment.
   if(d.box.h>=.08&&d.score>=.65||pitch.reliable&&field>=.65&&d.box.h>=.065&&d.score>=.25)return true;
   const whole=Math.min(...this.kits.map(k=>distance(feature.hist,k)));
   const matchingBands=feature.bands.filter(b=>this.kits.some(k=>distance(b,k)<.48)).length;
   if(whole<=.52||matchingBands>=2)return true;
   // Mixed or tiny shirt crops are inconclusive, not evidence of a spectator.
   if(d.box.h*frame.height<24||d.box.w*frame.width<7||Math.max(...feature.hist)<.3)return true;
   return false;
  });
 }

 private learn(samples:Sample[]){
  if(this.kits.length===2){
   // Adapt slowly using clear matches only. A referee or a new spectator
   // cannot replace a learned team color during a pan or a crowded frame.
   this.kits=this.kits.map((kit,index)=>{
    const matches=samples.filter(s=>distance(s.shirt.hist,kit)<.25&&distance(s.shirt.hist,this.kits[1-index])>distance(s.shirt.hist,kit)+.12);
    if(matches.length<2)return kit;
    const next=mean(matches);return kit.map((v,i)=>v*.98+next[i]*.02);
   });
   return;
  }
  if(samples.length<6)return;
  let best:{score:number;kits:number[][]}|undefined;
  // Find two supported modes; forcing every shirt into k=2 would promote a
  // lone referee, a goalkeeper or background clothing into a team color.
  for(let a=0;a<samples.length;a++)for(let b=a+1;b<samples.length;b++){
   if(distance(samples[a].shirt.hist,samples[b].shirt.hist)<.4)continue;
   const groups:Sample[][]=[[],[]];
   for(const sample of samples){
    const da=distance(sample.shirt.hist,samples[a].shirt.hist),db=distance(sample.shirt.hist,samples[b].shirt.hist);
    if(Math.min(da,db)>.28||Math.abs(da-db)<.1)continue;
    groups[da<db?0:1].push(sample);
   }
   if(groups.some(g=>g.length<3))continue;
   const kits=groups.map(mean);if(distance(kits[0],kits[1])<.35)continue;
   const weights=groups.map(g=>g.reduce((sum,s)=>sum+s.weight,0));
   const score=weights[0]+weights[1]+Math.min(...weights)*.5;
   if(!best||score>best.score)best={score,kits};
  }
  if(best)this.kits=best.kits;
 }
}
