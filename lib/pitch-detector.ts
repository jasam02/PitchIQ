import type {Box} from './tracking';
import type {Frame} from './patch-tracker';

export type PitchRegion={reliable:boolean;confidence:number;polygon:{x:number;y:number}[];score:(box:Box)=>number};
const W=160,H=90;
// Largest connected grass component with a bounded closing operation. Unlike
// row envelopes, this cannot bridge a large stand/advertising-board intrusion.
// Uncertain masks withhold automatic roster admission, not person detections.
export function detectPitch(frame:Frame):PitchRegion {
 const cells=new Uint8Array(W*H),seen=new Uint8Array(W*H);
 for(let y=0;y<H;y++)for(let x=0;x<W;x++){
  let count=0;
  for(const [dx,dy] of [[.25,.25],[.75,.25],[.25,.75],[.75,.75]]){
   const px=Math.min(frame.width-1,Math.floor((x+dx)*frame.width/W)),py=Math.min(frame.height-1,Math.floor((y+dy)*frame.height/H)),i=(py*frame.width+px)*4;
   const r=frame.data[i],g=frame.data[i+1],b=frame.data[i+2];
   // Worn/yellow turf often has R >= G. Blue suppression and chroma retain
   // those surfaces while excluding sky, white paint, and neutral concrete.
   if(g>22&&g>r*.88&&g>b*1.08&&Math.max(r,g)-b>8)count++;
  }
  cells[y*W+x]=count>=2?1:0;
 }
 // Close narrow gaps BEFORE connected components. Otherwise the halfway
 // line divides the field and the largest-component step discards one half.
 const bridged=cells.slice();
 for(let y=0;y<H;y++)for(let x=0;x<W;x++)if(cells[y*W+x]){
  for(let gap=1;gap<=3;gap++){
   if(x+gap+1<W&&cells[y*W+x+gap+1])for(let k=1;k<=gap;k++)bridged[y*W+x+k]=1;
   if(y+gap+1<H&&cells[(y+gap+1)*W+x])for(let k=1;k<=gap;k++)bridged[(y+k)*W+x]=1;
  }
 }
 cells.set(bridged);
 let largest:number[]=[];
 for(let i=0;i<cells.length;i++)if(cells[i]&&!seen[i]){
  const region=[i];seen[i]=1;
  for(let j=0;j<region.length;j++){const q=region[j],x=q%W;for(const n of [x?q-1:-1,x<W-1?q+1:-1,q-W,q+W])if(n>=0&&n<cells.length&&cells[n]&&!seen[n]){seen[n]=1;region.push(n);}}
  if(region.length>largest.length)largest=region;
 }
 const mask=new Uint8Array(W*H);for(const i of largest)mask[i]=1;
 const coverage=largest.length/(W*H),reliable=coverage>.15;
 // Fill only narrow holes (occluding bodies and painted lines), not full rows.
 for(let y=0;y<H;y++)for(let x=1;x<W-1;x++)if(!mask[y*W+x]){
  for(let d=1;d<=4&&x+d<W;d++)if(mask[y*W+x-1]&&mask[y*W+x+d]){for(let k=0;k<d;k++)mask[y*W+x+k]=1;break;}
 }
 const polygon:{x:number;y:number}[]=[],right:{x:number;y:number}[]=[];
 for(let y=0;y<H;y+=3){let lo=W,hi=-1;for(let x=0;x<W;x++)if(mask[y*W+x]){lo=Math.min(lo,x);hi=x;}if(hi>=lo){polygon.push({x:lo/W,y:y/H});right.push({x:(hi+1)/W,y:y/H});}}
 polygon.push(...right.reverse());
 return {reliable,confidence:reliable?Math.min(1,coverage*2):0,polygon,score(box){
  if(!reliable)return 0;
  const cx=(box.x+box.w/2)*W,fy=(box.y+box.h)*H;let hit=0,total=0;
  // Footpoint neighborhood allows ~2% sideline tolerance, including a foot
  // occluded by another body. Green shirts alone cannot satisfy this test.
  for(let dy=-1;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
   const x=Math.round(cx+dx),y=Math.round(fy+dy);if(x<0||x>=W||y<0||y>=H)continue;
   total++;hit+=mask[y*W+x];
  }
  return total?hit/total:0;
 }};
}
