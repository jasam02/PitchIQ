import type {Box} from './tracking';
import type {Frame} from './patch-tracker';
export type CameraMotion={scale:number;dx:number;dy:number;reliable:boolean;cut:boolean};
export const stillCamera:CameraMotion={scale:1,dx:0,dy:0,reliable:false,cut:false};
const grass=(r:number,g:number,b:number)=>g>r*1.08&&g>b*1.15&&g>35;
export function shirtFeature(frame:Frame,box:Box):number[]{
 const hist=Array(50).fill(0);let count=0;
 for(let y=0;y<12;y++)for(let x=0;x<8;x++){
  const px=Math.min(frame.width-1,Math.max(0,Math.floor((box.x+box.w*(.2+.6*(x+.5)/8))*frame.width)));
  const py=Math.min(frame.height-1,Math.max(0,Math.floor((box.y+box.h*(.18+.45*(y+.5)/12))*frame.height)));
  const i=(py*frame.width+px)*4,r=frame.data[i]/255,g=frame.data[i+1]/255,b=frame.data[i+2]/255;
  if(grass(r*255,g*255,b*255))continue;
  const max=Math.max(r,g,b),min=Math.min(r,g,b),delta=max-min;let bin=48;
  if(max<.32)bin=48;else if(delta<.12)bin=49;else{let hue=max===r?(g-b)/delta:max===g?2+(b-r)/delta:4+(r-g)/delta;hue=(hue+6)%6;bin=Math.min(23,Math.floor(hue*4))*2+(max>.65?1:0);}
  hist[bin]++;count++;
 }
 return count>=10?hist.map(v=>v/count):[];
}
export function appearanceDistance(a:number[],b:number[]){return !a.length||!b.length?0.25:1-a.reduce((s,v,i)=>s+Math.sqrt(v*(b[i]||0)),0);}
// A long, nearly horizontal white boundary in the lower image can separate
// the pitch from the technical area. If evidence is weak, retain grass-only filtering.
function nearTouchline(frame:Frame){
 const stride=Math.max(2,Math.round(frame.width/240)),bin=Math.max(3,Math.round(frame.height/135));
 const slopes=Array.from({length:25},(_,i)=>(i-12)*.01),votes=slopes.map(()=>new Map<number,Set<number>>());
 const isGrass=(x:number,y:number)=>{if(y<0||y>=frame.height)return false;const i=(y*frame.width+x)*4;return grass(frame.data[i],frame.data[i+1],frame.data[i+2]);};
 for(let x=0;x<frame.width;x+=stride)for(let y=Math.floor(frame.height*.6);y<frame.height-8;y+=2){const i=(y*frame.width+x)*4,r=frame.data[i],g=frame.data[i+1],b=frame.data[i+2];if(Math.min(r,g,b)<160||Math.max(r,g,b)-Math.min(r,g,b)>65||!isGrass(x,y-8)||!isGrass(x,y+8))continue;
  for(let j=0;j<slopes.length;j++){const intercept=Math.round((y-slopes[j]*x)/bin);let columns=votes[j].get(intercept);if(!columns){columns=new Set();votes[j].set(intercept,columns);}columns.add(x);}
 }
 let best:{slope:number;intercept:number;count:number}|undefined;
 for(let j=0;j<slopes.length;j++)for(const [intercept,columns] of votes[j])if(columns.size>frame.width/stride*.52&&Math.max(...columns)-Math.min(...columns)>frame.width*.7&&(!best||columns.size>best.count))best={slope:slopes[j],intercept:intercept*bin,count:columns.size};
 return best;
}
// Largest connected grass region, with small holes filled within each row.
// Recomputed each frame so the filter follows camera movement; not a field calibration.
export function pitchMask(frame:Frame){
 const width=160,height=90,green=new Uint8Array(width*height),seen=new Uint8Array(width*height);let largest:number[]=[];
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=(Math.floor((y+.5)*frame.height/height)*frame.width+Math.floor((x+.5)*frame.width/width))*4;green[y*width+x]=grass(frame.data[i],frame.data[i+1],frame.data[i+2])?1:0;}
 for(let i=0;i<green.length;i++){if(!green[i]||seen[i])continue;const component=[i];seen[i]=1;for(let j=0;j<component.length;j++){const q=component[j],x=q%width;for(const n of [x>0?q-1:-1,x<width-1?q+1:-1,q-width,q+width])if(n>=0&&n<green.length&&green[n]&&!seen[n]){seen[n]=1;component.push(n);}}if(component.length>largest.length)largest=component;}
 const left=Array(height).fill(width),right=Array(height).fill(-1);for(const q of largest){const y=Math.floor(q/width),x=q%width;left[y]=Math.min(left[y],x);right[y]=Math.max(right[y],x);}
 const reliable=largest.length>width*height*.12,boundary=reliable?nearTouchline(frame):undefined;
 return {reliable,contains:(box:Box)=>{if(!reliable)return true;if(boundary&&(box.y+box.h)*frame.height>boundary.slope*(box.x+box.w/2)*frame.width+boundary.intercept+Math.max(8,box.h*frame.height*.12))return false;const x=(box.x+box.w/2)*width,y=Math.floor((box.y+box.h)*height);for(let row=Math.max(0,y-2);row<=Math.min(height-1,y+2);row++)if(x>=left[row]-2&&x<=right[row]+2)return true;return false;}};
}
function grey(f:Frame){const a=new Float32Array(f.width*f.height);for(let i=0;i<a.length;i++)a[i]=f.data[i*4]*.299+f.data[i*4+1]*.587+f.data[i*4+2]*.114;return a;}
// Track textured background patches and robustly fit scale + translation.
// Conservative fallback when matches do not agree; not a projective homography.
export function estimateCamera(previous:Frame,current:Frame,people:Box[]=[]):CameraMotion{
 if(previous.width!==current.width||previous.height!==current.height)return stillCamera;
 const w=current.width,h=current.height,a=grey(previous),b=grey(current);const pairs:{x:number;y:number;u:number;v:number}[]=[];let change=0;for(let i=0;i<a.length;i++)change+=Math.abs(a[i]-b[i]);change/=a.length;
 for(let y=16;y<h-16;y+=20)for(let x=16;x<w-16;x+=24){if(people.some(p=>x/w>=p.x-.02&&x/w<=p.x+p.w+.02&&y/h>=p.y-.02&&y/h<=p.y+p.h+.02))continue;let mean=0,variance=0;for(let j=-3;j<=3;j++)for(let i=-3;i<=3;i++)mean+=a[(y+j)*w+x+i];mean/=49;for(let j=-3;j<=3;j++)for(let i=-3;i<=3;i++)variance+=(a[(y+j)*w+x+i]-mean)**2;if(variance/49<100)continue;
  const costs:{dx:number;dy:number;cost:number}[]=[];for(let dy=-12;dy<=12;dy++)for(let dx=-12;dx<=12;dx++){let cost=0;for(let j=-3;j<=3;j+=2)for(let i=-3;i<=3;i+=2)cost+=Math.abs(a[(y+j)*w+x+i]-b[(y+j+dy)*w+x+i+dx]);costs.push({dx,dy,cost:cost/16});}costs.sort((c,d)=>c.cost-d.cost);const best=costs[0],second=costs.find(c=>Math.hypot(c.dx-best.dx,c.dy-best.dy)>3);if(best.cost<18&&second&&second.cost-best.cost>1.5)pairs.push({x:x/w,y:y/h,u:(x+best.dx)/w,v:(y+best.dy)/h});
 }
 let best=stillCamera,bestInliers=0;
 const fit=(scale:number,dx:number,dy:number)=>{if(scale<.92||scale>1.08)return;const count=pairs.filter(p=>Math.hypot((p.u-(p.x*scale+dx))*w,(p.v-(p.y*scale+dy))*h)<2.5).length;if(count>bestInliers){bestInliers=count;best={scale,dx,dy,reliable:true,cut:false};}};
 for(let i=0;i<pairs.length;i++){const p=pairs[i];fit(1,p.u-p.x,p.v-p.y);for(let j=i+1;j<pairs.length;j++){const q=pairs[j],xx=p.x-q.x,yy=(p.y-q.y)*h/w,uu=p.u-q.u,vv=(p.v-q.v)*h/w,den=xx*xx+yy*yy;if(den<.03)continue;const scale=(xx*uu+yy*vv)/den;fit(scale,p.u-scale*p.x,p.v-scale*p.y);}}
 if(bestInliers>=6&&bestInliers>=pairs.length*.4)return best;
 return {...stillCamera,cut:change>38&&bestInliers<4};
}
