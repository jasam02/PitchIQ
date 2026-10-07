// Image -> pitch plane homography from user field marks (normalized DLT with Hartley normalization).
// Honest limit: there is no automatic metric calibration; without marks pitch coordinates are absent and
// camera-compensated image coordinates are used for spatial reasoning instead.
import type {FieldCamera,FieldMark} from '../tracking';
import {projectFieldMark} from '../field-reference';
import {footPoint} from './pitch';
import type {Box,Pt} from './types';
// 3x3 row-major. Maps normalized image coordinates to normalized pitch coordinates:
// x 0..1 along the touchline (left goal line -> right goal line), y 0..1 far touchline -> near touchline.
// Matrices from solveHomography are scaled so w > 0 over the source points (w <= 0 = beyond the horizon).
export type Homography=number[];
export const PITCH_LENGTH=105,PITCH_WIDTH=68;
export const PITCH_SLACK=.15; // toPitch accepts [-.15, 1.15] on both axes
export const pitchMeters=(p:Pt):Pt=>({x:p.x*PITCH_LENGTH,y:p.y*PITCH_WIDTH});

const mul3=(a:number[],b:number[])=>{const o=Array(9).fill(0);for(let r=0;r<3;r++)for(let c=0;c<3;c++)o[r*3+c]=a[r*3]*b[c]+a[r*3+1]*b[3+c]+a[r*3+2]*b[6+c];return o;};
const det3=(m:number[])=>m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]);
function inv3(m:number[]):number[]|undefined{
 const d=det3(m),scale=Math.max(...m.map(Math.abs));if(!Number.isFinite(d)||!(scale>0)||Math.abs(d)<1e-14*scale**3)return undefined;
 const a=[m[4]*m[8]-m[5]*m[7],m[2]*m[7]-m[1]*m[8],m[1]*m[5]-m[2]*m[4],m[5]*m[6]-m[3]*m[8],m[0]*m[8]-m[2]*m[6],m[2]*m[3]-m[0]*m[5],m[3]*m[7]-m[4]*m[6],m[1]*m[6]-m[0]*m[7],m[0]*m[4]-m[1]*m[3]];
 return a.map(v=>v/d);
}
// Hartley normalization: centroid to the origin, mean distance sqrt(2).
function normalize(pts:Pt[]){
 let cx=0,cy=0;for(const p of pts){cx+=p.x;cy+=p.y;}cx/=pts.length;cy/=pts.length;
 let md=0;for(const p of pts)md+=Math.hypot(p.x-cx,p.y-cy);md/=pts.length;if(!(md>1e-9))return undefined;
 const s=Math.SQRT2/md;return {t:[s,0,-s*cx,0,s,-s*cy,0,0,1],inv:[1/s,0,cx,0,1/s,cy,0,0,1],q:pts.map(p=>({x:(p.x-cx)*s,y:(p.y-cy)*s})),scale:md};
}
// Some 4 points with no 3 (nearly) collinear must exist, otherwise H is not determined.
function generalPosition(q:Pt[]){
 const area=(a:Pt,b:Pt,c:Pt)=>Math.abs((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)),ok=(a:Pt,b:Pt,c:Pt)=>area(a,b,c)>2e-2,n=q.length;
 if(n>12){let sxx=0,syy=0,sxy=0;for(const p of q){sxx+=p.x*p.x;syy+=p.y*p.y;sxy+=p.x*p.y;}const tr=sxx+syy,dt=sxx*syy-sxy*sxy,lo=tr/2-Math.sqrt(Math.max(0,tr*tr/4-dt));return lo/tr>1e-3;}
 for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)for(let k=j+1;k<n;k++){if(!ok(q[i],q[j],q[k]))continue;for(let l=k+1;l<n;l++)if(ok(q[i],q[j],q[l])&&ok(q[i],q[k],q[l])&&ok(q[j],q[k],q[l]))return true;}
 return false;
}
// Cyclic Jacobi eigen-decomposition of a symmetric matrix (here 9x9 AᵀA). Columns of v are eigenvectors.
function jacobi(a:number[][]){
 const n=a.length,v:number[][]=a.map((_,i)=>a.map((__,j)=>i===j?1:0));
 for(let sweep=0;sweep<60;sweep++){
  let off=0;for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)off+=a[i][j]*a[i][j];if(off<1e-30)break;
  for(let p=0;p<n;p++)for(let q=p+1;q<n;q++){
   if(Math.abs(a[p][q])<1e-300)continue;
   const theta=(a[q][q]-a[p][p])/(2*a[p][q]),t=Math.sign(theta||1)/(Math.abs(theta)+Math.sqrt(theta*theta+1)),c=1/Math.sqrt(t*t+1),s=t*c;
   for(let k=0;k<n;k++){const akp=a[k][p],akq=a[k][q];a[k][p]=c*akp-s*akq;a[k][q]=s*akp+c*akq;}
   for(let k=0;k<n;k++){const apk=a[p][k],aqk=a[q][k];a[p][k]=c*apk-s*aqk;a[q][k]=s*apk+c*aqk;}
   for(let k=0;k<n;k++){const vkp=v[k][p],vkq=v[k][q];v[k][p]=c*vkp-s*vkq;v[k][q]=s*vkp+c*vkq;}
  }
 }
 return {values:a.map((r,i)=>r[i]),vectors:v};
}
const wOf=(h:Homography,p:Pt)=>h[6]*p.x+h[7]*p.y+h[8];
// Least-squares homography from >= 4 correspondences. undefined for degenerate (collinear/duplicate) sets,
// non-unique solutions, a fold (points on both sides of the horizon) or a reprojection RMS above
// maxError x the destination spread.
export function solveHomography(src:Pt[],dst:Pt[],maxError=.05):Homography|undefined{
 const n=src.length;if(n<4||dst.length!==n)return undefined;
 if(![...src,...dst].every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)))return undefined;
 const ns=normalize(src),nd=normalize(dst);if(!ns||!nd||!generalPosition(ns.q)||!generalPosition(nd.q))return undefined;
 const m=Array.from({length:9},()=>Array(9).fill(0));
 for(let i=0;i<n;i++){const {x,y}=ns.q[i],{x:u,y:v}=nd.q[i];
  for(const r of [[-x,-y,-1,0,0,0,u*x,u*y,u],[0,0,0,-x,-y,-1,v*x,v*y,v]])for(let a=0;a<9;a++)for(let b=a;b<9;b++)m[a][b]+=r[a]*r[b];}
 for(let a=0;a<9;a++)for(let b=0;b<a;b++)m[a][b]=m[b][a];
 const {values,vectors}=jacobi(m),order=values.map((v,i)=>i).sort((a,b)=>values[a]-values[b]);
 if(values[order[1]]<1e-9*Math.max(1e-12,values[order[8]]))return undefined; // null space > 1: not unique
 const hn=vectors.map(r=>r[order[0]]);
 if(Math.abs(det3(hn))<1e-6)return undefined; // singular map (e.g. collinear image of a quad)
 let h=mul3(mul3(nd.inv,hn),ns.t);
 const c={x:src.reduce((s,p)=>s+p.x,0)/n,y:src.reduce((s,p)=>s+p.y,0)/n},wc=wOf(h,c);if(!(Math.abs(wc)>1e-12))return undefined;
 h=h.map(v=>v/wc);
 if(src.some(p=>!(wOf(h,p)>1e-9)))return undefined; // orientation flip within the set
 let err=0;for(let i=0;i<n;i++){const q=applyHomography(h,src[i]);if(!q)return undefined;err+=(q.x-dst[i].x)**2+(q.y-dst[i].y)**2;}
 if(Math.sqrt(err/n)>maxError*nd.scale)return undefined;
 return h;
}
export function applyHomography(h:Homography,p:Pt):Pt|undefined{
 const w=wOf(h,p);if(!(Math.abs(w)>1e-12))return undefined;
 const x=(h[0]*p.x+h[1]*p.y+h[2])/w,y=(h[3]*p.x+h[4]*p.y+h[5])/w;return Number.isFinite(x)&&Number.isFinite(y)?{x,y}:undefined;
}
// Exact inverse (adjugate / determinant, scaled by a positive factor), so w stays positive on mapped points.
export function invertHomography(h:Homography):Homography|undefined{
 if(h.length!==9||!h.every(Number.isFinite))return undefined;const inv=inv3(h);if(!inv)return undefined;
 const s=Math.max(...inv.map(Math.abs));return s>0?inv.map(v=>v/s):undefined;
}
const CORNERS:Record<string,Pt>={'far-left':{x:0,y:0},'far-right':{x:1,y:0},'near-right':{x:1,y:1},'near-left':{x:0,y:1}};
const CORNER_LINES:Record<string,[string,string]>={'far-left':['far-touchline','left-goal-line'],'far-right':['far-touchline','right-goal-line'],'near-right':['near-touchline','right-goal-line'],'near-left':['near-touchline','left-goal-line']};
function intersect(l1:Pt[],l2:Pt[]):Pt|undefined{
 const [p,q]=l1,[r,s]=l2,d1={x:q.x-p.x,y:q.y-p.y},d2={x:s.x-r.x,y:s.y-r.y},den=d1.x*d2.y-d1.y*d2.x,n1=Math.hypot(d1.x,d1.y),n2=Math.hypot(d2.x,d2.y);
 if(!(n1>1e-4&&n2>1e-4)||Math.abs(den)<.1*n1*n2)return undefined; // near-parallel (< ~6 degrees): unstable
 const t=((r.x-p.x)*d2.y-(r.y-p.y)*d2.x)/den,x=p.x+t*d1.x,y=p.y+t*d1.y;
 return x>-2&&x<3&&y>-2&&y<3?{x,y}:undefined;
}
// Build image->pitch homography for `time` from user field marks (corners, and corners implied by
// intersecting named boundary lines), projected with the saved camera chain. Needs 4 non-collinear points.
// error = RMS reprojection error in normalized pitch units. A mirrored solution (left/right or near/far
// labels swapped) is rejected: image and pitch axes have the same orientation.
export function pitchHomography(marks:FieldMark[],time:number,cameras:FieldCamera[]):{h:Homography;points:number;error:number}|undefined{
 const at=(name:string)=>{for(const m of marks.filter(m=>m.name===name).sort((a,b)=>Math.abs(a.time-time)-Math.abs(b.time-time))){const p=projectFieldMark(m,time,cameras);if(p&&p.points.every(q=>Number.isFinite(q.x)&&Number.isFinite(q.y)))return p.points;}return undefined;};
 const src:Pt[]=[],dst:Pt[]=[];
 for(const [name,target] of Object.entries(CORNERS)){
  let p=at(name)?.[0];
  if(!p){const [a,b]=CORNER_LINES[name],la=at(a),lb=at(b);if(la?.length===2&&lb?.length===2)p=intersect(la,lb);}
  if(p){src.push(p);dst.push(target);}
 }
 if(src.length<4)return undefined;
 const h=solveHomography(src,dst);if(!h||det3(h)<=0)return undefined;
 let err=0;for(let i=0;i<src.length;i++){const q=applyHomography(h,src[i])!;err+=(q.x-dst[i].x)**2+(q.y-dst[i].y)**2;}
 return {h,points:src.length,error:Math.sqrt(err/src.length)};
}
// Foot point of a box in pitch coordinates; undefined beyond the horizon or far outside the pitch.
export function toPitch(h:Homography,box:Box):Pt|undefined{
 const f=footPoint(box);if(!(wOf(h,f)>0))return undefined;const p=applyHomography(h,f);
 return p&&p.x>=-PITCH_SLACK&&p.x<=1+PITCH_SLACK&&p.y>=-PITCH_SLACK&&p.y<=1+PITCH_SLACK?p:undefined;
}
