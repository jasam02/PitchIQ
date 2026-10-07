// Playable region per frame: grass segmentation + boundary (touch/goal) line detection.
// Recomputed every step (never one permanent rectangle). A boundary line that briefly disappears is
// carried with the camera motion for a short time, then dropped. Image-plane only: no metric calibration.
import type {Box,BoundaryLine,BoundarySide,CameraMotion,FieldFilterConfig,FieldZone,Frame,PitchModel,Pt} from './types';
export const defaultFieldFilter:FieldFilterConfig={enabled:true,boundaryMargin:.35,minMargin:.006,maxMargin:.03};
// Field filter for the touchline tolerance chosen in the UI (fraction of player height): floor and ceiling scale with
// it, so the whole slider range has an effect and 0 means no tolerance.
export function marginFilter(boundaryMargin:number,base:FieldFilterConfig=defaultFieldFilter):FieldFilterConfig{
 const m=Number.isFinite(boundaryMargin)?Math.max(0,boundaryMargin):base.boundaryMargin,k=base.boundaryMargin>0?m/base.boundaryMargin:1;
 return {...base,boundaryMargin:m,minMargin:base.minMargin*k,maxMargin:base.maxMargin*k};
}
export const MIN_GRASS=.12; // grass share of the frame needed for a reliable pitch (close-ups, crowd shots fail)
export const LINE_MAX_AGE=2; // seconds a boundary line is carried with a reliable camera estimate
export const LINE_MAX_AGE_LOOSE=.6; // ... when the camera motion is unknown or unreliable
const isGrass=(r:number,g:number,b:number)=>g>r*1.08&&g>b*1.15&&g>35; // same predicate as track-vision
const isWhite=(r:number,g:number,b:number)=>{const lo=Math.min(r,g,b);return lo>=110&&Math.max(r,g,b)-lo<=70&&r+g+b>=380&&!isGrass(r,g,b);};

// ---------- sampling (one pass over a sparse pixel subset) ----------
// Fine grid <= 320 wide with square-ish cells. Each cell reads a short vertical and horizontal run of
// pixels through its centre, so thin painted lines of any orientation are still seen.
type Fine={w:number;h:number;frac:Float32Array;grass:Uint8Array;white:Uint8Array;dark:Uint8Array};
function sampleFrame(frame:Frame):Fine{
 const fw=frame.width,fh=frame.height,data=frame.data,w=Math.min(320,fw),h=Math.max(2,Math.min(fh,Math.round(w*fh/fw)));
 const cw=fw/w,ch=fh/h,sx=Math.max(1,Math.floor(cw/3)),sy=Math.max(1,Math.floor(ch/3));
 const frac=new Float32Array(w*h),grass=new Uint8Array(w*h),white=new Uint8Array(w*h),dark=new Uint8Array(w*h);
 for(let gy=0;gy<h;gy++){
  const y0=Math.floor(gy*ch),y1=Math.max(y0+1,Math.floor((gy+1)*ch)),ym=(y0+y1)>>1;
  for(let gx=0;gx<w;gx++){
   const x0=Math.floor(gx*cw),x1=Math.max(x0+1,Math.floor((gx+1)*cw)),xm=(x0+x1)>>1;let n=0,g=0,wh=0,lit=0;
   for(let y=y0;y<y1;y+=sy){const i=(y*fw+xm)*4,r=data[i],gg=data[i+1],b=data[i+2];n++;if(isGrass(r,gg,b))g++;else if(isWhite(r,gg,b))wh=1;if(Math.max(r,gg,b)>=28)lit=1;}
   for(let x=x0;x<x1;x+=sx){if(x===xm)continue;const i=(ym*fw+x)*4,r=data[i],gg=data[i+1],b=data[i+2];n++;if(isGrass(r,gg,b))g++;else if(isWhite(r,gg,b))wh=1;if(Math.max(r,gg,b)>=28)lit=1;}
   const k=gy*w+gx;frac[k]=g/n;grass[k]=g*2>=n?1:0;white[k]=wh;dark[k]=lit?0:1;
  }
 }
 return {w,h,frac,grass,white,dark};
}

// ---------- grass region (coarse grid, ~160 wide) ----------
// cw/ch: normalized size of one coarse cell. left/right per row, top/bottom per column (-1 = empty).
// x0..x1, y0..y1: coarse cells of the picture itself (letterbox / pillarbox bars excluded), where the view ends.
type Region={w:number;h:number;cw:number;ch:number;mask:Uint8Array;count:number;cx:number;cy:number;left:Int16Array;right:Int16Array;top:Int16Array;bottom:Int16Array;x0:number;x1:number;y0:number;y1:number};
// Fills short non-grass runs between grass cells (painted lines, players) before the pitch component is chosen. A
// run is filled only when each of its cells is a painted line (white) or a narrow object with grass on both sides
// across the run: a band of advertising boards is neither, so green seats behind it never join the pitch.
function closeGaps(m:Uint8Array,w:number,h:number,gap:number,white:Uint8Array){
 const src=m.slice(),grassNear=(x:number,y:number,dx:number,dy:number)=>{for(let k=1;k<=3;k++){const xx=x+dx*k,yy=y+dy*k;if(xx<0||yy<0||xx>=w||yy>=h)return false;if(src[yy*w+xx])return true;}return false;};
 const ok=(x:number,y:number,rows:boolean)=>white[y*w+x]===1||(rows?grassNear(x,y,0,-1)&&grassNear(x,y,0,1):grassNear(x,y,-1,0)&&grassNear(x,y,1,0));
 for(let y=0;y<h;y++){let last=-1;for(let x=0;x<w;x++)if(src[y*w+x]){if(last>=0&&x-last>1&&x-last-1<=gap){let fill=true;for(let k=last+1;k<x&&fill;k++)fill=ok(k,y,true);if(fill)for(let k=last+1;k<x;k++)m[y*w+k]=1;}last=x;}}
 for(let x=0;x<w;x++){let last=-1;for(let y=0;y<h;y++)if(src[y*w+x]){if(last>=0&&y-last>1&&y-last-1<=gap){let fill=true;for(let k=last+1;k<y&&fill;k++)fill=ok(x,k,false);if(fill)for(let k=last+1;k<y;k++)m[k*w+x]=1;}last=y;}}
}
// Opening with a cross removes 1-2 cell tendrils (green boards, seats) touching the pitch.
function openMask(m:Uint8Array,w:number,h:number){
 const e=new Uint8Array(w*h),o=new Uint8Array(w*h);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const k=y*w+x;e[k]=m[k]&&(x===0||m[k-1])&&(x===w-1||m[k+1])&&(y===0||m[k-w])&&(y===h-1||m[k+w])?1:0;}
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const k=y*w+x;o[k]=e[k]||(x>0&&e[k-1])||(x<w-1&&e[k+1])||(y>0&&e[k-w])||(y<h-1&&e[k+w])?1:0;}
 return o;
}
function grassRegion(f:Fine):Region{
 const f2=f.w>=200?2:1,w=Math.floor(f.w/f2),h=Math.floor(f.h/f2);let m=new Uint8Array(w*h);const white=new Uint8Array(w*h);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){let s=0,wh=0;for(let j=0;j<f2;j++)for(let i=0;i<f2;i++){const k=(y*f2+j)*f.w+x*f2+i;s+=f.frac[k];wh|=f.white[k];}m[y*w+x]=s>=f2*f2*.5?1:0;white[y*w+x]=wh;}
 closeGaps(m,w,h,3,white);m=openMask(m,w,h);
 // largest 4-connected component
 const label=new Int32Array(w*h).fill(-1),queue=new Int32Array(w*h);let bestLabel=-1,bestSize=0,next=0;
 for(let s=0;s<m.length;s++){
  if(!m[s]||label[s]>=0)continue;let head=0,tail=0;queue[tail++]=s;label[s]=next;
  while(head<tail){const q=queue[head++],x=q%w;
   if(x>0&&m[q-1]&&label[q-1]<0){label[q-1]=next;queue[tail++]=q-1;}
   if(x<w-1&&m[q+1]&&label[q+1]<0){label[q+1]=next;queue[tail++]=q+1;}
   if(q>=w&&m[q-w]&&label[q-w]<0){label[q-w]=next;queue[tail++]=q-w;}
   if(q+w<m.length&&m[q+w]&&label[q+w]<0){label[q+w]=next;queue[tail++]=q+w;}}
  if(tail>bestSize){bestSize=tail;bestLabel=next;}next++;
 }
 const mask=new Uint8Array(w*h),left=new Int16Array(h).fill(-1),right=new Int16Array(h).fill(-1),top=new Int16Array(w).fill(-1),bottom=new Int16Array(w).fill(-1);let sx=0,sy=0;
 const cw=f2/f.w,ch=f2/f.h;
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const k=y*w+x;if(label[k]!==bestLabel||bestLabel<0)continue;mask[k]=1;sx+=x;sy+=y;
  if(left[y]<0)left[y]=x;right[y]=x;if(top[x]<0)top[x]=y;bottom[x]=y;}
 // Uniformly black columns / rows at the frame edges (4:3 footage in a 16:9 frame) are not part of the view.
 const col=(x:number)=>{for(let y=0;y<f.h;y++)if(!f.dark[y*f.w+x])return false;return true;},row=(y:number)=>{for(let x=0;x<f.w;x++)if(!f.dark[y*f.w+x])return false;return true;};
 let bl=0,br=0,bt=0,bb=0;while(bl<f.w/4&&col(bl))bl++;while(br<f.w/4&&col(f.w-1-br))br++;while(bt<f.h/4&&row(bt))bt++;while(bb<f.h/4&&row(f.h-1-bb))bb++;
 return {w,h,cw,ch,mask,count:bestSize,cx:bestSize?(sx/bestSize+.5)*cw:.5,cy:bestSize?(sy/bestSize+.5)*ch:.5,left,right,top,bottom,x0:Math.ceil(bl/f2),x1:w-1-Math.ceil(br/f2),y0:Math.ceil(bt/f2),y1:h-1-Math.ceil(bb/f2)};
}

// ---------- geometry ----------
const cross=(o:Pt,a:Pt,b:Pt)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
export function convexHull(points:Pt[]):Pt[]{
 const p=[...points].sort((a,b)=>a.x-b.x||a.y-b.y);if(p.length<3)return p;
 const lower:Pt[]=[],upper:Pt[]=[];
 for(const q of p){while(lower.length>=2&&cross(lower[lower.length-2],lower[lower.length-1],q)<=1e-12)lower.pop();lower.push(q);}
 for(let i=p.length-1;i>=0;i--){const q=p[i];while(upper.length>=2&&cross(upper[upper.length-2],upper[upper.length-1],q)<=1e-12)upper.pop();upper.push(q);}
 return [...lower.slice(0,-1),...upper.slice(0,-1)];
}
export function polygonArea(poly:Pt[]){let s=0;for(let i=0;i<poly.length;i++){const a=poly[i],b=poly[(i+1)%poly.length];s+=a.x*b.y-b.x*a.y;}return Math.abs(s)/2;}
function centroid(poly:Pt[]):Pt{let x=0,y=0;for(const p of poly){x+=p.x;y+=p.y;}return {x:x/poly.length,y:y/poly.length};}
// Keep the part of a convex polygon on the same side of line a-b as `keep`.
export function clipPolygon(poly:Pt[],a:Pt,b:Pt,keep:Pt):Pt[]{
 const sign=cross(a,b,keep)>=0?1:-1,f=(p:Pt)=>sign*cross(a,b,p),out:Pt[]=[];
 for(let i=0;i<poly.length;i++){const p=poly[i],q=poly[(i+1)%poly.length],fp=f(p),fq=f(q);if(fp>=0)out.push(p);if((fp>=0)!==(fq>=0)){const t=fp/(fp-fq);out.push({x:p.x+(q.x-p.x)*t,y:p.y+(q.y-p.y)*t});}}
 return out;
}
function clipUnit(poly:Pt[]){
 let p=poly;const c={x:.5,y:.5};
 for(const [a,b] of [[{x:0,y:0},{x:1,y:0}],[{x:1,y:0},{x:1,y:1}],[{x:1,y:1},{x:0,y:1}],[{x:0,y:1},{x:0,y:0}]] as [Pt,Pt][]){if(p.length<3)return [];p=clipPolygon(p,a,b,c);}
 return p.length>=3?p:[];
}
const yAt=(l:{a:Pt;b:Pt},x:number)=>Math.abs(l.b.x-l.a.x)<1e-9?l.a.y:l.a.y+(l.b.y-l.a.y)*(x-l.a.x)/(l.b.x-l.a.x);
const xAt=(l:{a:Pt;b:Pt},y:number)=>Math.abs(l.b.y-l.a.y)<1e-9?l.a.x:l.a.x+(l.b.x-l.a.x)*(y-l.a.y)/(l.b.y-l.a.y);
const isHorizontal=(l:{a:Pt;b:Pt;side?:BoundarySide})=>l.side?l.side==='near'||l.side==='far':Math.abs(l.b.y-l.a.y)<=Math.abs(l.b.x-l.a.x)*.4;
function segDistance(p:Pt,a:Pt,b:Pt,aspect:number){
 const ax=a.x*aspect,bx=b.x*aspect,px=p.x*aspect,dx=bx-ax,dy=b.y-a.y,len=dx*dx+dy*dy;
 const t=len>0?Math.max(0,Math.min(1,((px-ax)*dx+(p.y-a.y)*dy)/len)):0;return Math.hypot(px-ax-t*dx,p.y-a.y-t*dy);
}
function lineDistance(p:Pt,a:Pt,b:Pt,aspect:number){const dx=(b.x-a.x)*aspect,dy=b.y-a.y,len=Math.hypot(dx,dy);return len>0?Math.abs(dx*(p.y-a.y)-dy*(p.x-a.x)*aspect)/len:Infinity;}

// ---------- boundary lines ----------
type Seg={kind:'h'|'v';a:Pt;b:Pt;support:number}; // support: fraction of grid columns (h) or rows (v) on the line
// Hough-style voting over points sorted by u (u = along the line axis, v = across). Votes count distinct
// u per bin so thick lines are not over-counted; each peak is refined by least squares and its inliers removed.
function hough(U:number[],V:number[],uLen:number,vLen:number,maxSlope:number,step:number,binW:number,maxLines:number,minSupport:number){
 const n=U.length,out:{u0:number;u1:number;v0:number;v1:number;support:number}[]=[];if(n<minSupport)return out;
 const slopes:number[]=[];for(let s=-maxSlope;s<=maxSlope+1e-9;s+=step)slopes.push(s);
 const uc=uLen/2,off=Math.ceil(maxSlope*uc/binW)+2,nb=Math.ceil(vLen/binW)+2*off+1,ns=slopes.length;
 const acc=new Int32Array(ns*nb),last=new Int32Array(ns*nb),alive=new Uint8Array(n).fill(1);
 for(let iter=0;iter<maxLines+2&&out.length<maxLines;iter++){
  acc.fill(0);last.fill(0);let best=0,bk=-1;
  for(let p=0;p<n;p++){if(!alive[p])continue;const u=U[p]+1,du=(U[p]-uc)/binW,v=V[p]/binW+off+.5;
   for(let j=0,base=0;j<ns;j++,base+=nb){const k=base+((v-slopes[j]*du)|0);if(last[k]!==u){last[k]=u;const c=++acc[k];if(c>best){best=c;bk=k;}}}}
  if(best<minSupport)break;
  let s=slopes[Math.floor(bk/nb)],c=(bk%nb-off)*binW;
  for(const tol of [binW+1,1.5]){let sw=0,su=0,sv=0,suu=0,suv=0;
   for(let p=0;p<n;p++){if(!alive[p])continue;const du=U[p]-uc;if(Math.abs(V[p]-(c+s*du))>tol)continue;sw++;su+=du;sv+=V[p];suu+=du*du;suv+=du*V[p];}
   const den=sw*suu-su*su;if(sw<2||den<=1e-9)break;s=(sw*suv-su*sv)/den;c=(sv-s*su)/sw;}
  const us:number[]=[];let distinct=0,lastU=-1;
  for(let p=0;p<n;p++){if(!alive[p])continue;const r=Math.abs(V[p]-(c+s*(U[p]-uc)));if(r<=1.5){us.push(U[p]);if(U[p]!==lastU){distinct++;lastU=U[p];}}if(r<=3)alive[p]=0;}
  if(distinct<minSupport||Math.abs(s)>maxSlope*1.15)continue;
  const u0=us[Math.floor(us.length*.02)],u1=us[Math.max(0,Math.ceil(us.length*.98)-1)];
  out.push({u0,u1,v0:c+s*(u0-uc),v1:c+s*(u1-uc),support:distinct});
 }
 return out;
}
// Long straight white lines with grass on both sides (perpendicular): near-horizontal (touchline family,
// |dy/dx| <= ~.36) and steep (goal-line family). Interior lines (halfway, penalty box) are found too and
// rejected later by the boundary tests.
function detectLines(f:Fine):Seg[]{
 const {w,h,grass,white}=f,d=Math.max(2,Math.round(h/60)),out:Seg[]=[];
 const near=(x:number,y:number,dx:number,dy:number)=>{for(let k=d;k<=2*d;k++){const xx=x+dx*k,yy=y+dy*k;if(xx>=0&&yy>=0&&xx<w&&yy<h&&grass[yy*w+xx])return true;}return false;};
 const hu:number[]=[],hv:number[]=[],vu:number[]=[],vv:number[]=[];
 for(let x=0;x<w;x++)for(let y=0;y<h;y++)if(white[y*w+x]&&near(x,y,0,-1)&&near(x,y,0,1)){hu.push(x);hv.push(y);}
 for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(white[y*w+x]&&near(x,y,-1,0)&&near(x,y,1,0)){vu.push(y);vv.push(x);}
 for(const l of hough(hu,hv,w,h,.36,.02,2,4,.2*w))out.push({kind:'h',a:{x:(l.u0+.5)/w,y:(l.v0+.5)/h},b:{x:(l.u1+.5)/w,y:(l.v1+.5)/h},support:l.support/w});
 for(const l of hough(vu,vv,h,w,2.9,.05,3,4,.15*h))out.push({kind:'v',a:{x:(l.v0+.5)/w,y:(l.u0+.5)/h},b:{x:(l.v1+.5)/w,y:(l.u1+.5)/h},support:l.support/h});
 return out;
}
// Grass beyond vs inside a line, whether the grass beyond ends inside the frame (boards/stands follow a
// real boundary; an interior line has pitch up to the frame edge), and the median depth of that band.
function bandStats(l:{a:Pt;b:Pt},side:BoundarySide,r:Region){
 const horizontal=side==='near'||side==='far',out=side==='near'||side==='right'?1:-1,tol=horizontal?r.ch:r.cw;let inner=0,beyond=0;
 for(let y=0;y<r.h;y++)for(let x=0;x<r.w;x++){if(!r.mask[y*r.w+x])continue;const px=(x+.5)*r.cw,py=(y+.5)*r.ch,o=out*(horizontal?py-yAt(l,px):px-xAt(l,py));if(o>tol)beyond++;else if(o<0)inner++;}
 let count=0,term=0,on=0;const depths:number[]=[];
 if(horizontal){const x0=Math.max(0,Math.floor(Math.min(l.a.x,l.b.x)/r.cw)),x1=Math.min(r.w-1,Math.floor(Math.max(l.a.x,l.b.x)/r.cw));
  for(let x=x0;x<=x1;x++){if(r.top[x]<0)continue;count++;const ly=yAt(l,(x+.5)*r.cw);
   if(side==='far'){if(r.top[x]>=r.y0+1)term++;depths.push(Math.max(0,ly-r.top[x]*r.ch));on+=ly>=r.top[x]*r.ch-r.ch?1:0;}else{if(r.bottom[x]<=r.y1-1)term++;depths.push(Math.max(0,(r.bottom[x]+1)*r.ch-ly));on+=ly<=(r.bottom[x]+2)*r.ch?1:0;}}}
 else{const y0=Math.max(0,Math.floor(Math.min(l.a.y,l.b.y)/r.ch)),y1=Math.min(r.h-1,Math.floor(Math.max(l.a.y,l.b.y)/r.ch));
  for(let y=y0;y<=y1;y++){if(r.left[y]<0)continue;count++;const lx=xAt(l,(y+.5)*r.ch);
   if(side==='left'){if(r.left[y]>=r.x0+1)term++;depths.push(Math.max(0,lx-r.left[y]*r.cw));on+=lx>=r.left[y]*r.cw-r.cw?1:0;}else{if(r.right[y]<=r.x1-1)term++;depths.push(Math.max(0,(r.right[y]+1)*r.cw-lx));on+=lx<=(r.right[y]+2)*r.cw?1:0;}}}
 depths.sort((a,b)=>a-b);
 return {inner,beyond,term:count?term/count:0,depth:depths.length?depths[depths.length>>1]:Infinity,count,on:count?on/count:0};
}
function sideOf(l:{a:Pt;b:Pt},horizontal:boolean,c:Pt):BoundarySide{return horizontal?(yAt(l,c.x)<c.y?'far':'near'):(xAt(l,c.y)<c.x?'left':'right');}
// Share of the line's extent where a thin painted line runs along the outer edge of the grass (just beyond it,
// with no white further out, so not a white board): the real boundary is there, a line inside it is interior.
function edgeLine(l:{a:Pt;b:Pt},side:BoundarySide,r:Region,f:Fine){
 const k=f.w/r.w,horizontal=side==='near'||side==='far',out=side==='near'||side==='right'?1:-1;let n=0,hit=0;
 const white=(along:number,across:number)=>{const x=horizontal?along:across,y=horizontal?across:along;return x>=0&&y>=0&&x<f.w&&y<f.h&&f.white[y*f.w+x]===1;};
 const from=Math.max(0,Math.floor(Math.min(horizontal?l.a.x:l.a.y,horizontal?l.b.x:l.b.y)/(horizontal?r.cw:r.ch))),to=Math.min((horizontal?r.w:r.h)-1,Math.floor(Math.max(horizontal?l.a.x:l.a.y,horizontal?l.b.x:l.b.y)/(horizontal?r.cw:r.ch)));
 for(let c=from;c<=to;c++){
  const e=horizontal?(side==='far'?r.top[c]:r.bottom[c]):(side==='left'?r.left[c]:r.right[c]);if(e<0)continue;n++;
  const edge=out>0?(e+1)*k-1:e*k,at=(d:number)=>{for(let i=0;i<k;i++)if(white(c*k+i,edge+out*d))return true;return false;};
  if((at(-1)||at(0)||at(1)||at(2)||at(3))&&!(at(5)||at(6)||at(7)))hit++;
 }
 return n?hit/n:0;
}
// A line is a boundary only if it is outermost on its side and the grass beyond it is a thin run-off band.
// Touchlines must be long; the far side and goal lines also need the grass beyond to end inside the view, so a
// penalty-box or halfway line with pitch running off-frame is never mistaken for a boundary; with deep grass
// beyond, a painted line at the grass edge proves the candidate interior. Short lines (technical-area markings,
// an oblique goal line) are filtered first so they never hide the real line inside them; white rows outside the
// grass region (stands) are skipped.
function chooseBoundaries(segs:Seg[],r:Region,aspect:number,f:Fine){
 const c={x:r.cx,y:r.cy},bySide=new Map<BoundarySide,{seg:Seg;dist:number}[]>(),found=new Map<BoundarySide,BoundaryLine>();
 for(const seg of segs){
  const horizontal=seg.kind==='h';
  if(horizontal?seg.support<.3||Math.abs(seg.b.x-seg.a.x)<.5:seg.support<.2||Math.abs(seg.b.y-seg.a.y)<.3)continue;
  const side=sideOf(seg,horizontal,c),list=bySide.get(side)||[];list.push({seg,dist:lineDistance(c,seg.a,seg.b,aspect)});bySide.set(side,list);
 }
 for(const [side,list] of bySide)for(const {seg} of list.sort((a,b)=>b.dist-a.dist)){
  const s=bandStats(seg,side,r);if(!s.count||s.on<.5)continue;
  // A failing line fails for every line inside it too (more grass beyond, same edges).
  if(s.beyond>=.25*s.inner)break;
  const interior=s.depth>.06&&edgeLine(seg,side,r,f)>=.5;
  if(side==='near'?s.term<.5&&s.depth>.12:side==='far'?s.term<.5||interior:s.term<.5||interior)break;
  found.set(side,{a:seg.a,b:seg.b,side,support:seg.support,age:0});break;
 }
 return found;
}
const moveLine=(l:BoundaryLine,cam:CameraMotion|undefined):BoundaryLine=>cam?{...l,a:{x:l.a.x*cam.scale+cam.dx,y:l.a.y*cam.scale+cam.dy},b:{x:l.b.x*cam.scale+cam.dx,y:l.b.y*cam.scale+cam.dy}}:l;

// ---------- public API ----------
export function analyzePitch(frame:Frame,time:number,previous?:PitchModel,camera?:CameraMotion):PitchModel{
 const aspect=frame.width/Math.max(1,frame.height),empty:PitchModel={time,reliable:false,aspect,polygon:[],grassPolygon:[],lines:[],coverage:0};
 if(frame.width<16||frame.height<16)return empty;
 const f=sampleFrame(frame),r=grassRegion(f);
 if(r.count<r.w*r.h*MIN_GRASS)return empty;
 const corners:Pt[]=[];
 for(let y=0;y<r.h;y++){if(r.left[y]<0)continue;const x0=r.left[y]*r.cw,x1=(r.right[y]+1)*r.cw,y0=y*r.ch,y1=(y+1)*r.ch;corners.push({x:x0,y:y0},{x:x0,y:y1},{x:x1,y:y0},{x:x1,y:y1});}
 const grassPolygon=convexHull(corners),found=chooseBoundaries(detectLines(f),r,aspect,f);
 if(previous&&previous.lines.length&&!camera?.cut){
  const dt=Math.max(0,time-previous.time),maxAge=camera?.reliable?LINE_MAX_AGE:LINE_MAX_AGE_LOOSE;
  for(const old of previous.lines){
   if(found.has(old.side))continue;const age=old.age+dt;if(age>maxAge)continue;
   const moved=moveLine(old,camera);if(sideOf(moved,isHorizontal(old),{x:r.cx,y:r.cy})!==old.side)continue;
   const s=bandStats(moved,old.side,r);if(s.beyond>=.35*s.inner)continue; // no longer consistent with the grass
   found.set(old.side,{...moved,age});
  }
 }
 const lines=[...found.values()],c={x:r.cx,y:r.cy};let polygon=grassPolygon;
 for(const l of lines){const clipped=clipPolygon(polygon,l.a,l.b,c);if(clipped.length>=3)polygon=clipped;}
 return {time,reliable:polygon.length>=3,aspect,polygon,grassPolygon,lines,coverage:polygonArea(polygon)};
}
// Move a model into the next frame's coordinates (p' = p*scale + d, as estimateCamera fits it).
export function warpPitch(model:PitchModel,camera:CameraMotion,time:number):PitchModel{
 if(camera.cut)return {...model,time,reliable:false,polygon:[],grassPolygon:[],lines:[],coverage:0};
 const t=(p:Pt)=>({x:p.x*camera.scale+camera.dx,y:p.y*camera.scale+camera.dy}),dt=Math.max(0,time-model.time);
 const polygon=clipUnit(model.polygon.map(t)),grassPolygon=clipUnit(model.grassPolygon.map(t));
 const lines=model.lines.map(l=>({...moveLine(l,camera),age:l.age+dt})).filter(l=>l.age<=LINE_MAX_AGE);
 return {time,reliable:model.reliable&&polygon.length>=3,aspect:model.aspect,polygon,grassPolygon,lines,coverage:polygonArea(polygon)};
}
export function footPoint(box:Box):Pt{return {x:box.x+box.w/2,y:box.y+box.h};}
export function pointInPolygon(p:Pt,poly:Pt[]):boolean{
 let inside=false;
 for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j];if((a.y>p.y)!==(b.y>p.y)&&p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x)inside=!inside;}
 return inside;
}
// Signed distance in height units (aspect-corrected): negative inside, positive outside.
export function signedDistance(p:Pt,poly:Pt[],aspect:number):number{
 if(poly.length<3)return Infinity;let best=Infinity;
 for(let i=0;i<poly.length;i++)best=Math.min(best,segDistance(p,poly[i],poly[(i+1)%poly.length],aspect));
 return pointInPolygon(p,poly)?-best:best;
}
// Nearest polygon edge to p: which side of the pitch it is, and whether it is a detected boundary line.
export function nearestBoundary(model:PitchModel,p:Pt):{distance:number;side:BoundarySide;label:string;line?:BoundaryLine}|undefined{
 const poly=model.polygon,asp=model.aspect;if(poly.length<3)return undefined;let best=Infinity,bi=0;
 for(let i=0;i<poly.length;i++){const d=segDistance(p,poly[i],poly[(i+1)%poly.length],asp);if(d<best){best=d;bi=i;}}
 const a=poly[bi],b=poly[(bi+1)%poly.length],c=centroid(poly);let nx=b.y-a.y,ny=-(b.x-a.x)*asp;const len=Math.hypot(nx,ny)||1;nx/=len;ny/=len;
 if(nx*(c.x-a.x)*asp+ny*(c.y-a.y)>0){nx=-nx;ny=-ny;}
 const line=model.lines.find(l=>lineDistance(l.a,a,b,asp)<.006&&lineDistance(l.b,a,b,asp)<.006);
 const side:BoundarySide=line?.side??(ny<-.6?'far':ny>.6?'near':nx<0?'left':'right');
 const label=line?(side==='near'||side==='far'?`${side} touchline`:`${side} goal line`):`${side} grass edge`;
 return {distance:pointInPolygon(p,poly)?-best:best,side,label,line};
}
export function zoneOf(model:PitchModel,box:Box,config:FieldFilterConfig=defaultFieldFilter):{zone:FieldZone;outsideBy:number}{
 if(!config.enabled)return {zone:'inside',outsideBy:0};
 if(!model.reliable||model.polygon.length<3)return {zone:'unknown',outsideBy:0};
 const d=signedDistance(footPoint(box),model.polygon,model.aspect);
 if(d<=0)return {zone:'inside',outsideBy:d}; // negative: depth inside the pitch
 const margin=Math.min(config.maxMargin,Math.max(config.minMargin,config.boundaryMargin*box.h));
 return {zone:d<=margin?'boundary':'outside',outsideBy:d};
}
