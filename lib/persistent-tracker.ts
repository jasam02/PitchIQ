import {iou,type Detection} from './detection-core';
import {appearanceDistance,stillCamera,type CameraMotion} from './track-vision';
import type {Box} from './tracking';
import {cosine} from './player-reid';
export type TrackDetection=Detection&{appearance?:number[];embedding?:number[];quality?:number;kit?:number[];fieldScore?:number;fieldReliable?:boolean};
export type PersistentTrack={team?:string;id:string;kind:'person'|'ball';box:Box;appearance:number[];embedding?:number[];vx:number;vy:number;time:number;lastSeen:number;misses:number;};
export type TrackSample={id:string;box:Box;score:number;evidence:'detection'|'predicted';recovered:boolean};
export function startTrack(id:string,kind:'person'|'ball',box:Box,time:number,appearance:number[]=[]):PersistentTrack{return {id,kind,box,appearance,vx:0,vy:0,time,lastSeen:time,misses:0};}
// Rectangular Hungarian assignment with private unmatched columns.
export function assignMinimum(costs:number[][],unmatched=.78):number[]{
 const n=costs.length;if(!n)return [];const real=costs[0].length,m=real+n;
 const u=Array(n+1).fill(0),v=Array(m+1).fill(0),p=Array(m+1).fill(0),way=Array(m+1).fill(0);
 for(let i=1;i<=n;i++){p[0]=i;let j0=0;const min=Array(m+1).fill(Infinity),used=Array(m+1).fill(false);do{used[j0]=true;const i0=p[j0];let delta=Infinity,j1=0;for(let j=1;j<=m;j++){if(used[j])continue;const c=(j<=real?costs[i0-1][j-1]:unmatched)-u[i0]-v[j];if(c<min[j]){min[j]=c;way[j]=j0;}if(min[j]<delta){delta=min[j];j1=j;}}for(let j=0;j<=m;j++)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else min[j]-=delta;j0=j1;}while(p[j0]!==0);do{const j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0!==0);}
 const result=Array(n).fill(-1);for(let j=1;j<=real;j++)if(p[j]&&costs[p[j]-1][j-1]<unmatched)result[p[j]-1]=j-1;return result;
}
const center=(b:Box)=>({x:b.x+b.w/2,y:b.y+b.h/2});
function warp(box:Box,c:CameraMotion):Box{return {x:box.x*c.scale+c.dx,y:box.y*c.scale+c.dy,w:box.w*c.scale,h:box.h*c.scale};}
function within(b:Box){return b.x>=0&&b.y>=0&&b.x+b.w<=1&&b.y+b.h<=1;}
export function stepTracks(tracks:PersistentTrack[],detections:TrackDetection[],time:number,camera:CameraMotion=stillCamera){
 const samples:TrackSample[]=[],issues:{playerId:string;time:number;reason:string}[]=[],next:PersistentTrack[]=[];
 if(camera.cut)return {tracks:[],samples,issues:tracks.map(t=>({playerId:t.id,time,reason:'Possible camera cut. Confirm identities on this new view.'}))};
 const cam=camera.reliable?camera:stillCamera;
 const predicted=tracks.map(t=>{const b=warp(t.box,cam),dt=Math.max(.001,time-t.time);return {...b,x:b.x+t.vx*dt,y:b.y+t.vy*dt};});
 const cost=(i:number,d:TrackDetection,low:boolean)=>{const t=tracks[i],b=predicted[i];if(t.kind!==d.kind)return 1e3;const a=center(b),q=center(d.box),gap=time-t.lastSeen,gate=t.kind==='ball'?.1:Math.min(.075,Math.max(.02,b.h*.9)+gap*.015),distance=Math.hypot(q.x-a.x,q.y-a.y),ratio=d.box.w*d.box.h/(b.w*b.h),appearance=t.kind==='ball'?0:appearanceDistance(t.appearance,d.appearance||[]);
  if(distance>gate||ratio<.35||ratio>2.9||appearance>.35||(low&&(gap>.5||distance>gate*.65||appearance>.25)))return 1e3;
  const learned=t.embedding&&d.embedding?cosine(t.embedding,d.embedding):undefined;
  if(learned!==undefined&&learned<.6)return 1e3;
  return learned===undefined?.5*distance/gate+.2*(1-iou(b,d.box))+.3*appearance:.35*distance/gate+.15*(1-iou(b,d.box))+.1*appearance+.4*(1-learned);
 };
 const matched=new Map<number,number>(),used=new Set<number>();
 for(const low of [false,true]){const rows=tracks.map((_,i)=>i).filter(i=>!matched.has(i)),cols=detections.map((_,i)=>i).filter(j=>!used.has(j)&&(low?detections[j].score<.25&&detections[j].score>=.08:detections[j].score>=.25));const costs=rows.map(i=>cols.map(j=>cost(i,detections[j],low))),assignment=assignMinimum(costs,low?.58:.78);
  assignment.forEach((col,r)=>{if(col<0)return;const i=rows[r],j=cols[col],c=costs[r][col];const alternatives=costs[r].filter((_,k)=>k!==col);const competing=costs.map(row=>row[col]).filter((_,k)=>k!==r);
   // Do not turn close ties into silent identity switches.
   if(alternatives.some(v=>Math.abs(v-c)<.045)||competing.some(v=>Math.abs(v-c)<.035))return;
   matched.set(i,j);used.add(j);
  });
 }
 tracks.forEach((t,i)=>{const j=matched.get(i),dt=Math.max(.001,time-t.time),gap=time-t.lastSeen;
  if(j!==undefined){const d=detections[j],old=center(warp(t.box,cam)),q=center(d.box);const vx=t.misses?t.vx:.55*t.vx+.45*(q.x-old.x)/dt,vy=t.misses?t.vy:.55*t.vy+.45*(q.y-old.y)/dt;next.push({...t,box:d.box,time,lastSeen:time,misses:0,vx,vy,embedding:d.quality&&d.quality>.55&&d.embedding?d.embedding:t.embedding});samples.push({id:t.id,box:d.box,score:d.score,evidence:'detection',recovered:t.misses>0});}
  else if(gap<=(t.kind==='ball'?.6:1.4)){const b=predicted[i];next.push({...t,box:b,time,misses:t.misses+1,vx:t.vx*.95,vy:t.vy*.95});if(gap<=(t.kind==='ball'?.2:.4)&&within(b))samples.push({id:t.id,box:b,score:Math.max(.05,.4-gap*.6),evidence:'predicted',recovered:false});}
  else issues.push({playerId:t.id,time:t.lastSeen,reason:'No confident match after the recovery window. Check identity here.'});
 });
 return {tracks:next,samples,issues};
}
