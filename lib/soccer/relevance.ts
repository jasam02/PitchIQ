// Is a person detection part of the match? Field zone (pitch.ts) + perspective size plausibility.
// Precision over recall for spectators; people a few pixels beyond a touchline stay ('boundary').
import {footPoint,nearestBoundary,pointInPolygon,signedDistance,zoneOf} from './pitch';
import type {Box,FieldFilterConfig,PitchModel,Pt,RejectedDetection,RejectReason,ScoredDetection,SizeModel,TrackDetection} from './types';
export const MIN_PERSON_SCORE=.08; // below this a detection is noise even for existing tracks
export const SIZE_FIT_SCORE=.35; // people used to fit the size model
export const SIZE_MIN_SAMPLES=6,SIZE_MAX_RATIO=2.2,SIZE_MIN_RATIO=.4,LYING_MIN_RATIO=.6;
const median=(v:number[])=>{const s=[...v].sort((a,b)=>a-b),n=s.length;return n?(n%2?s[n>>1]:(s[n/2-1]+s[n/2])/2):0;};
// Robust perspective size model (Theil-Sen): expected box height = a + b * footY. People lower in the
// image are closer to a camera above the pitch, so the slope is never negative.
export function fitSizeModel(people:{box:Box}[]):SizeModel{
 const pts=people.map(p=>({y:p.box.y+p.box.h,h:p.box.h})).filter(p=>Number.isFinite(p.y)&&p.h>0);
 if(!pts.length)return {a:0,b:0,reliable:false};
 const slopes:number[]=[];
 for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){const dy=pts[j].y-pts[i].y;if(Math.abs(dy)>=.03)slopes.push((pts[j].h-pts[i].h)/dy);}
 const b=slopes.length?Math.max(0,median(slopes)):0,a=median(pts.map(p=>p.h-b*p.y));
 const ys=pts.map(p=>p.y),spread=Math.max(...ys)-Math.min(...ys);
 // Without depth spread the slope is unknown; a constant model would mis-judge near/far people.
 return {a,b,reliable:pts.length>=SIZE_MIN_SAMPLES&&spread>=.06&&a+b*Math.min(...ys)>0};
}
export function expectedHeight(size:SizeModel,footY:number){return size.a+size.b*footY;}
// Split person detections into match-relevant (inside/boundary/unknown zone, plausible size) and rejected
// (with reason). Low-score detections (>= MIN_PERSON_SCORE) inside the pitch are kept for existing tracks.
// Ball detections are returned untouched in `balls`. `size` is the previous model; the returned one is
// refitted from this frame when enough people stand inside the pitch, else the previous one is kept.
// sizeRatio is 1 when no reliable size model exists. outsideBy is signed (negative = depth inside).
export function assessDetections(detections:TrackDetection[],pitch:PitchModel,config:FieldFilterConfig,size?:SizeModel):{accepted:ScoredDetection[];rejected:RejectedDetection[];balls:TrackDetection[];size:SizeModel}{
 const balls:TrackDetection[]=[],scored:ScoredDetection[]=[];
 for(const d of detections){if(d.kind==='ball'){balls.push(d);continue;}const {zone,outsideBy}=zoneOf(pitch,d.box,config);scored.push({...d,foot:footPoint(d.box),zone,outsideBy,sizeRatio:1});}
 const filtering=config.enabled&&pitch.reliable;let model:SizeModel=size??{a:0,b:0,reliable:false};
 if(filtering){const fit=fitSizeModel(scored.filter(d=>d.zone==='inside'&&d.score>=SIZE_FIT_SCORE));if(fit.reliable)model=fit;}
 const useSize=filtering&&model.reliable;
 // Too short but about as long as a standing player is tall (a player lying after a tackle, a diving keeper): the
 // long body axis is measured instead.
 if(useSize)for(const d of scored){const e=expectedHeight(model,d.foot.y),long=Math.max(d.box.h,d.box.w*pitch.aspect);if(e>.004)d.sizeRatio=d.box.h/e<SIZE_MIN_RATIO&&long/e>=LYING_MIN_RATIO?long/e:d.box.h/e;}
 const outside=scored.filter(d=>d.zone==='outside'),accepted:ScoredDetection[]=[],rejected:RejectedDetection[]=[];
 const reject=(d:ScoredDetection,reason:RejectReason,detail:string)=>rejected.push({box:d.box,score:d.score,reason,detail});
 for(const d of scored){
  if(d.zone==='outside'){const why=audience(d,pitch,outside);reject(d,why.audience?'audience':'outside-pitch',why.detail);continue;}
  if(useSize&&(d.sizeRatio>SIZE_MAX_RATIO||d.sizeRatio<SIZE_MIN_RATIO)){reject(d,'implausible-size',`height ${d.box.h.toFixed(3)} is ${d.sizeRatio.toFixed(2)}x the expected ${expectedHeight(model,d.foot.y).toFixed(3)} at foot y ${d.foot.y.toFixed(2)}`);continue;}
  if(d.score<MIN_PERSON_SCORE){reject(d,'low-confidence',`score ${d.score.toFixed(2)} below ${MIN_PERSON_SCORE}`);continue;}
  accepted.push(d);
 }
 return {accepted,rejected,balls,size:model};
}
// Spectators: beyond the far boundary, standing off the grass in the upper part of the frame (stands), or
// in a dense group of outside detections. Everyone else outside is 'outside-pitch' (bench, technical area,
// behind boards, beyond a goal line).
function audience(d:ScoredDetection,pitch:PitchModel,outside:ScoredDetection[]):{audience:boolean;detail:string}{
 const edge=nearestBoundary(pitch,d.foot),base=`foot ${d.outsideBy.toFixed(3)} beyond ${edge?.label??'pitch'}`;
 if(edge?.side==='far')return {audience:true,detail:`${base} (far side)`};
 const offGrass=pitch.grassPolygon.length>=3&&!pointInPolygon(d.foot,pitch.grassPolygon)&&signedDistance(d.foot,pitch.grassPolygon,pitch.aspect)>.01;
 if(offGrass&&d.foot.y<polygonMidY(pitch.polygon))return {audience:true,detail:`${base} (stands)`};
 const radius=Math.max(.08,2*d.box.h),near=outside.filter(o=>o!==d&&dist(o.foot,d.foot,pitch.aspect)<radius).length;
 if(near>=3&&!(edge?.side==='near'&&d.outsideBy<.15))return {audience:true,detail:`${base} (crowd of ${near+1})`};
 return {audience:false,detail:base};
}
const dist=(a:Pt,b:Pt,aspect:number)=>Math.hypot((a.x-b.x)*aspect,a.y-b.y);
function polygonMidY(poly:Pt[]){if(!poly.length)return .5;let lo=Infinity,hi=-Infinity;for(const p of poly){lo=Math.min(lo,p.y);hi=Math.max(hi,p.y);}return (lo+hi)/2;}
