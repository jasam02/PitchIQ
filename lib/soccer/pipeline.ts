// Soccer tracking pipeline, one call per detection step:
// frame -> pitch analysis -> outside-field + size filter -> local tracker -> candidate evidence -> team/role
// (temporal) -> appearance gallery & re-ID -> pitch coordinates -> GLOBAL IDENTITY MANAGER -> observations.
import {iou} from '../detection-core';
import {shirtFeature,stillCamera} from '../track-vision';
import {describe} from './appearance';
import {emptyTeamModel,fitTeamModel,kitVote} from './classify';
import {toPitch} from './homography';
import {IdentityManager,type Stab,type StepSample,type TeamSample} from './identity';
import {LocalTracker,localId} from './local-tracker';
import {analyzePitch,defaultFieldFilter,footPoint,zoneOf} from './pitch';
import {assessDetections} from './relevance';
import type {Box,CameraMotion,DebugFrame,DebugTrack,Descriptor,Frame,PitchModel,Pt,ReidEvent,RejectedDetection,RosterSlot,SavedIdentity,ScoredDetection,SizeModel,SoccerFrameInput,SoccerFrameResult,SoccerOptions,Team,TeamModel,TrackDetection} from './types';

export const defaultSoccerOptions:SoccerOptions={field:defaultFieldFilter,maxPerTeam:11,minHits:5,boundaryHits:10,reidMin:.72,reidMargin:.15,gallerySize:5};
export function soccerOptions(o:Partial<SoccerOptions>={}):SoccerOptions{
 const m={...defaultSoccerOptions,...o,field:{...defaultSoccerOptions.field,...(o.field||{})}};
 return {...m,minHits:Math.max(1,Math.round(m.minHits)),boundaryHits:Math.max(1,Math.round(m.boundaryHits)),gallerySize:Math.max(1,Math.min(6,Math.round(m.gallerySize))),maxPerTeam:Math.max(1,Math.round(m.maxPerTeam))};
}
const MIN_VOTE_QUALITY=.2;
const now=()=>typeof performance!=='undefined'?performance.now():Date.now();
const validBox=(b:Box|undefined):b is Box=>!!b&&[b.x,b.y,b.w,b.h].every(Number.isFinite)&&b.w>0&&b.h>0;
const r3=(v:number)=>Math.round(v*1000)/1000;
// Near a goal: from pitch coordinates when a homography exists, else close to a detected goal line.
function nearGoal(pitch:PitchModel,box:Box,p?:Pt){
 if(p)return (p.x<.17||p.x>.83)&&p.y>.12&&p.y<.88;
 const f=footPoint(box),asp=pitch.aspect;
 for(const l of pitch.lines){
  if(l.side!=='left'&&l.side!=='right')continue;
  const dx=(l.b.x-l.a.x)*asp,dy=l.b.y-l.a.y,len=Math.hypot(dx,dy);
  if(len>0&&Math.abs(dx*(f.y-l.a.y)-dy*(f.x-l.a.x)*asp)/len<.15)return true;
 }
 return false;
}
// Motion `first` followed by `second` (image = scale*p + d). Any cut is a cut; any unreliable part makes the whole
// unknown (identity transform, like estimateCamera's unreliable result).
export function composeCamera(first:CameraMotion,second:CameraMotion):CameraMotion{
 const cut=!!first.cut||!!second.cut;
 if(cut||!first.reliable||!second.reliable)return {...stillCamera,cut};
 const s=second.scale;
 return {scale:first.scale*s,dx:first.dx*s+second.dx,dy:first.dy*s+second.dy,reliable:true,cut:false};
}
const overlapsAny=(box:Box,others:Box[],aspect:number)=>others.some(o=>o!==box&&(iou(o,box)>.02||Math.hypot((o.x+o.w/2-box.x-box.w/2)*aspect,o.y+o.h/2-box.y-box.h/2)<.6*(o.w+box.w)/2*aspect));

// Kit clusters come from every on-pitch sample; user labels only orient them (which cluster is A) and a labelled
// player matching neither kit (typically the goalkeeper in slot 1) becomes a keeper prototype of that team.
// Labels alone define the prototypes only while too few people are visible to find two kits.
export function fitKits(samples:TeamSample[],previous?:TeamModel):TeamModel{
 const free=samples.map(s=>s.role?s:{jersey:s.jersey,...(s.weight!==undefined?{weight:s.weight}:{})}); // keeper hints keep their team
 const labelled=samples.filter(s=>s.team&&!s.role);
 let m=fitTeamModel(free,previous);
 if(!m.a||!m.b)return labelled.length?fitTeamModel(samples,previous):m;
 if(!labelled.length)return m;
 const probe=(jersey:number[]):Descriptor=>({jersey,shorts:[],socks:[],layout:[],quality:1});
 let keep=0,swap=0;const keepers:TeamSample[]=[];
 for(const s of labelled){const v=kitVote(m,probe(s.jersey));if(v.team){if(v.team===s.team)keep++;else swap++;}else if(v.outlier&&!v.refLike)keepers.push(s);}
 if(swap>keep){const flip=(t?:Team):Team|undefined=>t==='A'?'B':t==='B'?'A':undefined;m={...m,a:m.b,b:m.a,keepers:m.keepers.map(k=>k.team?{...k,team:flip(k.team)}:k)};}
 if(keep||swap)m={...m,anchored:true};
 if(keepers.length){
  const rest=free.filter((_,i)=>!keepers.includes(samples[i]));
  const r=fitTeamModel([...rest,...keepers.map(s=>({jersey:s.jersey,team:s.team,role:'goalkeeper' as const,weight:s.weight}))],m);
  if(r.a&&r.b)m={...r,anchored:m.anchored};
 }
 return m;
}

// Detection -> field filter -> local tracker -> role/team -> appearance re-ID -> global identities.
export class SoccerTracker{
 readonly options:SoccerOptions;
 // Durations of the last step in ms (pitch analysis and descriptors are reported separately).
 lastTiming={total:0,pitch:0,describe:0};
 private local=new LocalTracker<ScoredDetection>();
 private ids:IdentityManager;
 private pitch?:PitchModel;
 private size?:SizeModel;
 private model:TeamModel=emptyTeamModel;
 private fitAt=-Infinity;
 private lastTime?:number;
 // Stabilized ("camera-compensated") coordinates: image = S * stab + D within one camera segment.
 private segment=0;private S=1;private D={x:0,y:0};private drift=0;
 constructor(options:Partial<SoccerOptions>={},saved:SavedIdentity[]=[]){this.options=soccerOptions(options);this.ids=new IdentityManager(this.options,saved);}
 get teamModel():TeamModel{return this.model;}
 // Seed appearance for a roster identity from a user-confirmed label on an earlier frame.
 rememberLabel(playerId:string,team:'A'|'B'|'ref',frame:Frame,box:Box,time:number):void{
  if(!validBox(box)||!frame?.width||!frame?.height)return;
  this.ids.remember(playerId,team,describe(frame,box),box,time);this.fitAt=-Infinity;
 }
 step(input:SoccerFrameInput):SoccerFrameResult{
  const clock=now(),o=this.options,time=input.time,roster:RosterSlot[]=Array.isArray(input.roster)?input.roster:[];
  const frame:Frame=input.frame?.width>0&&input.frame.height>0&&input.frame.data?.length>=input.frame.width*input.frame.height*4?input.frame:{width:1,height:1,data:new Uint8ClampedArray(4)};
  let camera=input.camera??stillCamera;
  // Time going backwards or a long gap (seek) is treated like a camera cut.
  if(this.lastTime!==undefined&&(time<this.lastTime-1e-6||time-this.lastTime>3))camera={...camera,cut:true};
  if(camera.cut){this.segment++;this.S=1;this.D={x:0,y:0};this.drift=0;}
  else if(this.lastTime!==undefined){if(camera.reliable&&camera.scale>0){const s=camera.scale;this.S*=s;this.D={x:this.D.x*s+camera.dx,y:this.D.y*s+camera.dy};}else this.drift++;}
  const S=this.S,D=this.D,stabilize=(b:Box):Stab=>{const f=footPoint(b);return {x:(f.x-D.x)/S,y:(f.y-D.y)/S,h:b.h/S};};
  const t0=now(),pitch=analyzePitch(frame,time,this.pitch,camera),tPitch=now()-t0;this.pitch=pitch;
  const persons=(input.detections||[]).filter(d=>d&&d.kind==='person'&&validBox(d.box)&&Number.isFinite(d.score)).map(d=>d.appearance?.length?d:{...d,appearance:shirtFeature(frame,d.box)});
  const assessed=assessDetections(persons,pitch,o.field,this.size);this.size=assessed.size;
  // People just beyond a touchline (not audience) may continue an existing track, never start one.
  const rejected:RejectedDetection[]=assessed.rejected.map(r=>({...r})),byBox=new Map(persons.map(p=>[p.box,p])),continuation:ScoredDetection[]=[],held=new Map<Box,RejectedDetection>();
  for(const r of rejected){if(r.reason!=='outside-pitch')continue;const p=byBox.get(r.box);if(!p)continue;continuation.push({...p,foot:footPoint(p.box),zone:'outside',outsideBy:zoneOf(pitch,p.box,o.field).outsideBy,sizeRatio:1});held.set(p.box,r);}
  const loc=this.local.step(assessed.accepted,time,camera,continuation);
  for(const s of loc.samples){const r=s.detection&&held.get(s.detection.box);if(r)r.track=s.id;}
  const t1=now(),boxes=persons.map(p=>p.box),current=new Map<number,Descriptor>();
  for(const s of loc.samples)if(s.evidence==='detection')current.set(s.id,describe(frame,s.box,boxes));
  // User anchors: bind to the local track overlapping the label (IoU > .5) or start one from the label box.
  const slots=new Map(roster.map(s=>[s.id,s])),latest=new Map<string,Box>(),anchors:{playerId:string;track:number;box:Box;descriptor?:Descriptor}[]=[],taken=new Set<number>(),spawned=new Set<number>();
  for(const a of input.anchors||[])if(a&&validBox(a.box)&&slots.get(a.playerId)?.active)latest.set(a.playerId,a.box);
  for(const [playerId,box] of latest){
   let best=-1,score=.5;
   for(const t of this.local.tracks){const id=localId(t.id),v=iou(t.box,box);if(!taken.has(id)&&v>score){score=v;best=id;}}
   if(best<0){best=this.local.spawn(box,time,shirtFeature(frame,box));spawned.add(best);loc.samples.push({id:best,box,score:1,evidence:'detection',spawned:true,recovered:false});}
   taken.add(best);const d=describe(frame,box,boxes);if(!current.has(best)||spawned.has(best))current.set(best,d);
   const slot=slots.get(playerId)!;this.ids.bankSample(playerId,slot.team,slot.team==='ref'?'referee':slot.role??'player',d,time);
   anchors.push({playerId,track:best,box,descriptor:d});
  }
  const tDescribe=now()-t1;
  // Team model: refit about once a second (every step until two kits are known, and right after labels).
  if(time-this.fitAt>=(this.model.a&&this.model.b&&!anchors.length?1:.15)||time<this.fitAt){this.model=fitKits(this.ids.teamSamples(current),this.model);this.fitAt=time;}
  const samples:StepSample[]=loc.samples.map(s=>{
   const det=s.detection,observed=s.evidence==='detection',descriptor=observed?current.get(s.id):undefined;
   const pitchPt=input.homography?toPitch(input.homography,s.box):undefined;
   // Heavily occluded / tiny crops do not vote on the kit (they would look like an outlier, e.g. a referee).
   return {id:s.id,box:s.box,score:s.score,evidence:s.evidence,zone:det?det.zone:zoneOf(pitch,s.box,o.field).zone,...(descriptor?{descriptor,...(descriptor.quality>=MIN_VOTE_QUALITY?{vote:kitVote(this.model,descriptor)}:{})}:{}),
    occluded:overlapsAny(s.box,boxes,pitch.aspect),nearGoal:nearGoal(pitch,s.box,pitchPt),stab:stabilize(s.box),...(pitchPt?{pitch:pitchPt}:{})};
  });
  const res=this.ids.update({time,roster,samples,ended:loc.ended.map(e=>({id:e.id,cut:e.cut})),anchors,model:this.model,pitchReliable:pitch.reliable,filterEnabled:o.field.enabled,aspect:pitch.aspect||16/9,segment:this.segment,drift:this.drift,stabilize});
  for(const id of res.drop)this.local.remove(id);
  const debug:DebugFrame={time,pitch,tracks:res.tracks,rejected:[...rejected,...res.dropped]};
  this.lastTime=time;
  this.lastTiming={total:r3(now()-clock),pitch:r3(tPitch),describe:r3(tDescribe)};
  return {observations:res.observations,debug,events:res.events,issues:res.issues};
 }
 snapshot():SavedIdentity[]{return this.ids.snapshot();}
 // Sanity diagnostics (duplicate-looking identities, too many identities per team, etc.).
 diagnostics(time:number):ReidEvent[]{return this.ids.sanity(time);}
 // Read-only view of the live track at a box (for explainFrame); undefined when none overlaps.
 explainTrack(box:Box):DebugTrack|undefined{return this.ids.trackAt(box);}
}

// Stateless explanation of one frame for the debug view (no identities are changed).
export function explainFrame(frame:Frame,detections:TrackDetection[],time:number,roster:RosterSlot[],options:Partial<SoccerOptions>={},tracker?:SoccerTracker):DebugFrame{
 const o=soccerOptions(options),pitch=analyzePitch(frame,time);
 const persons=(detections||[]).filter(d=>d&&d.kind==='person'&&validBox(d.box)&&Number.isFinite(d.score));
 const {accepted,rejected}=assessDetections(persons,pitch,o.field);
 const model=tracker?.teamModel??emptyTeamModel,boxes=persons.map(p=>p.box),active=new Set((roster||[]).filter(s=>s.active).map(s=>s.id));
 const tracks:DebugTrack[]=accepted.map(d=>{
  const v=kitVote(model,describe(frame,d.box,boxes));
  const kit=!model.a||!model.b?'no team model yet':v.team?`kit vote ${v.team} (A ${v.distA.toFixed(2)}, B ${v.distB.toFixed(2)})`:v.outlier?`no team kit (A ${v.distA.toFixed(2)}, B ${v.distB.toFixed(2)}, referee ${v.distRef.toFixed(2)})`:`kit undecided (A ${v.distA.toFixed(2)}, B ${v.distB.toFixed(2)})`;
  const known=tracker?.explainTrack(d.box);
  if(known&&(!known.playerId||active.has(known.playerId)))return {...known,box:d.box,zone:d.zone,reason:[known.reason,kit].filter(Boolean).join('; ')};
  return {track:0,box:d.box,label:v.team?`${v.team}?`:'CAND',...(v.team?{team:v.team}:{}),roleLabel:'CANDIDATE',state:'candidate',identity:0,zone:d.zone,reason:`${kit}; score ${d.score.toFixed(2)}${d.sizeRatio!==1?`, size ${d.sizeRatio.toFixed(2)}x`:''}`};
 });
 return {time,pitch,tracks,rejected};
}
