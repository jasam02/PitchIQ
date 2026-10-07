// GLOBAL IDENTITY MANAGER. Sits above the local tracker: local track ids are temporary, global identities
// (roster slots such as A7 or R1) persist for the whole match.
// Rules: identities are never deleted when a player disappears (missing / offscreen); a new local track is never
// treated as a new player before re-identification has been tried; ambiguous matches are deferred, not guessed;
// one identity has at most one live track (exclusive assignment); identities are never swapped on proximity alone.
import {iou} from '../detection-core';
import {edgeOf} from '../player-recovery';
import {addToGallery,decodeDescriptor,encodeDescriptor,galleryDistance,MIN_GALLERY_QUALITY} from './appearance';
import {accumulateRole,decideRole,emptyRoleEvidence} from './classify';
import {PITCH_LENGTH,PITCH_WIDTH} from './homography';
import {globalLabel,roleLabel} from './types';
import type {Box,DebugTrack,Descriptor,FieldZone,GallerySample,IdentityStatus,KitVote,LocalState,Pt,ReidEvent,ReidScores,RejectedDetection,Role,RoleDecision,RoleEvidence,RosterSlot,SavedIdentity,SoccerObservation,SoccerOptions,Team,TeamModel} from './types';

// Final re-ID score weights (re-normalized over the components that are known for a pair).
export const REID_WEIGHTS={appearance:.35,uniform:.1,spatial:.3,movement:.15,temporal:.1};
export const PLAYER_HEIGHT=1.8; // metres; converts image distances when there is no homography
const BUFFER=50,DESCS=6,HIST=10,WINDOW=10,VOTES=15,MAX_EVENTS=60,MAX_SAVED=40,MAX_REFEREES=3;
const DEFER_EVERY=2,SANITY_EVERY=5,SANITY_REPEAT=30,SWAP_CAP=2,CLEAN=.4,KIT_MISMATCH=.6,KIT_MISSES=3;
// An identity whose track has had no detection for this long (only coasting) may be re-identified on a new track.
const LOST=.35;

const clamp=(v:number,a=0,b=1)=>v<a?a:v>b?b:v;
const r3=(v:number)=>Math.round(v*1000)/1000;
const pct=(v:number)=>`${Math.round(clamp(v)*100)}%`;
const avg=(v:number[])=>v.length?v.reduce((a,b)=>a+b,0)/v.length:0;
const defined=<T,>(v:(T|undefined)[])=>v.filter((x):x is T=>x!==undefined);
// Part distance (0 same .. 1 different) -> similarity. Same kit / same person stays below ~.25.
const similarity=(d:number)=>clamp(1-d/.6);
const validBox=(b:Box|undefined):b is Box=>!!b&&[b.x,b.y,b.w,b.h].every(Number.isFinite)&&b.w>0&&b.h>0;
const finitePt=(p:Pt|undefined):p is Pt=>!!p&&Number.isFinite(p.x)&&Number.isFinite(p.y);
const centreClose=(a:Box,b:Box,aspect:number)=>Math.hypot((a.x+a.w/2-b.x-b.w/2)*aspect,a.y+a.h/2-b.y-b.h/2)<.6*(a.w+b.w)/2*aspect;
const statuses:IdentityStatus[]=['active','missing','offscreen','unknown','substituted'];
const roles:Role[]=['player','goalkeeper','referee','unknown'];

export type Stab={x:number;y:number;h:number}; // camera-compensated foot point and box height (segment units)
export type GlobalPlayer={
 playerId:string;role:Role;team?:Team;status:IdentityStatus;gallery:GallerySample[];lastSeen:number;lastBox:Box;lastPitch?:Pt;exitEdge:string;velocity:Pt;
 identityConfidence:number;teamConfidence:number;roleConfidence:number;anchored:boolean;
 // runtime only
 track?:number;runner?:boolean;crossedAt:number;here:boolean;segment?:number;stab?:Stab;drift:number;pitchVelocity?:Pt;pendingSpatial:boolean;lastObserved:number;capUntil:number;anchoredAt:number;boundAt:number;
};
type Scored={appearance?:number;jersey:number;uniform?:number;team:boolean;spatial?:number;movement?:number;temporal:number;final:number;missing:number};
type Summary={playerId:string;n:number;final:number;appearance?:number;jersey:number;uniform?:number;team:boolean;spatial?:number;spatialShare:number;movement?:number;temporal:number;missing:number};
export type TrackState={
 id:number;born:number;bornSegment:number;firstStab:Stab;firstPitch?:Pt;firstZone:FieldZone;entryEdge:string;
 hits:number;inside:number;boundary:number;outside:number;outsideRun:number;
 evidence:RoleEvidence;decision?:RoleDecision;teamVotes:string[];descs:{d:Descriptor;time:number;occluded:boolean}[];current?:Descriptor;occluded:boolean;
 box:Box;score:number;kind:'detection'|'predicted';seenAt:number;zone:FieldZone;stab:Stab;pitch?:Pt;segment:number;velocity:Pt;pitchVelocity?:Pt;hist:{t:number;x:number;y:number}[];
 lastBox:Box;lastTime:number;lastScore:number;lastPitch?:Pt;
 state:LocalState;playerId?:string;team?:Team;role:Role;teamConfidence:number;roleConfidence:number;retired:boolean;runner?:boolean;reacquired:boolean;crossedAt:number;kitMiss:number;
 buffer:SoccerObservation[];hold:{o:SoccerObservation;heavy:boolean}[];reid:Map<string,Scored[]>;best:number;lastDeferred:number;deferrals:number;anchoredAt:number;reason:string;
};
export type StepSample={id:number;box:Box;score:number;evidence:'detection'|'predicted';zone:FieldZone;descriptor?:Descriptor;occluded:boolean;vote?:KitVote;nearGoal:boolean;stab:Stab;pitch?:Pt};
export type StepContext={
 time:number;roster:RosterSlot[];samples:StepSample[];ended:{id:number;cut:boolean}[];
 anchors:{playerId:string;track:number;box:Box;descriptor?:Descriptor}[];
 model:TeamModel;pitchReliable:boolean;filterEnabled:boolean;aspect:number;
 segment:number; // camera segment (changes on every cut); stabilized coordinates are comparable within one segment
 drift:number; // steps with an unreliable camera estimate in this segment (stabilized positions drift)
 stabilize:(box:Box)=>Stab;
};
// drop: local tracks the caller must remove (a coasting ghost whose identity moved to a new track).
export type StepOutput={observations:SoccerObservation[];events:ReidEvent[];issues:{playerId:string;time:number;reason:string}[];tracks:DebugTrack[];dropped:RejectedDetection[];drop:number[]};
// hard: one of the two was undetected or they overlapped heavily at some point (continuity is then a guess).
// a is always confirmed; b may be an unidentified track (ids[1] undefined).
type Pair={a:number;b:number;ids:[string,string|undefined];start:number;sep:number;hard:boolean;snap:Map<number,{x:number;y:number;t:number;h:number;vx:number;vy:number}>};
export type TeamSample={jersey:number[];team?:Team;role?:Role;weight?:number};

const slotRole=(s:RosterSlot):Role=>s.team==='ref'?'referee':s.role&&s.role!=='unknown'?s.role:'player';
const keeperish=(s:RosterSlot)=>s.role==='goalkeeper'||(!s.role&&s.number==='1');
function describeRole(role:Role,team?:Team){return role==='goalkeeper'&&!team?'GOALKEEPER (team unknown)':role==='unknown'?'UNKNOWN':roleLabel(role,team);}
function zoneRatio(t:TrackState){const n=t.inside+t.boundary+t.outside;return n?(t.inside+.5*t.boundary)/n:0;}

// Mean of the last WINDOW scored steps for one (track, identity) pair.
function summarize(playerId:string,steps:Scored[]):Summary{
 const m=(k:'appearance'|'uniform'|'spatial'|'movement')=>{const v=defined(steps.map(s=>s[k]));return v.length?avg(v):undefined;};
 const last=steps[steps.length-1];
 return {playerId,n:steps.length,final:avg(steps.map(s=>s.final)),appearance:m('appearance'),jersey:avg(steps.map(s=>s.jersey)),uniform:m('uniform'),team:steps.every(s=>s.team),spatial:m('spatial'),spatialShare:steps.filter(s=>s.spatial!==undefined).length/Math.max(1,steps.length),movement:m('movement'),temporal:avg(steps.map(s=>s.temporal)),missing:last?last.missing:0};
}
function scoresOf(s:Summary,second?:Summary):ReidScores{
 return {appearance:r3(s.appearance??.5),jersey:r3(s.jersey),uniform:r3(s.uniform??.5),team:s.team,spatial:r3(s.spatial??.5),movement:r3(s.movement??.5),temporal:r3(s.temporal),final:r3(s.final),...(second?{secondBest:r3(second.final)}:{}),missingSeconds:r3(s.missing)};
}
// Weighted mean over known components; a team mismatch is a hard gate (score 0).
export function combineReid(c:{appearance?:number;uniform?:number;spatial?:number;movement?:number;temporal?:number;team:boolean}){
 if(!c.team)return 0;let s=0,w=0;
 for(const k of Object.keys(REID_WEIGHTS) as (keyof typeof REID_WEIGHTS)[]){const v=c[k];if(v!==undefined&&Number.isFinite(v)){s+=REID_WEIGHTS[k]*v;w+=REID_WEIGHTS[k];}}
 return w?s/w:0;
}

export class IdentityManager{
 readonly registry=new Map<string,GlobalPlayer>();
 readonly tracks=new Map<number,TrackState>();
 private pairs=new Map<string,Pair>();
 private together=new Set<string>(); // same-team identity pairs ever visible at the same time
 private limits=new Map<string,number>();
 private bank=new Map<string,{jersey:number[];time:number;team?:Team;role?:Role}[]>(); // user-confirmed jersey samples per identity
 private roster:RosterSlot[]=[];
 private time=0;
 constructor(public options:SoccerOptions,saved:SavedIdentity[]=[]){for(const s of Array.isArray(saved)?saved:[])this.restore(s);}

 // ---------- persistence ----------
 private restore(s:SavedIdentity){
  if(!s||typeof s.playerId!=='string'||!s.playerId||this.registry.has(s.playerId))return;
  const box=validBox(s.lastBox)?{...s.lastBox}:{x:.5,y:.5,w:.01,h:.02},lastSeen=Number.isFinite(s.lastSeen)?s.lastSeen:0;
  const decoded=defined((Array.isArray(s.gallery)?s.gallery:[]).map(v=>decodeDescriptor(v))).slice(-6);
  const gallery=decoded.map((d,i)=>({d,time:lastSeen-(decoded.length-1-i)}));
  const status:IdentityStatus=s.status==='active'||!statuses.includes(s.status)?(edgeOf(box)?'offscreen':'missing'):s.status;
  const conf=(v:number)=>Number.isFinite(v)?clamp(v):0;
  this.registry.set(s.playerId,{playerId:s.playerId,role:roles.includes(s.role)?s.role:'unknown',...(s.team==='A'||s.team==='B'?{team:s.team}:{}),status,gallery,lastSeen,lastBox:box,...(finitePt(s.lastPitch)?{lastPitch:{...s.lastPitch}}:{}),
   exitEdge:String(s.exitEdge||'').slice(0,8),velocity:finitePt(s.velocity)?{...s.velocity}:{x:0,y:0},identityConfidence:conf(s.identityConfidence),teamConfidence:conf(s.teamConfidence),roleConfidence:conf(s.roleConfidence),anchored:!!s.anchored,
   drift:0,pendingSpatial:true,lastObserved:lastSeen,capUntil:0,anchoredAt:-Infinity,boundAt:-Infinity,crossedAt:-Infinity,here:false});
 }
 snapshot():SavedIdentity[]{
  let list=[...this.registry.values()];
  if(list.length>MAX_SAVED){const keep=new Set([...list].sort((a,b)=>b.lastSeen-a.lastSeen).slice(0,MAX_SAVED));list=list.filter(g=>keep.has(g));}
  const r6=(v:number)=>Number.isFinite(v)?Math.round(v*1e6)/1e6:0;
  return list.map(g=>({playerId:g.playerId,role:g.role,...(g.team?{team:g.team}:{}),status:g.status,gallery:g.gallery.slice(-6).map(s=>encodeDescriptor(s.d)),lastSeen:r3(Math.max(0,g.lastSeen)),
   lastBox:{x:r6(g.lastBox.x),y:r6(g.lastBox.y),w:Math.max(.001,r6(g.lastBox.w)),h:Math.max(.001,r6(g.lastBox.h))},...(g.lastPitch?{lastPitch:{x:r6(g.lastPitch.x),y:r6(g.lastPitch.y)}}:{}),
   exitEdge:g.exitEdge.slice(0,8),velocity:{x:r6(g.velocity.x),y:r6(g.velocity.y)},identityConfidence:r3(clamp(g.identityConfidence)),teamConfidence:r3(clamp(g.teamConfidence)),roleConfidence:r3(clamp(g.roleConfidence)),anchored:g.anchored}));
 }
 // A user-confirmed label on an earlier frame: seeds the identity's gallery and the team model.
 remember(playerId:string,team:Team|'ref',d:Descriptor,box:Box,time:number){
  if(!playerId||!validBox(box)||!Number.isFinite(time))return;
  let g=this.registry.get(playerId);
  if(!g){g=this.blank(playerId,team==='ref'?'referee':'player',team==='ref'?undefined:team,box,time);g.status='missing';this.registry.set(playerId,g);}
  if(g.track===undefined&&time>=g.lastSeen){g.lastSeen=time;g.lastBox={...box};g.lastObserved=Math.max(g.lastObserved,time);g.pendingSpatial=true;g.segment=undefined;g.exitEdge='';}
  g.anchored=true;this.learn(g,d,time,true);
 }
 private blank(playerId:string,role:Role,team:Team|undefined,box:Box,time:number):GlobalPlayer{
  return {playerId,role,...(team?{team}:{}),status:'missing',gallery:[],lastSeen:time,lastBox:{...box},exitEdge:'',velocity:{x:0,y:0},identityConfidence:0,teamConfidence:0,roleConfidence:0,anchored:false,drift:0,pendingSpatial:false,lastObserved:-Infinity,capUntil:0,anchoredAt:-Infinity,boundAt:time,crossedAt:-Infinity,here:false};
 }
 private learn(g:GlobalPlayer,d:Descriptor|undefined,time:number,user:boolean){
  if(!d)return;
  // A user-confirmed crop is trusted even when slightly small or blurred.
  const q=user&&d.quality>=.2?Math.max(d.quality,MIN_GALLERY_QUALITY):d.quality;
  g.gallery=addToGallery(g.gallery,{...d,quality:q},time,this.options.gallerySize);
  if(user)this.bankSample(g.playerId,g.role==='referee'?'ref':g.team,g.role,d,time);
 }
 // User-confirmed jersey sample for the team model (callable before the identity is bound this step).
 bankSample(playerId:string,team:Team|'ref'|undefined,role:Role,d:Descriptor|undefined,time:number){
  if(!d||!d.jersey.some(v=>v>0))return;
  const list=(this.bank.get(playerId)||[]).filter(s=>Math.abs(s.time-time)>.05);
  list.push({jersey:d.jersey,time,...(team&&team!=='ref'?{team}:{}),role:team==='ref'?'referee':role});list.sort((a,b)=>a.time-b.time);
  this.bank.delete(playerId);this.bank.set(playerId,list.slice(-4));
  if(this.bank.size>80)this.bank.delete(this.bank.keys().next().value as string);
 }
 // Jersey samples for fitTeamModel: user-confirmed samples carry their team (anchored model); live tracks are
 // unlabelled, with role hints from confirmed referees/goalkeepers.
 teamSamples(current:Map<number,Descriptor>):TeamSample[]{
  const out:TeamSample[]=[];
  for(const [id,list] of this.bank){
   const g=this.registry.get(id),role=g?g.role:list[list.length-1]?.role,team=g?g.team:list[list.length-1]?.team;
   for(const s of list.slice(-2))out.push({jersey:s.jersey,...(role!=='referee'&&team?{team}:{}),...(role==='referee'||role==='goalkeeper'?{role}:{}),weight:1.5});}
  const ids=new Set([...this.tracks.keys(),...current.keys()]);
  for(const id of [...ids].sort((a,b)=>a-b)){
   const t=this.tracks.get(id),d=current.get(id)??t?.descs[t.descs.length-1]?.d;
   if(!d||d.quality<.25||!d.jersey.some(v=>v>0)||t?.retired||t?.state==='rejected'||t?.zone==='outside')continue;
   const g=t?.playerId?this.registry.get(t.playerId):undefined,role=t?.state==='confirmed'?(g?.role??t.role):undefined;
   out.push({jersey:d.jersey,...(role==='referee'||role==='goalkeeper'?{role}:{}),weight:d.quality});
  }
  return out;
 }

 // ---------- per step ----------
 update(ctx:StepContext):StepOutput{
  const out:StepOutput={observations:[],events:[],issues:[],tracks:[],dropped:[],drop:[]};
  this.time=ctx.time;this.roster=ctx.roster;
  const slots=new Map(ctx.roster.map(s=>[s.id,s]));
  this.syncRoster(ctx,slots);
  for(const e of ctx.ended)this.endTrack(e.id,out);
  const seen=new Set<number>();
  for(const s of ctx.samples){this.observe(s,ctx);seen.add(s.id);}
  for(const a of ctx.anchors)this.anchor(a,ctx,slots,out);
  const live=[...this.tracks.values()].filter(t=>seen.has(t.id)).sort((a,b)=>a.id-b.id);
  for(const t of live){this.checkTeam(t,ctx,out);this.checkKeeper(t,ctx,out);}
  for(const t of live)if(t.state!=='confirmed'&&t.state!=='uncertain')this.promote(t,ctx,out);
  this.resolve(live.filter(t=>t.state==='uncertain'),ctx,slots,out);
  this.guardSwaps(live.filter(t=>t.state!=='rejected'&&!t.retired),ctx,out);
  this.checkReacquired(live,ctx,out);
  this.refresh(live,ctx,slots,out);
  out.tracks=live.map(t=>this.debugTrack(t));
  if(this.allow('sanity-scan',ctx.time,SANITY_EVERY))for(const e of this.sanity(ctx.time))if(this.allow('sanity:'+e.message.slice(0,60),ctx.time,SANITY_REPEAT))out.events.push(e);
  if(out.events.length>MAX_EVENTS)out.events.splice(0,out.events.length-MAX_EVENTS);
  return out;
 }
 // Substitutions (inactive roster slot) and roster-defined roles. Restored identities may use their last box
 // as a spatial reference only when they were seen within .3 s of the first step.
 private syncRoster(ctx:StepContext,slots:Map<string,RosterSlot>){
  for(const g of this.registry.values()){
   if(g.pendingSpatial){g.pendingSpatial=false;if(g.track===undefined&&Math.abs(ctx.time-g.lastSeen)<.3){g.segment=ctx.segment;g.stab=ctx.stabilize(g.lastBox);g.drift=ctx.drift;}}
   const slot=slots.get(g.playerId);if(!slot)continue;
   if(slot.team==='ref'){g.role='referee';delete g.team;}else{g.team=slot.team;if(slot.role&&slot.role!=='unknown'&&slot.role!=='referee')g.role=slot.role;else if(g.role==='referee'||g.role==='unknown')g.role='player';}
   if(!slot.active&&g.status!=='substituted'){
    const t=g.track!==undefined?this.tracks.get(g.track):undefined;
    if(t){t.playerId=undefined;t.state='unknown';t.retired=true;t.reason=`${globalLabel(g.playerId)} substituted; this track is not re-identified`;t.buffer=[];}
    g.track=undefined;g.status='substituted';
   }else if(slot.active&&g.status==='substituted')g.status=g.exitEdge?'offscreen':'missing';
  }
 }
 private endTrack(id:number,out:StepOutput){
  const t=this.tracks.get(id);if(!t)return;
  this.tracks.delete(id);this.limits.delete('team-'+id);
  this.dropPairs(id,out);
  this.releaseHold(t,out);
  const g=t.playerId?this.registry.get(t.playerId):undefined;
  if(g&&g.track===id){
   // Never deleted: missing, or offscreen when the last box touched an image edge.
   const edge=edgeOf(t.lastBox);g.track=undefined;g.exitEdge=edge;if(g.status!=='substituted')g.status=edge?'offscreen':'missing';return;
  }
  if(t.state==='candidate'||t.state==='unknown'||t.state==='rejected'){
   if(t.hits)out.dropped.push({box:t.lastBox,score:t.lastScore,reason:'low-confidence',detail:`track ${id} ended after ${t.hits} detection${t.hits===1?'':'s'} before promotion${t.reason?` (${t.reason})`:''}`.slice(0,200),track:id});
  }else if(t.state==='uncertain'&&!t.retired)out.events.push({time:this.time,kind:'deferred',track:id,message:`Track ${id} (${describeRole(t.role,t.team)}) ended before its identity could be decided; its observations were not saved.`});
 }
 private observe(s:StepSample,ctx:StepContext){
  let t=this.tracks.get(s.id);
  if(!t){t={id:s.id,born:ctx.time,bornSegment:ctx.segment,firstStab:s.stab,...(s.pitch?{firstPitch:s.pitch}:{}),firstZone:s.evidence==='detection'?s.zone:'unknown',entryEdge:edgeOf(s.box),hits:0,inside:0,boundary:0,outside:0,outsideRun:0,
   evidence:emptyRoleEvidence(),teamVotes:[],descs:[],occluded:false,box:s.box,score:s.score,kind:s.evidence,seenAt:ctx.time,zone:s.zone,stab:s.stab,segment:ctx.segment,velocity:{x:0,y:0},hist:[],
   lastBox:s.box,lastTime:ctx.time,lastScore:s.score,state:'candidate',role:'unknown',teamConfidence:0,roleConfidence:0,retired:false,reacquired:false,crossedAt:-Infinity,kitMiss:0,buffer:[],hold:[],reid:new Map(),best:0,lastDeferred:-Infinity,deferrals:0,anchoredAt:-Infinity,reason:''};
   this.tracks.set(s.id,t);}
  t.box=s.box;t.score=s.score;t.kind=s.evidence;t.seenAt=ctx.time;t.stab=s.stab;t.pitch=s.pitch;t.current=undefined;t.occluded=s.occluded;
  if(t.segment!==ctx.segment){t.segment=ctx.segment;t.hist=[];t.velocity={x:0,y:0};delete t.pitchVelocity;}
  t.reacquired=false;
  if(s.evidence!=='detection')return;
  t.reacquired=t.hits>0&&ctx.time-t.lastTime>.35;
  t.zone=s.zone;t.hits++;
  if(s.zone==='inside')t.inside++;else if(s.zone==='boundary')t.boundary++;else if(s.zone==='outside')t.outside++;
  t.outsideRun=s.zone==='outside'?t.outsideRun+1:0;
  if(s.descriptor){t.current=s.descriptor;if(s.descriptor.quality>=.2){t.descs.push({d:s.descriptor,time:ctx.time,occluded:s.occluded});if(t.descs.length>DESCS)t.descs.shift();}}
  if(s.vote){t.evidence=accumulateRole(t.evidence,s.vote,{nearGoal:s.nearGoal,zone:s.zone});t.teamVotes.push(s.vote.team??(s.vote.outlier?'x':'-'));if(t.teamVotes.length>VOTES)t.teamVotes.shift();}
  const prev=t.hist[t.hist.length-1];
  if(prev){const dt=ctx.time-prev.t;if(dt>1e-3&&dt<=1.5){const vx=(s.stab.x-prev.x)/dt,vy=(s.stab.y-prev.y)/dt;t.velocity=t.hist.length>1?{x:.6*t.velocity.x+.4*vx,y:.6*t.velocity.y+.4*vy}:{x:vx,y:vy};}}
  t.hist.push({t:ctx.time,x:s.stab.x,y:s.stab.y});if(t.hist.length>HIST)t.hist.shift();
  if(s.pitch&&t.lastPitch&&ctx.time-t.lastTime>1e-3&&ctx.time-t.lastTime<=1.5){const dt=ctx.time-t.lastTime,vx=(s.pitch.x-t.lastPitch.x)/dt,vy=(s.pitch.y-t.lastPitch.y)/dt;t.pitchVelocity=t.pitchVelocity?{x:.6*t.pitchVelocity.x+.4*vx,y:.6*t.pitchVelocity.y+.4*vy}:{x:vx,y:vy};}
  else if(!s.pitch)delete t.pitchVelocity;
  t.lastBox=s.box;t.lastTime=ctx.time;t.lastScore=s.score;t.lastPitch=s.pitch;
 }
 // User labels override automatic identities: bind, unbind any previous holder, seed gallery + team model.
 private anchor(a:StepContext['anchors'][number],ctx:StepContext,slots:Map<string,RosterSlot>,out:StepOutput){
  const slot=slots.get(a.playerId),t=this.tracks.get(a.track);if(!slot||!slot.active||!t)return;
  let g=this.registry.get(a.playerId);
  if(!g){g=this.blank(a.playerId,slotRole(slot),slot.team==='ref'?undefined:slot.team,a.box,ctx.time);this.registry.set(g.playerId,g);}
  const changed=t.playerId!==g.playerId||g.track!==t.id,notes:string[]=[];
  if(g.track!==undefined&&g.track!==t.id){const old=this.tracks.get(g.track);if(old&&old.playerId===g.playerId){this.release(old,`user label moved ${globalLabel(g.playerId)} to track ${t.id}`);notes.push(`was on track ${old.id}`);}}
  if(t.playerId&&t.playerId!==g.playerId){const y=this.registry.get(t.playerId);if(y&&y.track===t.id){this.unbind(y,true);notes.push(`replaces automatic ${globalLabel(y.playerId)}`);}}
  t.playerId=g.playerId;t.state='confirmed';t.team=g.team;t.role=g.role;t.anchoredAt=ctx.time;t.reid.clear();t.retired=false;t.reason='user label';t.teamVotes=[];t.teamConfidence=1;t.roleConfidence=1;
  if(changed)g.boundAt=ctx.time;
  g.track=t.id;g.status='active';g.here=true;g.anchored=true;g.anchoredAt=ctx.time;g.identityConfidence=1;g.teamConfidence=1;g.roleConfidence=1;g.capUntil=0;g.crossedAt=-Infinity;t.crossedAt=-Infinity;
  this.learn(g,a.descriptor,ctx.time,true);
  if(changed){this.flush(t,g,ctx.time,out);out.events.push({time:ctx.time,kind:'anchor',track:t.id,playerId:g.playerId,message:`User label binds ${globalLabel(g.playerId,true)} to track ${t.id}${notes.length?` (${notes.join('; ')})`:''}.`});}
 }
 // Identity leaves a track that turned out not to be it: drop what was learnt while bound, spatial unknown.
 private unbind(g:GlobalPlayer,wrong:boolean){
  if(wrong){g.gallery=g.gallery.filter(s=>s.time<g.boundAt);g.segment=undefined;}
  g.track=undefined;g.status='missing';g.exitEdge='';
 }
 private release(t:TrackState,why:string){
  t.playerId=undefined;t.state=t.decision&&t.role!=='unknown'?'uncertain':'candidate';t.reid.clear();t.buffer=[];t.reason=why;
 }
 // A labelled player in a keeper kit near a goal becomes the goalkeeper (unless the roster fixes the role),
 // so keeper prototypes with a team exist for later re-identification.
 private checkKeeper(t:TrackState,ctx:StepContext,out:StepOutput){
  const g=t.state==='confirmed'&&t.playerId?this.registry.get(t.playerId):undefined;
  if(!g||g.role!=='player'||t.hits%5!==0||t.evidence.votes.length<2*this.options.minHits)return;
  const slot=this.roster.find(s=>s.id===g.playerId);if(!slot||slot.role)return;
  const d=decideRole(t.evidence,ctx.model,this.options.minHits);
  if(d.role!=='goalkeeper'||(d.team&&d.team!==g.team)||d.roleConfidence<.6)return;
  g.role='goalkeeper';t.role='goalkeeper';
  out.events.push({time:ctx.time,kind:'role',track:t.id,playerId:g.playerId,message:`${globalLabel(g.playerId,true)} classified as goalkeeper (keeper kit near a goal, role ${pct(d.roleConfidence)}).`});
 }
 // Team switching needs >= 80% of the last 15 votes for the other kit.
 private checkTeam(t:TrackState,ctx:StepContext,out:StepOutput){
  if((t.state!=='confirmed'&&t.state!=='uncertain')||!t.team||t.role!=='player'||t.teamVotes.length<VOTES)return;
  const other:Team=t.team==='A'?'B':'A',n=t.teamVotes.filter(v=>v===other).length;if(n<.8*VOTES)return;
  const g=t.playerId?this.registry.get(t.playerId):undefined,msg=`Track ${t.id}${g?` (${globalLabel(g.playerId)})`:''} matched team ${other}'s kit in ${n} of the last ${VOTES} detections`;
  if(g&&(ctx.time-t.anchoredAt<10||ctx.time-g.anchoredAt<10)){
   if(this.allow('team-'+t.id,ctx.time,10)){out.events.push({time:ctx.time,kind:'team-change',track:t.id,playerId:g.playerId,message:`${msg}; kept because of a recent user label. Check this label.`});out.issues.push({playerId:g.playerId,time:ctx.time,reason:'Kit looks like the other team. Check this label.'});}
   return;
  }
  if(g&&g.track===t.id){this.unbind(g,false);g.segment=undefined;out.issues.push({playerId:g.playerId,time:ctx.time,reason:'Kit changed to the other team; identity released. Check this player.'});}
  t.playerId=undefined;t.team=other;t.state='uncertain';t.reid.clear();t.buffer=[];t.teamVotes=[];t.reason=`kit changed to team ${other}`;
  out.events.push({time:ctx.time,kind:'team-change',track:t.id,...(g?{playerId:g.playerId}:{}),message:`${msg}; ${g?'identity released and ':''}re-identifying as team ${other}.`});
 }
 // CANDIDATE -> role. Never from one frame: minHits detections, mostly inside the pitch, a temporal role decision.
 private promote(t:TrackState,ctx:StepContext,out:StepOutput){
  if(t.retired){t.state='unknown';return;}
  const o=this.options,d=decideRole(t.evidence,ctx.model,o.minHits),zr=zoneRatio(t);t.decision=d;
  const votes=t.evidence.votes,refShare=votes.length?votes.filter(v=>v.refLike).length/votes.length:0;
  // Assistant referees run the touchline: allowed only with an established referee kit and a ref-like majority.
  const runner=d.role==='referee'&&!!ctx.model.referee&&refShare>=.5,born=t.firstZone==='boundary',pitchOk=ctx.pitchReliable||!ctx.filterEnabled;
  const need=born?o.boundaryHits:o.minHits;
  let ok=pitchOk&&d.label!=='CANDIDATE'&&d.role!=='unknown'&&t.hits>=o.minHits;
  if(ok)ok=runner?t.hits>=need:zr>=.8&&(!born||(t.hits>=o.boundaryHits&&t.inside>=3));
  if(ok){
   t.state='uncertain';t.team=d.team;t.role=d.role;t.teamConfidence=d.teamConfidence;t.roleConfidence=d.roleConfidence;t.reason='';
   if(d.role==='referee')t.runner=runner&&zr<.8; // assistant referee on the touchline vs the central referee
   out.events.push({time:ctx.time,kind:'promotion',track:t.id,message:`Track ${t.id} promoted to ${describeRole(d.role,d.team)} after ${t.hits} detections (inside ${pct(zr)}, team ${pct(d.teamConfidence)}, role ${pct(d.roleConfidence)}).`});
   if(d.role!=='player')out.events.push({time:ctx.time,kind:'role',track:t.id,message:`Track ${t.id} classified as ${describeRole(d.role,d.team)}${runner?' (touchline runner in the referee kit)':''}.`});
   return;
  }
  t.state=t.hits>=3*o.minHits?(zr<.8&&!runner?'rejected':'unknown'):'candidate';
  t.reason=!pitchOk?'pitch not reliable on this frame: no promotion':t.state==='rejected'?`mostly outside the playable area (inside ${pct(zr)})`:t.state==='unknown'?`no consistent kit or role after ${t.hits} detections`:`${t.hits}/${need} detections, inside ${pct(zr)}${born?' (born on the boundary)':''}`;
 }
 // Seconds since the identity's bound track last had a detection (0 = observed now, Infinity = unbound).
 private lostFor(g:GlobalPlayer){const t=g.track!==undefined?this.tracks.get(g.track):undefined;return g.track===undefined||!t?Infinity:Math.max(0,this.time-t.lastTime);}
 private briefLost(t:TrackState){return [...this.registry.values()].find(g=>g.track!==undefined&&g.track!==t.id&&this.sameGroup(t,g)&&this.lostFor(g)>0&&this.lostFor(g)<LOST);}
 private sameGroup(t:TrackState,g:GlobalPlayer){return t.role==='referee'?g.role==='referee':g.role!=='referee'&&(!t.team||!g.team||g.team===t.team);}
 private candidatesFor(t:TrackState,slots:Map<string,RosterSlot>){
  const out:GlobalPlayer[]=[];
  for(const g of this.registry.values()){
   if(!slots.get(g.playerId)?.active||g.track===t.id)continue;
   if(g.track!==undefined?this.lostFor(g)<LOST:!(g.status==='missing'||g.status==='offscreen'||g.status==='unknown'))continue;
   if(t.role==='referee'){if(g.role==='referee'&&(g.runner===undefined||t.runner===undefined||g.runner===t.runner))out.push(g);continue;}
   if(g.role==='referee')continue;
   if(t.role==='goalkeeper'){if(g.role==='goalkeeper'||(!t.team||g.team===t.team)&&g.role==='unknown')out.push(g);continue;} // other-team keepers fail the team gate
   if(g.team===t.team&&(g.role==='player'||g.role==='unknown'))out.push(g);
  }
  return out;
 }
 private resolve(list:TrackState[],ctx:StepContext,slots:Map<string,RosterSlot>,out:StepOutput){
  const o=this.options;
  for(const t of list){
   if(t.retired)continue;
   const cands=this.candidatesFor(t,slots);
   if(!cands.length){t.reid.clear();t.best=0;this.createOrHold(t,ctx,slots,out);continue;}
   const ids=new Set(cands.map(g=>g.playerId));for(const k of [...t.reid.keys()])if(!ids.has(k))t.reid.delete(k);
   for(const g of cands){
    const s=this.score(t,g,ctx);if(s.appearance===undefined&&s.spatial===undefined)continue; // count/edges/time alone never decide
    const steps=t.reid.get(g.playerId)||[];steps.push(s);if(steps.length>WINDOW)steps.shift();t.reid.set(g.playerId,steps);
   }
  }
  // stranger: an unseen teammate (free roster slot) wears the same kit, so it scores the candidate's appearance
  // with neutral spatial / movement / temporal evidence. A reconnect must beat it like any other candidate.
  type Plan={t:TrackState;all:Summary[];need:number;stranger?:number};
  const plans:Plan[]=[];
  for(const t of list){
   if(t.state!=='uncertain'||!t.reid.size)continue;
   const all=[...t.reid].map(([id,s])=>summarize(id,s)).sort((a,b)=>b.final-a.final||a.playerId.localeCompare(b.playerId));
   const b=all[0],stranger=t.role!=='referee'&&this.freeSlot(t,slots)?combineReid({appearance:b.appearance,uniform:b.uniform,spatial:b.spatial!==undefined?.5:undefined,movement:b.movement!==undefined?.5:undefined,temporal:.5,team:true}):undefined;
   t.best=b.final;plans.push({t,all,need:b.spatialShare>=.5?3:5,...(stranger!==undefined?{stranger}:{})});
  }
  const lead=(p:Plan)=>p.all[0].final-Math.max(p.all[1]?.final??0,p.stranger??0);
  const decisive=(p:Plan)=>p.all[0].n>=p.need&&p.all[0].final>=o.reidMin&&lead(p)>=o.reidMargin;
  // Exclusive assignment: clearest margins first; a tie with another open track defers both.
  const open=new Set(plans),taken=new Set<string>(),why=new Map<Plan,string>();
  for(let pass=0;pass<3;pass++){
   let progress=false;
   for(const p of plans.filter(p=>open.has(p)&&decisive(p)).sort((a,b)=>lead(b)-lead(a)||a.t.id-b.t.id)){
    const best=p.all[0],lost=this.briefLost(p.t);
    // Another identity of this group just lost detection and is not a candidate yet: it could be this person.
    if(lost){why.set(p,`${globalLabel(lost.playerId)} just lost detection`);continue;}
    if(taken.has(best.playerId)){why.set(p,`${globalLabel(best.playerId)} was just assigned to another track`);continue;}
    const rival=plans.find(q=>q!==p&&open.has(q)&&q.all.some(s=>s.playerId===best.playerId&&s.n>=2&&s.final>=best.final-o.reidMargin));
    if(rival){why.set(p,`track ${rival.t.id} matches ${globalLabel(best.playerId)} almost as well`);continue;}
    taken.add(best.playerId);open.delete(p);why.delete(p);this.reconnect(p.t,p.all,ctx,out);progress=true;
   }
   if(!progress)break;
  }
  for(const p of plans){
   if(!open.has(p))continue;
   const best=p.all[0],implausible=(s:Summary)=>s.final<.3||(s.spatial!==undefined&&s.spatialShare>=.5&&s.spatial<.15)||(p.stranger!==undefined&&p.stranger-s.final>=o.reidMargin);
   if(p.all.every(s=>s.n>=p.need&&implausible(s))&&!this.briefLost(p.t)){
    const slot=this.freeSlot(p.t,slots);
    if(slot){for(const s of p.all.slice(0,3))out.events.push({time:ctx.time,kind:'reid-rejected',track:p.t.id,playerId:s.playerId,message:`Track ${p.t.id} is not ${globalLabel(s.playerId,true)}: ${s.final<.3?`score ${pct(s.final)}`:s.spatial!==undefined&&s.spatial<.15?`spatially implausible (${pct(s.spatial)})`:`an unseen teammate fits better (${pct(p.stranger??0)} vs ${pct(s.final)})`}.`,scores:scoresOf(s)});
     this.createIdentity(p.t,slot,ctx,out,'every missing teammate is implausible here');continue;}
   }
   if(best.n<p.need&&!why.has(p)){p.t.reason=`collecting re-ID evidence (${best.n}/${p.need})`;continue;}
   const reason=why.get(p)??(best.final<o.reidMin?`best score ${pct(best.final)} below ${pct(o.reidMin)}`:p.stranger!==undefined&&p.stranger>=(p.all[1]?.final??0)?`an unseen teammate would score ${pct(p.stranger)}`:`lead ${pct(lead(p))} below ${pct(o.reidMargin)}`);
   this.defer(p.t,p.all,ctx,out,reason);
  }
 }
 // Components in [0,1]; undefined = unknown (its weight is removed). Spatial and movement evidence compare where
 // the identity was last seen with where the new track first appeared (gap = time between the two).
 private score(t:TrackState,g:GlobalPlayer,ctx:StepContext):Scored{
  const gap=Math.max(0,t.born-g.lastSeen),k=Math.min(gap,2);
  const team=t.role==='referee'||!t.team||!g.team||t.team===g.team;
  const probes=t.descs.filter(x=>!x.occluded).map(x=>x.d),use=probes.length?probes:t.descs.map(x=>x.d);
  const gd=g.gallery.length&&use.length?galleryDistance(g.gallery,use):undefined;
  const appearance=gd?similarity(gd.total):undefined,uniform=gd?(similarity(gd.shorts)+similarity(gd.socks))/2:undefined,jersey=gd?similarity(gd.jersey):.5;
  const same=g.segment!==undefined&&g.segment===ctx.segment&&t.bornSegment===ctx.segment&&!!g.stab;
  // Detected at the same time as the identity was still detected on another track: two different people.
  const together=same&&t.born<g.lastSeen-1e-3;
  let spatial:number|undefined,movement:number|undefined;
  if(t.firstPitch&&g.lastPitch){
   const v=g.pitchVelocity??{x:0,y:0},m=Math.hypot((t.firstPitch.x-g.lastPitch.x-v.x*k)*PITCH_LENGTH,(t.firstPitch.y-g.lastPitch.y-v.y*k)*PITCH_WIDTH),sigma=3+7*gap;
   spatial=Math.exp(-.5*(m/sigma)**2);
  }
  if(same&&g.stab){
   const extra=Math.max(0,ctx.drift-g.drift),h=Math.max(.005,(g.stab.h+t.firstStab.h)/2),metres=(dx:number,dy:number)=>Math.hypot(dx*ctx.aspect,dy)/h*PLAYER_HEIGHT;
   if(spatial===undefined&&extra<=6){const m=metres(t.firstStab.x-g.stab.x-g.velocity.x*k,t.firstStab.y-g.stab.y-g.velocity.y*k),sigma=3+7*gap+5*extra;spatial=Math.exp(-.5*(m/sigma)**2);}
   // Exit edge vs entry edge, plus whether the new track appeared where the old one was heading.
   const exit=g.exitEdge,entry=t.entryEdge;let e=exit?(entry===exit?1:entry?.25:.35):(entry?.4:.8);
   const speed=metres(g.velocity.x,g.velocity.y),dx=t.firstStab.x-g.stab.x,dy=t.firstStab.y-g.stab.y,dist=metres(dx,dy);
   if(speed>1&&dist>1){const cos=(dx*g.velocity.x*ctx.aspect**2+dy*g.velocity.y)/(Math.hypot(dx*ctx.aspect,dy)*Math.hypot(g.velocity.x*ctx.aspect,g.velocity.y));e=.7*e+.3*(.5+.5*cos);}
   movement=clamp(e);
  }
  if(together)spatial=0;
  const temporal=clamp(Math.exp(-gap/30),.2,1);
  return {appearance,jersey,uniform,team,spatial,movement,temporal,final:together?0:combineReid({appearance,uniform,spatial,movement,temporal,team}),missing:gap};
 }
 private reconnect(t:TrackState,all:Summary[],ctx:StepContext,out:StepOutput){
  const best=all[0],second=all[1],g=this.registry.get(best.playerId);if(!g)return;
  const missing=best.missing;
  if(g.track!==undefined&&g.track!==t.id){const ghost=g.track;this.tracks.delete(ghost);out.drop.push(ghost);this.dropPairs(ghost,out);}
  const crossed=this.bind(t,g,ctx,best.final);
  out.events.push({time:ctx.time,kind:'reid',track:t.id,playerId:g.playerId,message:`Track ${t.id} re-identified as ${globalLabel(g.playerId,true)}: ${pct(best.final)}${second?` vs ${pct(second.final)} for ${globalLabel(second.playerId)}`:' (only missing candidate)'}, missing ${missing.toFixed(1)} s${best.spatial===undefined?', no spatial evidence':''}${crossed?'; follows a crossing, check':''}.`,scores:scoresOf(best,second)});
  if(crossed)out.issues.push({playerId:g.playerId,time:ctx.time,reason:'Re-identified shortly after a crossing: check identity.'});
  for(const s of all.slice(1,4))out.events.push({time:ctx.time,kind:'reid-rejected',track:t.id,playerId:s.playerId,message:`Track ${t.id} is not ${globalLabel(s.playerId,true)}: ${pct(s.final)}, ${pct(best.final-s.final)} behind ${globalLabel(best.playerId)}.`,scores:scoresOf(s)});
  this.flush(t,g,ctx.time,out);
 }
 private defer(t:TrackState,all:Summary[],ctx:StepContext,out:StepOutput,reason:string){
  const best=all[0],second=all[1];
  t.reason=`deferred: ${globalLabel(best.playerId)} ${pct(best.final)}${second?` vs ${globalLabel(second.playerId)} ${pct(second.final)}`:''} (${reason})`;
  if(!this.deferAllowed(t,ctx.time))return;
  out.events.push({time:ctx.time,kind:'deferred',track:t.id,playerId:best.playerId,message:`Track ${t.id}: identity deferred, best ${globalLabel(best.playerId,true)} ${pct(best.final)}${second?`, second ${globalLabel(second.playerId,true)} ${pct(second.final)}`:''} (${reason}).`.slice(0,240),scores:scoresOf(best,second)});
 }
 private createOrHold(t:TrackState,ctx:StepContext,slots:Map<string,RosterSlot>,out:StepOutput){
  if(t.role==='goalkeeper'&&!t.team){
   t.reason='goalkeeper team unknown: waiting for a goalkeeper label';
   if(this.deferAllowed(t,ctx.time))out.events.push({time:ctx.time,kind:'deferred',track:t.id,message:`Track ${t.id}: goalkeeper kit without a known team; label this goalkeeper once to identify it.`});
   return;
  }
  // Someone of this group just lost detection (tie or occlusion): wait until it is a re-ID candidate.
  const lost=this.briefLost(t);
  if(lost){t.reason=`waiting: ${globalLabel(lost.playerId)} lost detection ${this.lostFor(lost).toFixed(1)} s ago`;return;}
  const slot=this.freeSlot(t,slots);
  if(slot){this.createIdentity(t,slot,ctx,out,'no missing identity of this team and role');return;}
  const ref=t.role==='referee',n=ref?this.count(undefined,true):this.count(t.team,false);
  const full=ref?n>=MAX_REFEREES:n>=this.options.maxPerTeam;
  t.reason=full?`${ref?'referees':'team '+t.team} already ${n} identities: kept unidentified`:`no free ${ref?'official':'team '+t.team} roster slot`;
  if(this.allow('cap-'+(ref?'ref':t.team),ctx.time,SANITY_EVERY))out.events.push({time:ctx.time,kind:'sanity',track:t.id,message:full?`${ref?'Officials':'Team '+t.team} already ${ref?'have':'has'} ${n} identities; track ${t.id} (${describeRole(t.role,t.team)}) stays unidentified. Likely a missed re-identification or an unrecorded substitution.`:`No free active ${ref?'official (R1-R3)':'team '+t.team} roster slot for track ${t.id} (${describeRole(t.role,t.team)}); it stays unidentified. Add the player or record the substitution.`});
 }
 private count(team:Team|undefined,referees:boolean){
  let n=0;const active=new Set(this.roster.filter(s=>s.active).map(s=>s.id));
  for(const g of this.registry.values())if(g.status!=='substituted'&&active.has(g.playerId)&&(referees?g.role==='referee':g.role!=='referee'&&g.team===team))n++;
  return n;
 }
 // Lowest free active roster slot of the participant's group. Goalkeepers prefer a keeper slot / jersey 1;
 // outfield players leave those for the keeper while other slots are free.
 private freeSlot(t:TrackState,slots:Map<string,RosterSlot>):RosterSlot|undefined{
  const ref=t.role==='referee';
  if(ref?this.count(undefined,true)>=MAX_REFEREES:!t.team||this.count(t.team,false)>=this.options.maxPerTeam)return undefined;
  const free=[...slots.values()].filter(s=>s.active&&!this.registry.has(s.id)&&(ref?s.team==='ref':s.team===t.team&&s.role!=='referee'));
  if(ref)return free[0];
  if(t.role==='goalkeeper')return free.find(s=>s.role==='goalkeeper')??free.find(keeperish)??free[0];
  return free.find(s=>!keeperish(s))??free.find(s=>s.role!=='goalkeeper');
 }
 private createIdentity(t:TrackState,slot:RosterSlot,ctx:StepContext,out:StepOutput,why:string){
  const g=this.blank(slot.id,slot.team==='ref'?'referee':slot.role==='goalkeeper'?'goalkeeper':t.role==='goalkeeper'?'goalkeeper':'player',slot.team==='ref'?undefined:slot.team,t.box,ctx.time);
  this.registry.set(g.playerId,g);
  for(const x of t.descs)if(!x.occluded)this.learn(g,x.d,x.time,false);
  const conf=t.role==='referee'?t.roleConfidence:t.teamConfidence;
  const crossed=this.bind(t,g,ctx,clamp(.55+.4*conf,.5,.95));
  out.events.push({time:ctx.time,kind:'new-identity',track:t.id,playerId:g.playerId,message:`Track ${t.id} becomes new identity ${globalLabel(g.playerId,true)} (${describeRole(g.role,g.team)}; ${why}${crossed?'; follows a crossing, check':''}).`});
  if(crossed)out.issues.push({playerId:g.playerId,time:ctx.time,reason:'New identity shortly after a crossing: possible duplicate, check.'});
  this.flush(t,g,ctx.time,out);
 }
 // Returns true when the binding follows a recent crossing (confidence capped, caller flags it).
 private bind(t:TrackState,g:GlobalPlayer,ctx:StepContext,conf:number){
  t.playerId=g.playerId;t.state='confirmed';t.reid.clear();t.reason='';
  if(t.role==='goalkeeper'&&g.role!=='goalkeeper'&&g.role!=='referee'&&!this.roster.find(s=>s.id===g.playerId)?.role)g.role='goalkeeper';
  if(!t.team&&g.team)t.team=g.team;
  if(t.role==='referee'&&t.runner!==undefined)g.runner=t.runner;
  g.track=t.id;g.status='active';g.here=true;g.boundAt=ctx.time;g.identityConfidence=r3(clamp(conf));g.teamConfidence=r3(t.teamConfidence);g.roleConfidence=r3(t.roleConfidence);g.capUntil=0;
  const crossed=ctx.time-g.crossedAt<10||ctx.time-t.crossedAt<10;
  if(crossed)g.capUntil=ctx.time+SWAP_CAP;
  return crossed;
 }
 // Buffered observations of a track that was not yet identified are emitted retroactively as 'reidentified'
 // (only for times when the identity had no observation of its own).
 private flush(t:TrackState,g:GlobalPlayer,time:number,out:StepOutput){
  const conf=this.conf(g,time);
  for(const o of t.buffer)if(o.time>g.lastObserved+1e-6&&o.time<time-1e-6)out.observations.push({...o,playerId:g.playerId,evidence:'reidentified',conf});
  t.buffer=[];
 }
 private conf(g:GlobalPlayer,time:number){return r3(time<g.capUntil?Math.min(.5,g.identityConfidence):g.identityConfidence);}
 // Occlusion / swap guard for confirmed tracks: snapshot motion when two overlap, compare KEEP vs SWAP after
 // they separate. Appearance must favour a swap; trajectories alone (proximity) never swap identities.
 private guardSwaps(list:TrackState[],ctx:StepContext,out:StepOutput){
  for(const [k,p] of [...this.pairs]){
   const a=this.tracks.get(p.a),b=this.tracks.get(p.b);
   // A pair survives while both tracks live (a hidden player's track coasts without samples).
   const valid=!!a&&!!b&&a.state==='confirmed'&&a.playerId===p.ids[0]&&(p.ids[1]?b.state==='confirmed'&&b.playerId===p.ids[1]:b.state!=='confirmed')&&ctx.time-p.start<=6;
   if(!valid||!a||!b){this.pairs.delete(k);if(a&&b&&ctx.time-p.start>6)this.dissolve(p,out);for(const t of [a,b])if(t&&!this.pending(t.id))this.releaseHold(t,out);continue;}
   if(a.seenAt!==ctx.time||b.seenAt!==ctx.time||a.kind!=='detection'||b.kind!=='detection'){p.hard=true;p.sep=0;continue;}
   const overlap=iou(a.box,b.box);if(overlap>.4)p.hard=true;
   p.sep=overlap<.05&&!centreClose(a.box,b.box,ctx.aspect)?p.sep+1:0;
   // Judge once separated with clean crops of both (or after a few more steps without them: appearance unknown).
   const clean=(t:TrackState)=>!!t.current&&t.current.quality>=CLEAN&&!t.occluded;
   if(p.sep>=2&&((clean(a)&&clean(b))||p.sep>=6)){this.pairs.delete(k);this.judgeCrossing(a,b,p,ctx,out,clean(a)&&clean(b));for(const t of [a,b])if(!this.pending(t.id))this.releaseHold(t,out);}
  }
  for(let i=0;i<list.length;i++)for(let j=i+1;j<list.length;j++){
   let a=list[i],b=list[j];
   if(a.state!=='confirmed'){if(b.state!=='confirmed')continue;[a,b]=[b,a];}
   const key=Math.min(a.id,b.id)+'-'+Math.max(a.id,b.id);
   if(this.pairs.has(key)||!a.playerId)continue;
   const overlap=iou(a.box,b.box);if(overlap<=.15&&!centreClose(a.box,b.box,ctx.aspect))continue;
   const snap=(t:TrackState)=>{const h=[...t.hist].reverse().find(x=>x.t<ctx.time-1e-6)??t.hist[t.hist.length-1]??{t:ctx.time,x:t.stab.x,y:t.stab.y};return {x:h.x,y:h.y,t:h.t,h:t.stab.h,vx:t.velocity.x,vy:t.velocity.y};};
   this.pairs.set(key,{a:a.id,b:b.id,ids:[a.playerId,b.state==='confirmed'?b.playerId:undefined],start:ctx.time,sep:0,hard:overlap>.4||a.kind!=='detection'||b.kind!=='detection',snap:new Map([[a.id,snap(a)],[b.id,snap(b)]])});
  }
 }
 // Uncertainty after a crossing: confidence capped now, and any re-identification of these identities / tracks
 // within 10 s is capped and flagged too (an undetected swap would surface there).
 private flagCrossing(ids:GlobalPlayer[],tracks:TrackState[],track:number,message:string,out:StepOutput){
  for(const g of ids){g.capUntil=Math.max(g.capUntil,this.time+SWAP_CAP);g.crossedAt=this.time;}
  for(const t of tracks)t.crossedAt=this.time;
  if(!this.allow('cross-'+ids.map(g=>g.playerId).sort().join('|'),this.time,DEFER_EVERY))return;
  out.events.push({time:this.time,kind:'swap-uncertain',track,playerId:ids[0].playerId,message});
  for(const g of ids)out.issues.push({playerId:g.playerId,time:this.time,reason:'Crossed a similar player: check identities.'});
 }
 // A pair that ends without a clean separation (a track ended or it timed out) after one was hidden is flagged.
 private dissolve(p:Pair,out:StepOutput){
  if(!p.hard)return;
  const ids=defined([this.registry.get(p.ids[0]),p.ids[1]?this.registry.get(p.ids[1]):undefined]),tracks=defined([this.tracks.get(p.a),this.tracks.get(p.b)]);if(!ids.length)return;
  this.flagCrossing(ids,tracks,p.a,`${ids.map(g=>globalLabel(g.playerId)).join(' and ')}${p.ids[1]?'':` and track ${p.b}`} were hidden together and did not separate cleanly. Identities kept; confidence lowered. Check identities.`,out);
 }
 private dropPairs(id:number,out:StepOutput){
  for(const [k,p] of [...this.pairs]){
   if(p.a!==id&&p.b!==id)continue;
   this.pairs.delete(k);this.dissolve(p,out);
   const other=this.tracks.get(p.a===id?p.b:p.a);if(other&&!this.pending(other.id))this.releaseHold(other,out);
  }
 }
 private partners(t:TrackState){const out:TrackState[]=[];for(const p of this.pairs.values()){const o=p.a===t.id?p.b:p.b===t.id?p.a:undefined;const x=o!==undefined?this.tracks.get(o):undefined;if(x)out.push(x);}return out;}
 private pending(id:number){for(const p of this.pairs.values())if(p.a===id||p.b===id)return true;return false;}
 // Observations held while a crossing is being judged are released under the final identity (after a corrected
 // swap they go to the right player); a track that lost its identity in between drops them.
 private releaseHold(t:TrackState,out:StepOutput){
  const g=t.state==='confirmed'&&t.playerId?this.registry.get(t.playerId):undefined,slot=g&&this.roster.find(s=>s.id===g.playerId);
  // While the two boxes overlapped heavily (or this one only coasted) either label is positionally right: keep it.
  if(g&&slot?.active)for(const {o,heavy} of t.hold)out.observations.push(heavy?{...o,conf:Math.min(o.conf,this.conf(g,this.time))}:{...o,playerId:g.playerId,conf:this.conf(g,this.time)});
  t.hold=[];
 }
 private judgeCrossing(a:TrackState,b:TrackState,p:Pair,ctx:StepContext,out:StepOutput,clean:boolean){
  const ga=this.registry.get(p.ids[0]),gb=p.ids[1]?this.registry.get(p.ids[1]):undefined;if(!ga||(p.ids[1]&&!gb))return;
  const sa=p.snap.get(a.id)!,sb=p.snap.get(b.id)!,h=Math.max(.005,(a.stab.h+b.stab.h)/2);
  const at=(s:typeof sa)=>{const k=Math.min(2,ctx.time-s.t);return {x:s.x+s.vx*k,y:s.y+s.vy*k};};
  const m=(q:Stab,e:Pt)=>Math.hypot((q.x-e.x)*ctx.aspect,q.y-e.y)/h*PLAYER_HEIGHT,qa=at(sa),qb=at(sb);
  const trajectory=(m(a.stab,qb)+m(b.stab,qa))-(m(a.stab,qa)+m(b.stab,qb)); // metres; > 0 favours KEEP
  const d=(g:GlobalPlayer,t:TrackState)=>clean&&t.current&&g.gallery.length?galleryDistance(g.gallery,[t.current])?.total:undefined;
  const aa=d(ga,a),ab=d(ga,b),bb=gb?d(gb,b):0,ba=gb?d(gb,a):0;
  // > 0 favours KEEP. With an unidentified partner only ga's gallery can be compared (one-sided).
  const look=aa!==undefined&&bb!==undefined&&ab!==undefined&&ba!==undefined?(ab+ba)-(aa+bb):undefined;
  const names=gb?`${globalLabel(ga.playerId)} and ${globalLabel(gb.playerId)}`:`${globalLabel(ga.playerId)} and track ${b.id}`;
  // Swap only on appearance: clearly different kits always; a moderate difference only if the paths do not object.
  if(look!==undefined&&(look<=-.5||(look<=-.2&&trajectory<4))){
   const why=`appearance favours the swap by ${(-look).toFixed(2)}, trajectory ${trajectory.toFixed(1)} m`;
   if(gb){
    a.playerId=gb.playerId;b.playerId=ga.playerId;ga.track=b.id;gb.track=a.id;ga.boundAt=gb.boundAt=ctx.time;
    [a.team,b.team,a.role,b.role]=[gb.team,ga.team,gb.role,ga.role];a.teamVotes=[];b.teamVotes=[];
    for(const [g,t] of [[ga,b],[gb,a]] as [GlobalPlayer,TrackState][])out.events.push({time:ctx.time,kind:'swap-corrected',track:t.id,playerId:g.playerId,message:`Identities of ${names} swapped back after crossing: ${globalLabel(g.playerId)} is track ${t.id} (${why}).`});
    return;
   }
   // The identity moved to the unidentified track: the two people's evidence follows them; the ambiguous
   // stretch is dropped (only observations from while they overlapped, which fit either label, are kept).
   for(const {o,heavy} of a.hold)if(heavy)out.observations.push(o);
   a.hold=[];a.buffer=[];b.buffer=[];b.hold=[];
   this.swapPersons(a,b);ga.track=b.id;ga.boundAt=ctx.time;a.reason=`identity moved to track ${b.id} after crossing`;
   out.events.push({time:ctx.time,kind:'swap-corrected',track:b.id,playerId:ga.playerId,message:`${globalLabel(ga.playerId)} moved from track ${a.id} to track ${b.id} after crossing (${why}).`});
   return;
  }
  // Same kit: crossing and turning back look alike once one player was hidden, so continuity is not proof.
  const similar=look===undefined||Math.abs(look)<.2;
  if((similar&&(p.hard||trajectory<2))||(look!==undefined&&look<=-.2)){
   const why=similar?`kits too similar${p.hard?', one was hidden':''}, paths ${trajectory>=0?'favour keeping':'favour swapping'} by ${Math.abs(trajectory).toFixed(1)} m`:`appearance suggests a swap but the paths disagree (${trajectory.toFixed(1)} m)`;
   this.flagCrossing(gb?[ga,gb]:[ga],[a,b],a.id,`${names} crossed (tracks ${a.id}/${b.id}); ${why}. Identities kept; confidence lowered. Check identities.`,out);
  }
 }
 // Everything learnt about the person on a track (evidence, role, identity state) moves with the person.
 private swapPersons(a:TrackState,b:TrackState){
  const keys=['born','bornSegment','firstStab','firstPitch','firstZone','entryEdge','hits','inside','boundary','outside','outsideRun','evidence','decision','teamVotes','descs','state','playerId','team','role','teamConfidence','roleConfidence','retired','runner','buffer','reid','best','lastDeferred','deferrals','anchoredAt','crossedAt','reason'] as const;
  const x=a as unknown as Record<string,unknown>,y=b as unknown as Record<string,unknown>;
  for(const k of keys){const v=x[k];x[k]=y[k];y[k]=v;if(x[k]===undefined)delete x[k];if(y[k]===undefined)delete y[k];}
 }
 // A confirmed track picked up again after being hidden (>= 2 steps) right next to a similar person may now follow
 // that person: same-kit evidence cannot tell, so confidence is lowered and the identity flagged (never swapped).
 private checkReacquired(live:TrackState[],ctx:StepContext,out:StepOutput){
  for(const t of live){
   if(!t.reacquired||t.state!=='confirmed'||!t.playerId||this.pending(t.id))continue;
   const g=this.registry.get(t.playerId);if(!g)continue;
   const w=t.box.w*ctx.aspect,cx=t.box.x+t.box.w/2,cy=t.box.y+t.box.h/2;
   const similar=live.find(o=>o!==t&&o.state!=='rejected'&&!o.retired&&(o.state==='confirmed'?o.team===t.team&&o.role===t.role:!o.teamVotes.length||o.teamVotes[o.teamVotes.length-1]===t.team||t.role!=='player')&&Math.hypot((o.box.x+o.box.w/2-cx)*ctx.aspect,o.box.y+o.box.h/2-cy)<3*w);
   if(!similar)continue;
   g.capUntil=Math.max(g.capUntil,ctx.time+SWAP_CAP);
   if(!this.allow('reacq-'+g.playerId,ctx.time,DEFER_EVERY))continue;
   out.events.push({time:ctx.time,kind:'swap-uncertain',track:t.id,playerId:g.playerId,message:`${globalLabel(g.playerId)} (track ${t.id}) re-acquired after being hidden next to a similar player (track ${similar.id}). Identity kept; confidence lowered. Check identity.`});
   out.issues.push({playerId:g.playerId,time:ctx.time,reason:'Re-acquired after being hidden near a similar player: check identity.'});
  }
 }
 // Identity runtime state, gallery updates and observations for confirmed tracks; buffers for the rest.
 private refresh(live:TrackState[],ctx:StepContext,slots:Map<string,RosterSlot>,out:StepOutput){
  const visible:GlobalPlayer[]=[];
  for(const t of live){
   const g=t.state==='confirmed'&&t.playerId?this.registry.get(t.playerId):undefined;
   if(t.state==='confirmed'&&(!g||g.track!==t.id)){t.playerId=undefined;t.state='uncertain';}
   if(!g||g.track!==t.id){
    if(t.retired)continue;
    t.buffer.push({playerId:'',time:ctx.time,box:t.box,score:t.score,evidence:t.kind,track:t.id,conf:0,...(t.pitch?{pitch:t.pitch}:{})});
    if(t.buffer.length>BUFFER)t.buffer.shift();continue;
   }
   // A crop whose jersey clearly contradicts the identity (another kit) is not this player: withhold it.
   // Repeated contradictions mean the local track moved to someone else: the identity is released.
   const dj=t.kind==='detection'&&t.current&&t.current.quality>=.3&&g.gallery.length&&t.current.jersey.some(v=>v>0)?galleryDistance(g.gallery,[t.current])?.jersey:undefined;
   if(dj!==undefined&&dj>KIT_MISMATCH){
    t.kitMiss++;
    if(t.kitMiss>=KIT_MISSES){
     this.unbind(g,false);this.release(t,`kit no longer matches ${globalLabel(g.playerId)}`);t.kitMiss=0;
     out.events.push({time:ctx.time,kind:'reid-rejected',track:t.id,playerId:g.playerId,message:`Track ${t.id} no longer matches ${globalLabel(g.playerId,true)}'s kit (jersey distance ${dj.toFixed(2)}); identity released.`});
     out.issues.push({playerId:g.playerId,time:ctx.time,reason:'Tracked person stopped matching this kit; identity released. Check this player.'});
    }else t.reason=`kit mismatch (${dj.toFixed(2)}): observation withheld`;
    continue;
   }
   t.kitMiss=0;
   if(t.kind==='detection'){
    g.lastSeen=ctx.time;g.lastBox=t.box;g.stab=t.stab;g.segment=ctx.segment;g.drift=ctx.drift;g.velocity={...t.velocity};g.exitEdge='';g.status='active';
    if(t.pitch){g.lastPitch=t.pitch;if(t.pitchVelocity)g.pitchVelocity=t.pitchVelocity;else delete g.pitchVelocity;}else{delete g.lastPitch;delete g.pitchVelocity;}
    if(t.current&&!t.occluded&&t.zone!=='outside'&&this.conf(g,ctx.time)>=.6)g.gallery=addToGallery(g.gallery,t.current,ctx.time,this.options.gallerySize);
    visible.push(g);
   }
   if(slots.get(g.playerId)?.active&&!(t.zone==='outside'&&t.outsideRun>10)){
    const o:SoccerObservation={playerId:g.playerId,time:ctx.time,box:t.box,score:t.score,evidence:t.kind,track:t.id,conf:this.conf(g,ctx.time),...(t.pitch?{pitch:t.pitch}:{})};
    if(this.pending(t.id)){t.hold.push({o,heavy:t.kind!=='detection'||this.partners(t).some(x=>iou(x.box,t.box)>.3)});if(t.hold.length>BUFFER)out.observations.push(t.hold.shift()!.o);}else out.observations.push(o);
    g.lastObserved=ctx.time;
   }
  }
  for(let i=0;i<visible.length;i++)for(let j=i+1;j<visible.length;j++){const a=visible[i],b=visible[j];if(a.team&&a.team===b.team)this.together.add([a.playerId,b.playerId].sort().join('|'));}
 }
 debugTrack(t:TrackState):DebugTrack{
  const g=t.state==='confirmed'&&t.playerId?this.registry.get(t.playerId):undefined;
  let label='CAND',roleLbl:DebugTrack['roleLabel']='CANDIDATE',team=t.team,identity=0;
  if(g){label=globalLabel(g.playerId);roleLbl=roleLabel(g.role,g.team);team=g.team;identity=this.conf(g,this.time);}
  else if(t.state==='uncertain'){label=t.role==='referee'?'REF-?':t.team?`${t.team}-?`:'GK-?';roleLbl='IDENTITY_UNCERTAIN';identity=t.best;}
  else if(t.state==='unknown'||t.state==='rejected'){label='UNK';roleLbl='UNKNOWN';}
  return {track:t.id,box:t.box,...(g?{playerId:g.playerId}:{}),label,...(team?{team}:{}),roleLabel:roleLbl,state:t.state,identity:r3(identity),zone:t.zone,...(t.reason?{reason:t.reason}:{})};
 }
 trackAt(box:Box):DebugTrack|undefined{
  let best:TrackState|undefined,score=.5;
  for(const t of this.tracks.values()){const v=iou(t.box,box);if(v>score){score=v;best=t;}}
  return best?this.debugTrack(best):undefined;
 }
 // Sanity: too many identities per team / officials, and same-team identities that were never visible together
 // yet look alike (possible duplicate identity).
 sanity(time:number):ReidEvent[]{
  const out:ReidEvent[]=[],o=this.options;
  for(const team of ['A','B'] as Team[]){const n=this.count(team,false);if(n>o.maxPerTeam)out.push({time,kind:'sanity',message:`Team ${team} has ${n} active identities (expected at most ${o.maxPerTeam}). Check for duplicates or unrecorded substitutions.`});}
  const refs=this.count(undefined,true);if(refs>MAX_REFEREES)out.push({time,kind:'sanity',message:`${refs} referee identities (expected at most ${MAX_REFEREES}).`});
  const alike:{a:string;b:string;d:number}[]=[];
  for(const team of ['A','B'] as Team[]){
   // Only identities tracked in this session (co-visibility is not saved); two user-labelled ones are distinct.
   const ids=[...this.registry.values()].filter(g=>g.here&&g.team===team&&g.role!=='referee'&&g.status!=='substituted'&&g.gallery.length>=2);
   for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){
    const a=ids[i],b=ids[j];if((a.anchored&&b.anchored)||this.together.has([a.playerId,b.playerId].sort().join('|')))continue;
    const d=galleryDistance(a.gallery,b.gallery.map(s=>s.d))?.total;if(d!==undefined&&d<.08)alike.push({a:a.playerId,b:b.playerId,d});
   }
  }
  for(const x of alike.sort((p,q)=>p.d-q.d).slice(0,3))out.push({time,kind:'sanity',playerId:x.a,message:`${globalLabel(x.a,true)} and ${globalLabel(x.b,true)} were never visible together and look alike (distance ${x.d.toFixed(2)}): possible duplicate identity.`});
  return out;
 }
 // 'deferred' events: at most every 2 s per track, backing off to every 30 s while the track stays undecided.
 private deferAllowed(t:TrackState,time:number){
  if(time-t.lastDeferred<Math.min(30,DEFER_EVERY*2**Math.min(4,Math.max(0,t.deferrals-1)))&&time>=t.lastDeferred)return false;
  t.lastDeferred=time;t.deferrals++;return true;
 }
 private allow(key:string,time:number,every:number){
  const last=this.limits.get(key);if(last!==undefined&&time-last<every&&time>=last)return false;
  this.limits.set(key,time);if(this.limits.size>400)this.limits.delete(this.limits.keys().next().value as string);return true;
 }
}
