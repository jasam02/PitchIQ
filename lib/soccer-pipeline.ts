import {assignMinimum,startTrack,stepTracks,type PersistentTrack,type TrackDetection} from './persistent-tracker';
import {appearanceDistance,type CameraMotion} from './track-vision';
import {cosine,quantize} from './player-reid';
import {iou} from './detection-core';
import {emptySoccerState,type GlobalIdentity,type ReIDEvent,type SoccerObservation,type SoccerRole,type SoccerState} from './soccer-schema';
import {isActive,type Box,type TrackingDoc} from './tracking';

const foot=(b:Box)=>({x:b.x+b.w/2,y:b.y+b.h});
const clamp=(n:number)=>Math.max(0,Math.min(1,n));
const teamOf=(role:SoccerRole)=>role.endsWith('_A')?'A':role.endsWith('_B')?'B':'UNKNOWN';
const playerRole=(team:string):SoccerRole=>team==='A'?'PLAYER_TEAM_A':team==='B'?'PLAYER_TEAM_B':'UNKNOWN';
type LocalEvidence={born:number;last:number;hits:number;fieldEvidence:number;motion:number;votes:number[];roleVotes:Record<string,number>;team:GlobalIdentity['team'];role:SoccerRole;teamConfidence:number;roleConfidence:number;candidate?:string;candidateHits:number;candidateTime:number;globalId?:string;detection?:TrackDetection};

export class TeamClassifier {
 constructor(public centers:number[][]){}
 seed(team:string,feature:number[]){if((team==='A'||team==='B')&&feature.length)this.centers[team==='A'?0:1]=[...feature];}
 discover(detections:TrackDetection[]){
  if(this.centers.length===2&&this.centers.every(c=>c?.length))return;
  const features=detections.filter(d=>d.kind==='person'&&(d.fieldScore||0)>.65&&d.score>.5&&d.appearance?.length).map(d=>d.appearance!);
  if(features.length<6)return;
  // Two sizeable uniform modes; isolated referee/keeper/staff colors cannot
  // become a team center. Naming is arbitrary until a manual team seed exists.
  const groups=features.map(f=>({f,members:features.filter(g=>appearanceDistance(f,g)<.16)})).sort((a,b)=>b.members.length-a.members.length);
  const a=groups[0],b=groups.find(g=>g.members.length>=3&&appearanceDistance(a.f,g.f)>.35);
  if(a.members.length<3||!b)return;
  if(this.centers[0]?.length){const next=[a,b].find(g=>appearanceDistance(this.centers[0],g.f)>.35);if(next)this.centers[1]=next.f;}
  else if(this.centers[1]?.length){const next=[a,b].find(g=>appearanceDistance(this.centers[1],g.f)>.35);if(next)this.centers[0]=next.f;}
  else this.centers=[a.f,b.f];
 }
 classify(feature:number[]=[]):{team:GlobalIdentity['team'];confidence:number}{
  if(!feature.length||!this.centers[0]?.length||!this.centers[1]?.length)return {team:'UNKNOWN',confidence:0};
  const a=appearanceDistance(feature,this.centers[0]),b=appearanceDistance(feature,this.centers[1]),best=Math.min(a,b),margin=Math.abs(a-b);
  if(best>.25||margin<.1)return {team:'UNKNOWN',confidence:0};
  return {team:a<b?'A':'B',confidence:clamp((1-best)*Math.min(1,margin/.25))};
 }
}

export class RoleClassifier {
 // Referee and goalkeeper uniforms need a confirmed role example; an outlier
 // is UNKNOWN, never presumed a referee or assigned to the nearest team.
 classify(d:TrackDetection,team:GlobalIdentity['team'],identities:GlobalIdentity[]){
  const special=identities.filter(g=>g.manualRole&&g.role!=='UNKNOWN'&&!g.role.startsWith('PLAYER')&&g.kit.length&&d.kit?.length===g.kit.length).map(g=>{
   const distances=[0,50,100].map(start=>appearanceDistance(g.kit.slice(start,start+50),d.kit!.slice(start,start+50)));
   return {role:g.role,cost:distances[0]*.55+distances[1]*.3+distances[2]*.15};
  }).sort((a,b)=>a.cost-b.cost);
  const best=special[0];
  if(best&&best.cost<.15&&(!special[1]||special[1].role===best.role||special[1].cost-best.cost>.08))return {role:best.role,confidence:1-best.cost};
  return {role:playerRole(team),confidence:team==='UNKNOWN'?0:.8};
 }
}

export class GlobalIdentityManager {
 constructor(public state:SoccerState){}
 log(event:ReIDEvent,debug:boolean){
  if(event.accepted||debug){this.state.events.push(event);this.state.events=[...this.state.events.filter(e=>e.accepted).slice(-80),...this.state.events.filter(e=>!e.accepted).slice(-20)].sort((a,b)=>a.time-b.time);}
 }
 gallery(g:GlobalIdentity,d:TrackDetection,time:number){
  if(!d.embedding||(d.quality||0)<.55||g.identityConfidence<.88)return;
  if(g.gallery.some(e=>Math.abs(time-e.time)<1))return;
  const embedding=quantize(d.embedding);
  if(g.gallery.length&&Math.max(...g.gallery.map(e=>cosine(e.embedding,embedding)))>.985)return;
  g.gallery.push({embedding,quality:d.quality!,time});
  // Keep the initial trusted anchor; retain four recent, distinct, good crops.
  if(g.gallery.length>5)g.gallery.splice(1,1);
 }
 candidate(g:GlobalIdentity,d:TrackDetection,team:GlobalIdentity['team'],role:SoccerRole,time:number,pitch?:{x:number;y:number}){
  const gap=Math.max(0,time-g.lastSeen),appearance=Math.max(0,...g.gallery.map(e=>cosine(e.embedding,d.embedding||[])));
  let reason='',spatial=.5,trajectory=.5;
  if(g.state==='SUBSTITUTED')reason='Substituted identity';
  else if(g.team!=='UNKNOWN'&&team!=='UNKNOWN'&&g.team!==team)reason='Conflicting team';
  else if(role==='REFEREE'&&g.role!=='REFEREE'||g.role==='REFEREE'&&role!=='REFEREE')reason='Conflicting role';
  else if(!d.embedding||!g.gallery.length||appearance<(gap>120?.88:.82))reason='Insufficient learned appearance evidence';
  if(pitch&&g.pitch&&g.pitchTime!==undefined){
   const seconds=Math.max(.2,time-g.pitchTime),meters=Math.hypot((pitch.x-g.pitch.x)*105,(pitch.y-g.pitch.y)*68);
   if(meters>13*seconds+4)reason='Physically implausible pitch displacement';
   spatial=clamp(1-meters/(13*seconds+8));
  }
  if(g.cameraReliable&&g.cameraBox&&gap<=2){
   const expected=foot(g.cameraBox),actual=foot(d.box),distance=Math.hypot(actual.x-expected.x-g.velocity.x*gap,actual.y-expected.y-g.velocity.y*gap);
   trajectory=clamp(1-distance/(.04+gap*.1));
   if(gap<1&&distance>.1+gap*.15)reason='Implausible compensated image trajectory';
  }
  const jersey=1-appearanceDistance(g.jersey,d.appearance||[]),confidence=clamp(.76*appearance+.1*jersey+.06*spatial+.03*trajectory+.05*(team===g.team?1:.5));
  return {appearance,spatial,trajectory,teamMatch:team!=='UNKNOWN'&&team===g.team,gap,confidence,reason};
 }
}

export class SoccerPipeline {
 readonly manager:GlobalIdentityManager;
 readonly teams:TeamClassifier;
 readonly roles=new RoleClassifier();
 tracks:PersistentTrack[]=[];
 private locals=new Map<string,LocalEvidence>();
 constructor(private doc:TrackingDoc,start:number,private debug=false){
  const state:SoccerState=doc.soccer?JSON.parse(JSON.stringify(doc.soccer)):emptySoccerState();
  // A seek backwards must not reuse future appearance evidence.
  state.events=state.events.filter(e=>e.time<=start);
  state.identities=state.identities.filter(g=>g.history.some(h=>h.time<=start));
  for(const g of state.identities){g.cameraReliable=false;g.gallery=g.gallery.filter(e=>e.time<=start);g.history=g.history.filter(h=>h.time<=start);g.state=g.state==='SUBSTITUTED'?'SUBSTITUTED':'MISSING';delete g.localId;
   const last=g.history.at(-1);if(g.lastSeen>start&&last){g.lastSeen=last.time;g.box={...g.box,x:last.image.x-g.box.w/2,y:last.image.y-g.box.h};g.pitch=last.pitch;g.pitchTime=last.pitch?last.time:undefined;}}
  this.manager=new GlobalIdentityManager(state);this.teams=new TeamClassifier(start<(doc.checkpoint?.time||0)?[]:state.teams);
 }
 get state(){this.manager.state.teams=this.teams.centers.length?[this.teams.centers[0]||[],this.teams.centers[1]||[]]:[];return this.manager.state;}
 private local(time:number,box:Box,d?:TrackDetection){
  const id='track-'+this.manager.state.nextLocal++;
  this.tracks.push({...startTrack(id,'person',box,time,d?.appearance||[]),embedding:d?.embedding});
  this.locals.set(id,{born:time,last:time,hits:0,fieldEvidence:0,motion:0,votes:[0,0],roleVotes:{},team:'UNKNOWN',role:'UNKNOWN',teamConfidence:0,roleConfidence:0,candidateHits:0,candidateTime:-1,detection:d});return id;
 }
 seed(id:string,box:Box,time:number,d:TrackDetection,role?:SoccerRole){
  const player=this.doc.players.find(p=>p.id===id),team=player?.team==='A'?'A':player?.team==='B'?'B':'UNKNOWN';
  const resolvedRole=role||player?.role||playerRole(team);
  let g=this.state.identities.find(g=>g.id===id);
  if(!g){if(this.state.identities.length>=96)throw Error('Identity registry capacity reached. Review duplicate assignments.');g={id,team,role:resolvedRole,state:'ACTIVE',manualRole:!!role||!!player?.role,identityConfidence:1,teamConfidence:team==='UNKNOWN'?0:1,roleConfidence:1,gallery:[],jersey:d.appearance||[],kit:d.kit||[],lastSeen:time,box,velocity:{x:0,y:0},history:[],teamVotes:team==='A'?[10,0]:team==='B'?[0,10]:[0,0],roleVotes:{[resolvedRole]:10}};this.state.identities.push(g);}
  // Explicit corrections replace contaminated appearance history from here on.
  g.gallery=g.gallery.filter(e=>e.time<time);g.role=resolvedRole;g.team=teamOf(resolvedRole);g.manualRole=!!role||!!player?.role;g.identityConfidence=1;g.state='ACTIVE';g.lastSeen=time;g.box=box;g.jersey=d.appearance||[];g.kit=d.kit||[];
  this.tracks=this.tracks.filter(t=>this.locals.get(t.id)?.globalId!==id&&iou(t.box,box)<.6);
  const localId=this.local(time,box,d),e=this.locals.get(localId)!;e.globalId=id;e.team=g.team;e.role=g.role;e.teamConfidence=g.teamConfidence;e.roleConfidence=g.roleConfidence;g.localId=localId;
  if(g.role.startsWith('PLAYER'))this.teams.seed(team,d.appearance||[]);
  this.manager.gallery(g,d,time);g.history.push({time,image:foot(box)});g.history=g.history.slice(-40);
 }
 cut(){this.tracks=[];this.locals.clear();for(const g of this.state.identities){if(g.state==='ACTIVE')g.state='MISSING';delete g.localId;g.cameraReliable=false;g.velocity={x:0,y:0};}}
 step(detections:TrackDetection[],time:number,camera:CameraMotion,mapPitch:(b:Box)=>{x:number;y:number}|undefined){
  if(camera.cut)this.cut();
  for(const g of this.state.identities){if(!camera.reliable)g.cameraReliable=false;else if(g.cameraBox){const b=g.cameraBox;g.cameraBox={x:b.x*camera.scale+camera.dx,y:b.y*camera.scale+camera.dy,w:b.w*camera.scale,h:b.h*camera.scale};}}
  this.teams.discover(detections);
  // Keep local observations when the field is uncertain. Only permanent
  // roster admission requires sustained reliable field evidence below.
  const people=detections.filter(d=>d.kind==='person'&&(!d.fieldReliable||(d.fieldScore||0)>=.2));
  const result=stepTracks(this.tracks,people,time,camera.cut?{...camera,cut:false}:camera);this.tracks=result.tracks;
  const used=new Set<string>();
  for(const sample of result.samples)if(sample.evidence==='detection'){
   const d=people.find(d=>iou(d.box,sample.box)>.95);if(d){used.add(d.id);this.locals.get(sample.id)!.detection=d;}
  }
  for(const d of people)if(d.score>=.25&&!used.has(d.id)&&!this.tracks.some(t=>iou(t.box,d.box)>.15)){
   if(this.tracks.length>=64)break;this.local(time,d.box,d);
  }
  const activeIds=new Set(this.tracks.map(t=>t.id));for(const id of this.locals.keys())if(!activeIds.has(id))this.locals.delete(id);
  for(const g of this.state.identities){if(this.doc.players.some(p=>p.id===g.id)&&!isActive(this.doc,g.id,time)){g.state='SUBSTITUTED';for(const e of this.locals.values())if(e.globalId===g.id)delete e.globalId;delete g.localId;continue;}const bound=this.tracks.find(t=>this.locals.get(t.id)?.globalId===g.id);if(bound){g.localId=bound.id;g.state=bound.lastSeen===time?'ACTIVE':'OFF_FIELD';}else {g.state='MISSING';delete g.localId;}}
  const seen=this.tracks.filter(t=>t.lastSeen===time);
  for(const tr of seen){const e=this.locals.get(tr.id)!,d=e.detection!;const dt=Math.min(.3,Math.max(0,time-e.last));e.fieldEvidence+=dt*(d.fieldScore||0);if(camera.reliable)e.motion+=Math.hypot(tr.vx,tr.vy)*dt;e.hits++;e.last=time;
   const vote=this.teams.classify(d.appearance);if(vote.team!=='UNKNOWN')e.votes[vote.team==='A'?0:1]+=vote.confidence;
   const total=e.votes[0]+e.votes[1],winner=e.votes[0]>=e.votes[1]?0:1;
   if(total>=4&&e.votes[winner]/total>.85){const next=winner===0?'A':'B';if(e.team==='UNKNOWN'||e.team===next||total>20&&e.votes[winner]/total>.97)e.team=next;e.teamConfidence=e.votes[winner]/total;}
   const roleVote=this.roles.classify(d,e.team,this.state.identities);if(roleVote.role!=='UNKNOWN')e.roleVotes[roleVote.role]=(e.roleVotes[roleVote.role]||0)+roleVote.confidence;
   const votes=Object.entries(e.roleVotes).sort((a,b)=>b[1]-a[1]),sum=votes.reduce((s,v)=>s+v[1],0);
   if(votes[0]?.[1]>=4&&votes[0][1]/sum>.85&&(e.role==='UNKNOWN'||votes[0][0]===e.role||sum>30&&votes[0][1]/sum>.97)){e.role=votes[0][0] as SoccerRole;e.roleConfidence=votes[0][1]/sum;}
  }
  // Missing neural evidence between scheduled inference frames is not a
  // rejected match and must not erase the multi-sighting confirmation streak.
  const unresolved=seen.filter(t=>!this.locals.get(t.id)!.globalId&&this.locals.get(t.id)!.detection?.embedding);
  const missing=this.state.identities.filter(g=>g.state==='MISSING');
  const evidence=unresolved.map(tr=>{const e=this.locals.get(tr.id)!;return missing.map(g=>this.manager.candidate(g,e.detection!,e.team,e.role,time,mapPitch(tr.box)));});
  const costs=evidence.map(row=>row.map(v=>v.reason?1000:1-v.confidence));
  const assigned=assignMinimum(costs,.12);
  assigned.forEach((col,row)=>{
   const tr=unresolved[row],e=this.locals.get(tr.id)!;
   if(col<0){e.candidateHits=0;return;}
   const g=missing[col],score=evidence[row][col];
   const alternatives=this.state.identities.filter(other=>other.id!==g.id&&other.state!=='SUBSTITUTED').map(other=>this.manager.candidate(other,e.detection!,e.team,e.role,time,mapPitch(tr.box)));
   const ambiguous=alternatives.some(s=>s.appearance>=score.appearance-.06)||evidence.some((r,i)=>i!==row&&!r[col].reason&&r[col].confidence>=score.confidence-.06);
   if(ambiguous){e.candidateHits=0;this.manager.log({time,localId:tr.id,globalId:g.id,...score,accepted:false,reason:'Ambiguous appearance or competing local track'},this.debug);return;}
   // Count only fresh neural observations, spaced in time, not reused vectors.
   if(e.detection?.embedding&&time-e.candidateTime>=.45){e.candidateHits=e.candidate===g.id?e.candidateHits+1:1;e.candidate=g.id;e.candidateTime=time;}
   if(e.candidateHits>=3&&time-e.born>=1){e.globalId=g.id;g.localId=tr.id;g.identityConfidence=score.confidence;g.state='ACTIVE';this.manager.log({time,localId:tr.id,globalId:g.id,...score,accepted:true,reason:'Learned appearance, team and repeated independent sightings agree'},this.debug);}
  });
  if(this.debug)unresolved.forEach((tr,row)=>missing.forEach((g,col)=>{const v=evidence[row][col];if(v.reason&&this.locals.get(tr.id)?.detection?.embedding)this.manager.log({time,localId:tr.id,globalId:g.id,...v,accepted:false},true);}));
  const observations:SoccerObservation[]=[];
  for(const tr of seen){const e=this.locals.get(tr.id)!,d=e.detection!;
   // New roster admission is deliberately conservative: same-team missing
   // identities must be resolved, not replaced with extra permanent players.
   const nearby=seen.filter(other=>other.id!==tr.id&&Math.hypot(foot(other.box).x-foot(tr.box).x,foot(other.box).y-foot(tr.box).y)<.3).length;
   const sustained=e.fieldEvidence>=2&&(e.motion>.004||e.fieldEvidence>=4)&&nearby>=2;
   if(!e.globalId&&sustained&&e.hits>=8&&e.team!=='UNKNOWN'&&e.role.startsWith('PLAYER')&&(d.fieldScore||0)>.65&&d.embedding&&(d.quality||0)>.55&&!this.state.identities.some(g=>g.team===e.team&&(g.state==='MISSING'||g.state==='OFF_FIELD'))){
    const slot=this.doc.players.find(p=>p.team===e.team&&isActive(this.doc,p.id,time)&&!this.state.identities.some(g=>g.id===p.id));
    if(slot){const g:GlobalIdentity={id:slot.id,localId:tr.id,team:e.team,role:e.role,state:'ACTIVE',manualRole:false,identityConfidence:.9,teamConfidence:e.teamConfidence,roleConfidence:e.roleConfidence,gallery:[],jersey:d.appearance||[],kit:d.kit||[],lastSeen:time,box:tr.box,velocity:{x:tr.vx,y:tr.vy},history:[],teamVotes:e.votes,roleVotes:e.roleVotes};this.state.identities.push(g);e.globalId=g.id;}
    else if(this.debug)this.manager.log({time,localId:tr.id,globalId:'UNKNOWN_PLAYER',accepted:false,reason:'Roster capacity: possible duplicate or substitution; confirm identity',appearance:0,spatial:0,gap:0,confidence:0},true);
   }
   const g=this.state.identities.find(g=>g.id===e.globalId),pitch=mapPitch(tr.box);
   if(g&&g.state!=='SUBSTITUTED'){
    const ownSimilarity=d.embedding&&g.gallery.length?Math.max(...g.gallery.map(item=>cosine(item.embedding,d.embedding!))):undefined;
    const competingSimilarity=d.embedding?Math.max(0,...this.state.identities.filter(other=>other.id!==g.id&&other.state!=='SUBSTITUTED').flatMap(other=>other.gallery.map(item=>cosine(item.embedding,d.embedding!)))):0;
    const appearanceConflict=ownSimilarity!==undefined&&(ownSimilarity<.78||competingSimilarity>=ownSimilarity-.04);
    // A strong contradictory role/team suspends the local attachment; it does
    // not rewrite the player's established identity from an anomalous frame.
    if(appearanceConflict||(e.team!=='UNKNOWN'&&g.team!=='UNKNOWN'&&e.team!==g.team)||(e.role==='REFEREE'&&g.role!=='REFEREE')){g.state='MISSING';delete g.localId;delete e.globalId;this.manager.log({time,localId:tr.id,globalId:g.id,accepted:false,reason:'Continuous track identity uncertain; gallery update withheld',appearance:ownSimilarity||0,spatial:.5,gap:time-g.lastSeen,confidence:0},this.debug);}
    else {g.state='ACTIVE';g.localId=tr.id;g.box=tr.box;g.cameraBox=tr.box;g.cameraReliable=true;g.lastSeen=time;g.velocity={x:tr.vx,y:tr.vy};if(pitch){g.pitch=pitch;g.pitchTime=time;}
     if(!g.kit.length&&d.kit?.length)g.kit=d.kit;
     this.manager.gallery(g,d,time);g.history.push({time,image:foot(tr.box),pitch});g.history=g.history.slice(-40);
    }
   }
   const known=e.globalId&&g?.state==='ACTIVE'?g:undefined;
   observations.push({time,globalId:known?.id,localId:tr.id,box:tr.box,team:known?.team||e.team,role:known?.role||e.role,identityConfidence:known?.identityConfidence||0,teamConfidence:known?.teamConfidence||e.teamConfidence,roleConfidence:known?.roleConfidence||e.roleConfidence,pitch,uncertain:!known||known.identityConfidence<.88});
  }
  return observations;
 }
}
