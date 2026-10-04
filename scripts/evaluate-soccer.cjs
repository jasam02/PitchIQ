// node scripts/evaluate-soccer.cjs pitchiq-tracking.json ground-truth.json
// Ground truth: {frames:[{time,people:[{id,role,box,visible:true}]}]}
// role is SPECTATOR, REFEREE, PLAYER_TEAM_A/B or GOALKEEPER_TEAM_A/B.
const fs=require('node:fs'),load=require('../tests/load-ts.cjs');
const {assignMinimum}=load('lib/persistent-tracker.ts'),{iou}=load('lib/detection-core.ts');
const [predFile,truthFile]=process.argv.slice(2);if(!predFile||!truthFile)throw Error('Provide exported tracking JSON and annotated ground-truth JSON.');
const pred=JSON.parse(fs.readFileSync(predFile,'utf8')),truth=JSON.parse(fs.readFileSync(truthFile,'utf8'));
if(!Array.isArray(truth.frames)||!truth.frames.length)throw Error('Ground truth needs annotated frames; unlabeled detections cannot measure identity accuracy.');
const metrics={falseSpectatorDetections:0,refereeAsPlayerErrors:0,idSwitches:0,fragmentedPlayerTracks:0,duplicateGlobalIdentities:0,successfulReIDEvents:0,incorrectReIDEvents:0,unscoredReIDEvents:0,averagePlayerIdentityConfidence:0,visiblePlayerCoverage:0,unmatchedPredictedPlayers:0,perPlayer:{}};
const previous=new Map(),ids=new Map(),evaluated=new Map();let confidence=0,confidenceN=0,visible=0,tracked=0;
for(const frame of [...truth.frames].sort((a,b)=>a.time-b.time)){
 const people=frame.people.filter(p=>p.visible!==false),near=(pred.observations||[]).filter(p=>Math.abs(p.time-frame.time)<=.105);
 // One nearest observation per local track, then one-to-one spatial matching.
 const candidates=[...new Map(near.sort((a,b)=>Math.abs(b.time-frame.time)-Math.abs(a.time-frame.time)).map(p=>[p.localId,p])).values()];
 const costs=people.map(p=>candidates.map(q=>1-iou(p.box,q.box))),assignment=assignMinimum(costs,.7),used=new Set();
 people.forEach((p,i)=>{
  const c=assignment[i]>=0?candidates[assignment[i]]:undefined;if(c)used.add(assignment[i]);
  const tactical=c&&!!c.globalId&&!c.uncertain&&(/^(PLAYER|GOALKEEPER)_/.test(c.role));
  if(p.role==='SPECTATOR'){if(tactical)metrics.falseSpectatorDetections++;return;}
  if(p.role==='REFEREE'){if(tactical)metrics.refereeAsPlayerErrors++;return;}
  visible++;const stat=metrics.perPlayer[p.id]??={visibleFrames:0,trackedFrames:0,coverage:0};stat.visibleFrames++;
  const prior=previous.get(p.id);
  if(tactical){tracked++;stat.trackedFrames++;confidence+=c.identityConfidence;confidenceN++;
   if(prior?.globalId&&prior.globalId!==c.globalId)metrics.idSwitches++;
   if(prior?.gap&&prior?.ever)metrics.fragmentedPlayerTracks++;
   const set=ids.get(p.id)||new Set();set.add(c.globalId);ids.set(p.id,set);
   previous.set(p.id,{globalId:c.globalId,ever:true,gap:false});
  }else previous.set(p.id,{...prior,gap:true});
  if(c)evaluated.set(c.localId+':'+c.time.toFixed(3),p.id);
 });
 candidates.forEach((p,i)=>{if(p.globalId&&!p.uncertain&&/^(PLAYER|GOALKEEPER)_/.test(p.role)&&!used.has(i))metrics.unmatchedPredictedPlayers++;});
}
// Map global IDs to their majority annotated identity before evaluating returns.
const votes=new Map();for(const o of pred.observations||[]){const gt=evaluated.get(o.localId+':'+o.time.toFixed(3));if(gt&&o.globalId){const v=votes.get(o.globalId)||new Map();v.set(gt,(v.get(gt)||0)+1);votes.set(o.globalId,v);}}
for(const event of pred.reidEvents||pred.soccer?.events||[])if(event.accepted){const gt=evaluated.get(event.localId+':'+event.time.toFixed(3)),v=votes.get(event.globalId),expected=v&&[...v].sort((a,b)=>b[1]-a[1])[0]?.[0];if(!gt||!expected)metrics.unscoredReIDEvents++;else if(gt===expected)metrics.successfulReIDEvents++;else metrics.incorrectReIDEvents++;}
metrics.duplicateGlobalIdentities=[...ids.values()].reduce((n,s)=>n+Math.max(0,s.size-1),0);
metrics.averagePlayerIdentityConfidence=confidenceN?confidence/confidenceN:0;metrics.visiblePlayerCoverage=visible?tracked/visible:0;
for(const p of Object.values(metrics.perPlayer))p.coverage=p.trackedFrames/p.visibleFrames;
console.log(JSON.stringify({...metrics,notes:['Only annotated frames are scored. Include spectators and referees in ground truth.','Scores are not calibrated probabilities. Re-ID events without matching annotations are unscored.','A genuine complete disappearance is not a visible-frame miss.']},null,2));
