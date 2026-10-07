const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {makeFrame}=require('./helpers/ts-require.cjs');
const {SoccerTracker,explainFrame,defaultSoccerOptions}=require('../lib/soccer/pipeline.ts');
const {LocalTracker,localKey,localId}=require('../lib/soccer/local-tracker.ts');
const {combineReid,REID_WEIGHTS,IdentityManager}=require('../lib/soccer/identity.ts');
const {DESCRIPTOR_LENGTH}=require('../lib/soccer/appearance.ts');
const {iou}=require('../lib/detection-core.ts');
const {rejectText}=require('../lib/soccer/types.ts');
const tracking=require('../lib/tracking.ts');

// ---------- synthetic broadcast scene (640x360) ----------
// Stands (grey, people painted as fans) at the top, advertising boards, far touchline at y=.2, striped pitch
// with a halfway line, near touchline at y=.85, a strip of run-off grass, then a non-grass bench/track area.
const W=640,H=360,FAR=.2,NEAR=.85,BENCH=.9;
let seed=7;const rnd=()=>{seed=(seed*1103515245+12345)%2147483648;return seed/2147483648;};
const KITS={
 red:{shirt:[200,30,35],shorts:[240,240,240],socks:[200,30,35]},
 blue:{shirt:[30,60,190],shorts:[20,20,30],socks:[240,240,240]},
 ref:{shirt:[235,215,40],shorts:[15,15,15],socks:[15,15,15]},
 fan:{shirt:[200,30,35],shorts:[60,60,60],socks:[60,60,60]},
 keeper:{shirt:[245,130,20],shorts:[245,130,20],socks:[245,130,20]},
 redDark:{shirt:[200,30,35],shorts:[20,20,20],socks:[20,20,20]}, // same shirt as red, different shorts/socks
 coat:{shirt:[60,60,70],shorts:[60,60,70],socks:[30,30,30]},
};
const SKIN=[205,160,130],BOOT=[25,25,25];
const base=makeFrame(W,H,(x,y)=>{
 const Y=y/H,X=x/W,n=(rnd()-.5)*12;
 const c=Y<.15?(((x*7+y*13)%23<3)?[200,200,200]:[95,95,105]):Y<.18?(Math.floor(X*12)%2?[235,235,235]:[40,60,200]):
  Y>=BENCH?(Math.floor(X*20)%2?[90,70,50]:[150,150,155]):
  (Math.abs(Y-FAR)<.004||Math.abs(Y-NEAR)<.004||(Y>FAR&&Y<NEAR&&Math.abs(X-.5)<.002))?[230,235,230]:
  (Math.floor(X*16)%2?[40,140,50]:[52,162,62]);
 return c.map(v=>Math.max(0,Math.min(255,v+n)));
});
// A person is {key, x, y (foot point), kit, h?}; the box follows a simple perspective size model.
const boxOf=p=>{const h=p.h??(.14+.08*p.y),w=h*.32*H/W;return {x:p.x-w/2,y:p.y-h,w,h};};
// People lower in the frame stand in front (painted last), as the appearance model assumes.
function paint(people){
 const data=new Uint8ClampedArray(base.data);
 for(const p of [...people].sort((a,b)=>a.y-b.y)){
  const b=boxOf(p),k=KITS[p.kit],x0=Math.round(b.x*W),y0=Math.round(b.y*H),w=Math.round(b.w*W),h=Math.round(b.h*H);
  for(let j=0;j<h;j++)for(let i=0;i<w;i++){
   const x=x0+i,y=y0+j;if(x<0||y<0||x>=W||y>=H)continue;const u=i/w,v=j/h;let c=null;
   if(v<.15){if(u>.35&&u<.65)c=SKIN;}else if(v<.5){if(u>.15&&u<.85)c=k.shirt;}else if(v<.7){if(u>.2&&u<.8)c=k.shorts;}
   else if(v<.94){if((u>.25&&u<.45)||(u>.55&&u<.75))c=k.socks;}else if((u>.22&&u<.46)||(u>.54&&u<.78))c=BOOT;
   if(c){const q=(y*W+x)*4;data[q]=c[0];data[q+1]=c[1];data[q+2]=c[2];}
  }
 }
 return {width:W,height:H,data};
}
const detect=(p,i)=>({id:'d'+i,kind:'person',score:p.score??.85,box:boxOf(p)});
const still={scale:1,dx:0,dy:0,reliable:true,cut:false};
const roster=(inactive=[])=>[...['A','B'].flatMap(t=>Array.from({length:11},(_,i)=>({id:t+(i+1),team:t,number:String(i+1),active:!inactive.includes(t+(i+1))}))),...[1,2,3].map(i=>({id:'R'+i,team:'ref',number:'',role:'referee',active:true}))];
const T=k=>Number((k*.2).toFixed(4));
// Drives tracker.step every 0.2 s. scene(t,k) returns people; visible:false = painted but not detected,
// paint:false = neither. Returns per-step {t,k,r,who,seen}: who maps person key -> playerId observed at that time
// ('-' none). Observations are pooled over the whole run (some are emitted a few steps late, like the UI appends
// them); during the run (o.each) who only reflects the current step's output.
function run(tracker,steps,scene,o={}){
 const log=[],pool=[];
 for(let k=o.k0??0;k<(o.k0??0)+steps;k++){
  const t=T(k),people=scene(t,k).filter(Boolean),seen=people.filter(p=>p.visible!==false&&p.paint!==false);
  const r=tracker.step({time:t,frame:o.frame?.(t,k)??paint(people.filter(p=>p.paint!==false)),detections:o.detections?.(t,k,seen)??seen.map(detect),camera:o.camera?.(t,k)??still,anchors:o.anchors?.(t,k,seen)??[],roster:o.roster?.(t,k)??roster(),...(o.homography?{homography:o.homography}:{})});
  pool.push(...r.observations);
  const who={};for(const p of seen)who[p.key]=observed(r.observations,boxOf(p),t)?.playerId??'-';
  const entry={t,k,r,who,seen};log.push(entry);o.each?.(entry);
 }
 for(const l of log)for(const p of l.seen)l.who[p.key]=observed(pool,boxOf(p),l.t)?.playerId??'-';
 log.pool=pool;
 return log;
}
// The direct ('detection') observation at time t that best matches a person's box.
function observed(list,box,t){let best,score=.6;for(const o of list)if(o.evidence==='detection'&&Math.abs(o.time-t)<1e-6){const v=iou(o.box,box);if(v>score){score=v;best=o;}}return best;}
const events=(log,kind)=>log.flatMap(l=>l.r.events.filter(e=>!kind||e.kind===kind));
const snapshotOf=(tracker,id)=>tracker.snapshot().find(s=>s.playerId===id);

const RED=[{key:'r1',x:.15,y:.45,kit:'red'},{key:'r2',x:.3,y:.6,kit:'red'},{key:'r3',x:.45,y:.75,kit:'red'},{key:'r4',x:.62,y:.4,kit:'red'}];
const BLUE=[{key:'b1',x:.2,y:.72,kit:'blue'},{key:'b2',x:.4,y:.35,kit:'blue'},{key:'b3',x:.72,y:.65,kit:'blue'},{key:'b4',x:.87,y:.3,kit:'blue'}];
const REF={key:'ref',x:.55,y:.55,kit:'ref'};
// Spectators in the stands, a coach on the bench and a steward on the run-off grass beyond the near touchline.
const OUTSIDE=[{key:'fan1',x:.3,y:.12,h:.07,kit:'fan'},{key:'fan2',x:.5,y:.13,h:.07,kit:'blue'},{key:'fan3',x:.7,y:.12,h:.07,kit:'fan'},{key:'coach',x:.9,y:.97,h:.12,kit:'coat'},{key:'steward',x:.1,y:.895,kit:'red'}];
const wobble=(p,k,i)=>({...p,x:p.x+.006*Math.sin(k*.7+i)});

// Scenario A: 8 players + referee, people outside the field, and a late red player entering at step 8.
let scenarioA;
function getA(){
 if(scenarioA)return scenarioA;
 const tracker=new SoccerTracker();
 const log=run(tracker,22,(t,k)=>[...[...RED,...BLUE,REF].map((p,i)=>wobble(p,k,i)),...OUTSIDE,k>=8&&{key:'late',x:.8,y:.75,kit:'red'}]);
 return scenarioA={tracker,log};
}

test('spectators, coach and steward are rejected every step and never tracked or promoted',()=>{
 const {log}=getA();
 for(const l of log){
  for(const p of OUTSIDE){
   const box=boxOf(p),rej=l.r.debug.rejected.find(r=>iou(r.box,box)>.9);
   assert.ok(rej,`${p.key} rejected at ${l.t}`);
   assert.equal(rej.reason,p.key.startsWith('fan')?'audience':'outside-pitch',`${p.key}: ${rej.detail}`);
   assert.ok(rejectText[rej.reason].startsWith('REJECTED'));
   assert.ok(rej.detail.includes('beyond'),rej.detail);
   assert.ok(!l.r.debug.tracks.some(d=>iou(d.box,box)>.3),`${p.key} tracked at ${l.t}`);
   assert.ok(!l.r.observations.some(o=>iou(o.box,box)>.3),`${p.key} observed at ${l.t}`);
  }
 }
 assert.equal(events(log,'promotion').length,10);
 assert.ok(log[0].r.debug.pitch.reliable);
});

test('candidates need minHits detection steps before any observation; earlier steps are emitted retroactively',()=>{
 const {log}=getA(),minHits=defaultSoccerOptions.minHits;
 for(const l of log.slice(0,minHits-1))assert.equal(l.r.observations.length,0,`no observation at step ${l.k}`);
 const first=log[minHits-1].r;
 assert.equal(first.observations.filter(o=>o.evidence==='detection').length,9);
 const retro=first.observations.filter(o=>o.evidence==='reidentified');
 assert.equal(retro.length,9*(minHits-1));
 assert.ok(retro.every(o=>o.time<first.debug.time));
 // The late player is detected from step 8 but observed only from step 8+minHits-1.
 for(const l of log.filter(l=>l.k>=8))assert.equal(l.who.late==='-',l.k<8+minHits-1,`late player at step ${l.k}: ${l.who.late}`);
 const late=log.find(l=>l.who.late&&l.who.late!=='-');
 assert.ok(late.r.events.some(e=>e.kind==='new-identity'&&e.playerId===late.who.late));
 // Candidate debug labels while waiting.
 const cand=log[9].r.debug.tracks.find(d=>iou(d.box,boxOf({x:.8,y:.75}))>.6);
 assert.equal(cand.label,'CAND');assert.equal(cand.state,'candidate');assert.equal(cand.roleLabel,'CANDIDATE');
});

test('players get stable team identities; the referee kit becomes REFEREE in an R slot, never a team slot',()=>{
 const {log,tracker}=getA(),last=log[log.length-1];
 const ids=Object.fromEntries([...RED,...BLUE,REF].map(p=>[p.key,last.who[p.key]]));
 for(const p of RED)assert.match(ids[p.key],/^A(\d+)$/);
 for(const p of BLUE)assert.match(ids[p.key],/^B(\d+)$/);
 assert.equal(ids.ref,'R1');
 assert.equal(new Set(Object.values(ids)).size,Object.keys(ids).length,'no duplicate playerIds');
 // Stable while visible.
 for(const l of log.slice(defaultSoccerOptions.minHits))for(const p of [...RED,...BLUE,REF])assert.equal(l.who[p.key],ids[p.key],`${p.key} at ${l.t}`);
 const ref=last.r.debug.tracks.find(d=>d.playerId==='R1');
 assert.equal(ref.roleLabel,'REFEREE');assert.equal(ref.label,'REF-1');assert.equal(ref.team,undefined);
 assert.ok(events(log,'role').some(e=>/REFEREE/.test(e.message)));
 const saved=snapshotOf(tracker,'R1');assert.equal(saved.role,'referee');assert.equal(saved.team,undefined);
 assert.ok(!tracker.snapshot().some(s=>s.team&&s.role==='referee'));
 const red=last.r.debug.tracks.find(d=>d.playerId===ids.r1);
 assert.equal(red.roleLabel,'PLAYER_TEAM_A');assert.match(red.label,/^A-\d\d$/);assert.ok(red.identity>.5);
 // The keeper slot (jersey 1) is left free for a goalkeeper.
 assert.ok(!Object.values(ids).includes('A1')&&!Object.values(ids).includes('B1'));
 assert.ok(tracker.teamModel.a&&tracker.teamModel.b,'two kits learnt');
 assert.ok(tracker.teamModel.referee,'referee kit learnt from the confirmed official');
});

test('a player leaving through the right edge returns on a new local track with the same playerId (reid event with scores)',()=>{
 const tracker=new SoccerTracker(),others=[...RED.slice(1),...BLUE,REF];
 let away;
 const log=run(tracker,60,(t,k)=>{
  let p;
  if(t<=3){const x=.8+.02*k,b=boxOf({x,y:.5});p={key:'P',x,y:.5,kit:'red',...(b.x+b.w<=1?{}:{paint:false})};}
  else if(t>=8){const b=boxOf({x:.5,y:.5});p={key:'P',x:1-b.w/2-.01*(k-40),y:.5,kit:'red'};}
  return [p,...others];
 },{each:l=>{if(l.k===35)away=tracker.snapshot();}});
 const before=log.filter(l=>l.t>1&&l.t<2).map(l=>l.who.P),idBefore=before[0];
 assert.match(idBefore,/^A\d+$/);assert.ok(before.every(id=>id===idBefore));
 // While away the identity is kept as OFF_SCREEN with its exit edge (never deleted).
 const mid=away.find(s=>s.playerId===idBefore);assert.equal(mid.status,'offscreen');assert.equal(mid.exitEdge,'right');
 const reid=events(log,'reid');
 assert.equal(reid.length,1);
 const e=reid[0];
 assert.equal(e.playerId,idBefore);
 const oldTrack=log.find(l=>l.t===1).r.debug.tracks.find(d=>d.playerId===idBefore).track;
 assert.notEqual(e.track,oldTrack,'a new local track');
 for(const k of ['appearance','jersey','uniform','spatial','movement','temporal','final','missingSeconds'])assert.ok(Number.isFinite(e.scores[k]),k);
 assert.equal(e.scores.team,true);
 assert.ok(e.scores.final>=defaultSoccerOptions.reidMin);
 assert.ok(e.scores.appearance>.8&&e.scores.movement>.9,JSON.stringify(e.scores));
 assert.ok(e.scores.missingSeconds>4);
 assert.match(e.message,/re-identified as TeamA_Player_\d\d/);
 // After the reconnect the returning player is observed as the same identity; earlier steps are back-filled.
 const after=log.filter(l=>l.t>e.time);
 assert.ok(after.length>5&&after.every(l=>l.who.P===idBefore));
 const retro=log.find(l=>l.t===e.time).r.observations.filter(o=>o.playerId===idBefore&&o.evidence==='reidentified');
 assert.ok(retro.length>=2&&retro.every(o=>o.time>=8&&o.time<e.time));
 assert.equal(events(log,'new-identity').filter(x=>x.time>1).length,0,'no new identity for the returning player');
 assert.equal(new Set(tracker.snapshot().map(s=>s.playerId)).size,9);
});

test('an identity that leaves is kept OFF_SCREEN (never deleted) with its exit edge, last box and gallery',()=>{
 const tracker=new SoccerTracker(),others=[...RED.slice(1),...BLUE,REF];
 const log=run(tracker,30,(t,k)=>{const x=.8+.02*k,b=boxOf({x,y:.5});return [{key:'P',x,y:.5,kit:'red',...(b.x+b.w<=1?{}:{paint:false})},...others];});
 const id=log[6].who.P,saved=snapshotOf(tracker,id);
 assert.equal(saved.status,'offscreen');assert.equal(saved.exitEdge,'right');
 assert.ok(saved.gallery.length>=1&&saved.gallery.every(v=>v.length===DESCRIPTOR_LENGTH));
 assert.ok(saved.lastSeen<3&&saved.lastBox.x+saved.lastBox.w>.96);
});

// Three red players are identified, then a camera cut; afterwards one red player (Q) is visible. From step 25 the
// user labels two other red players as P2 and P3 (anchors).
function cutScenario(options){
 const tracker=new SoccerTracker(options),red=[{key:'P1',x:.3,y:.5,kit:'red'},{key:'P2',x:.5,y:.7,kit:'red'},{key:'P3',x:.7,y:.45,kit:'red'}];
 const blue=[{key:'b1',x:.2,y:.35,kit:'blue'},{key:'b2',x:.4,y:.3,kit:'blue'},{key:'b3',x:.85,y:.65,kit:'blue'}];
 const blue2=blue.map(b=>({...b,x:1-b.x,y:b.y+.1})),ids={};
 const log=run(tracker,40,(t,k)=>{
  if(k<10)return [...red,...blue];
  return [{key:'Q',x:.5,y:.55,kit:'red'},...blue2,...(k>=25?[{key:'S2',x:.2,y:.6,kit:'red'},{key:'S3',x:.8,y:.6,kit:'red'}]:[])];
 },{camera:(t,k)=>({...still,cut:k===10}),each:l=>{if(l.k===9)Object.assign(ids,l.who);},
  anchors:(t,k,seen)=>k===25?[{playerId:ids.P2,box:boxOf(seen.find(p=>p.key==='S2'))},{playerId:ids.P3,box:boxOf(seen.find(p=>p.key==='S3'))}]:[]});
 return {tracker,log,ids};
}
test('two same-kit teammates missing without spatial evidence are deferred, never guessed',()=>{
 const {log,ids}=cutScenario();
 assert.equal(new Set([ids.P1,ids.P2,ids.P3]).size,3);
 const waiting=log.filter(l=>l.k>=10&&l.k<25);
 // No observation for Q and no new identity for it while P1, P2 and P3 are equally likely.
 assert.ok(waiting.every(l=>l.who.Q==='-'));
 const qTrack=waiting[waiting.length-1].r.debug.tracks.find(d=>iou(d.box,boxOf({x:.5,y:.55}))>.6);
 assert.equal(qTrack.state,'uncertain');assert.equal(qTrack.label,'A-?');assert.equal(qTrack.roleLabel,'IDENTITY_UNCERTAIN');
 assert.match(qTrack.reason,/deferred/);
 const deferred=events(waiting,'deferred').filter(e=>e.track===qTrack.track);
 assert.ok(deferred.length>=1&&deferred.length<=4,'deferred events are rate limited');
 for(const e of deferred){assert.ok(e.scores.secondBest!==undefined);assert.ok(e.scores.final-e.scores.secondBest<defaultSoccerOptions.reidMargin);}
 assert.equal(events(waiting,'new-identity').length,0);
 assert.equal(events(waiting,'reid').length,0);
 assert.ok(waiting.every(l=>['b1','b2','b3'].every(k=>l.who[k]==='-')),'same-kit blue players are not guessed either');
 // Even after the user labels P2 and P3 elsewhere, Q could be an unseen teammate (8 free team-A slots): it waits.
 const later=log.filter(l=>l.k>=25);
 assert.ok(later.every(l=>l.who.S2===ids.P2&&l.who.S3===ids.P3));
 assert.ok(later.every(l=>l.who.Q==='-'),'no guess while unseen teammates are possible');
 assert.ok(events(later,'deferred').some(e=>e.track===qTrack.track&&/unseen teammate/.test(e.message)));
 assert.ok(!log.pool.some(o=>o.track===qTrack.track));
});
test('with every teammate accounted for, the last missing identity is resolved by elimination',()=>{
 const {log,ids}=cutScenario({maxPerTeam:3});
 const waiting=log.filter(l=>l.k>=10&&l.k<25),resolved=log.filter(l=>l.k>=25);
 assert.ok(waiting.every(l=>l.who.Q==='-'));
 assert.ok(resolved.every(l=>l.who.Q===ids.P1&&l.who.S2===ids.P2&&l.who.S3===ids.P3));
 const qTrack=log[24].r.debug.tracks.find(d=>iou(d.box,boxOf({x:.5,y:.55}))>.6).track;
 const reid=events(resolved,'reid').find(e=>e.playerId===ids.P1);
 assert.ok(reid&&reid.track===qTrack);
 assert.equal(reid.scores.secondBest,undefined);assert.match(reid.message,/no spatial evidence/);
 assert.equal(events(resolved,'anchor').length,2);
 assert.ok(!log.pool.some(o=>o.track===qTrack&&o.playerId!==ids.P1),'Q is never labelled P2 or P3');
});

test('team capacity: a 12th distinct team-A participant gets no 12th identity and raises a sanity event',()=>{
 const tracker=new SoccerTracker();
 const reds=Array.from({length:12},(_,i)=>({key:'a'+i,x:.07+.086*(i%11)+(i>=11?.04:0),y:i>=11?.75:[.3,.45,.6,.75][i%4],kit:'red'}));
 const blues=Array.from({length:5},(_,i)=>({key:'b'+i,x:.113+.172*i,y:i%2?.52:.38,kit:'blue'}));
 const log=run(tracker,60,(t,k)=>[...reds,...blues].map((p,i)=>wobble(p,k,i)));
 const last=log[log.length-1],aIds=reds.map(p=>last.who[p.key]).filter(id=>id!=='-');
 assert.equal(aIds.length,11);
 assert.equal(new Set(aIds).size,11);
 assert.ok(aIds.every(id=>/^A\d+$/.test(id)));
 const ids=new Set(log.flatMap(l=>l.r.observations.map(o=>o.playerId)).filter(id=>id.startsWith('A')));
 assert.equal(ids.size,11,'never more than 11 team-A identities');
 const sanity=events(log,'sanity');
 assert.ok(sanity.some(e=>/Team A already has 11 identities/.test(e.message)),sanity.map(e=>e.message).join('\n'));
 assert.ok(sanity.length<=4,'sanity events are rate limited');
 const odd=last.r.debug.tracks.find(d=>d.state==='uncertain');
 assert.ok(odd);assert.equal(odd.label,'A-?');assert.equal(odd.playerId,undefined);
 assert.equal(last.r.observations.filter(o=>o.track===odd.track).length,0);
 // The uncertain track's observation buffer is bounded.
 const state=tracker.ids.tracks.get(odd.track);assert.ok(state.buffer.length<=50);
 assert.equal(tracker.snapshot().filter(s=>s.team==='A').length,11);
});

test('two same-kit teammates crossing keep their identities (no swap), and the crossing is flagged',()=>{
 const tracker=new SoccerTracker(),mates=[{key:'m1',x:.15,y:.75,kit:'red'},{key:'m2',x:.85,y:.75,kit:'red'},{key:'b1',x:.2,y:.35,kit:'blue'},{key:'b2',x:.4,y:.3,kit:'blue'},{key:'b3',x:.7,y:.3,kit:'blue'},{key:'b4',x:.9,y:.4,kit:'blue'}];
 const log=run(tracker,30,(t,k)=>{
  const P={key:'P',x:.3+.02*k,y:.55,kit:'red'},Q={key:'Q',x:.7-.02*k,y:.57,kit:'red'};
  // While they overlap heavily the detector only sees the front player (Q).
  return [{...P,visible:iou(boxOf(P),boxOf(Q))<=.45},Q,...mates];
 });
 const p=log[6].who.P,q=log[6].who.Q;
 assert.match(p,/^A\d+$/);assert.match(q,/^A\d+$/);assert.notEqual(p,q);
 for(const l of log.slice(5)){if(l.who.P)assert.ok(l.who.P===p||l.who.P==='-',`P at ${l.t}: ${l.who.P}`);assert.ok(l.who.Q===q||l.who.Q==='-',`Q at ${l.t}: ${l.who.Q}`);}
 assert.equal(log[log.length-1].who.P,p);assert.equal(log[log.length-1].who.Q,q);
 assert.equal(events(log,'swap-corrected').length,0);
 // Kits are identical and one player was hidden: identities kept, confidence capped, both flagged.
 const flag=events(log,'swap-uncertain');assert.equal(flag.length,1);
 const after=log.find(l=>l.t===flag[0].time).r;
 assert.ok(after.observations.filter(o=>o.playerId===p||o.playerId===q).every(o=>o.conf<=.5));
 assert.deepEqual(after.issues.map(i=>i.playerId).sort(),[p,q].sort());
 assert.ok(log[log.length-1].r.observations.find(o=>o.playerId===p).conf>.5,'confidence recovers after ~2 s');
});

test('players of different kits crossing keep their identities without warnings',()=>{
 const tracker=new SoccerTracker(),mates=[{key:'m1',x:.15,y:.75,kit:'red'},{key:'m2',x:.85,y:.75,kit:'red'},{key:'m3',x:.5,y:.8,kit:'red'},{key:'b1',x:.2,y:.35,kit:'blue'},{key:'b2',x:.4,y:.3,kit:'blue'},{key:'b3',x:.7,y:.3,kit:'blue'}];
 // They meet and turn back: motion alone would suggest they crossed.
 const log=run(tracker,30,(t,k)=>{const m=Math.max(0,k<=10?k:20-k),P={key:'P',x:.3+.02*m,y:.55,kit:'red'},Q={key:'Q',x:.7-.02*m,y:.57,kit:'blue'};return [{...P,visible:iou(boxOf(P),boxOf(Q))<=.45},Q,...mates];});
 const p=log[6].who.P,q=log[6].who.Q;
 assert.match(p,/^A\d+$/);assert.match(q,/^B\d+$/);
 for(const l of log.slice(5))assert.ok((l.who.P===p||l.who.P===undefined)&&l.who.Q===q,`${l.t}: ${l.who.P}/${l.who.Q}`);
 assert.equal(events(log,'swap-uncertain').length+events(log,'swap-corrected').length,0);
});

test('user anchors override automatic identities immediately',()=>{
 const tracker=new SoccerTracker(),people=[...RED,...BLUE,REF];
 const log=run(tracker,20,(t,k)=>[...people,k>=12&&{key:'new',x:.85,y:.6,kit:'blue'}],{anchors:(t,k,seen)=>{
  if(k===8)return [{playerId:'A9',box:boxOf(seen.find(p=>p.key==='r2'))}];
  if(k===12)return [{playerId:'B9',box:boxOf(seen.find(p=>p.key==='new'))}];
  if(k===15)return [{playerId:'A9',box:boxOf(seen.find(p=>p.key==='r3'))}];
  return [];}});
 const auto=log[7].who.r2;assert.match(auto,/^A\d+$/);assert.notEqual(auto,'A9');
 // Anchor on a confirmed track: the label replaces the automatic identity at once (conf 1).
 const at8=log[8];assert.equal(at8.who.r2,'A9');
 assert.equal(observed(at8.r.observations,boxOf(people[1]),at8.t).conf,1);
 const ev=at8.r.events.find(e=>e.kind==='anchor');assert.equal(ev.playerId,'A9');assert.match(ev.message,new RegExp(`replaces automatic ${auto.replace(/^A/,'A-0')}`));
 assert.ok(!at8.r.observations.some(o=>o.playerId===auto&&Math.abs(o.time-at8.t)<1e-6),'the replaced identity is not observed on that player');
 const old=snapshotOf(tracker,auto);assert.ok(old,'the replaced identity is kept, not deleted');
 // Anchor on a brand-new person: confirmed on its first frame, no minHits wait.
 assert.equal(log[12].who.new,'B9');
 assert.equal(log[11].who.new,undefined);
 // Moving a label to another player releases the previous holder (it is re-identified, never duplicated).
 const at15=log[15];assert.equal(at15.who.r3,'A9');assert.notEqual(at15.who.r2,'A9');
 assert.match(at15.r.events.find(e=>e.kind==='anchor'&&e.playerId==='A9').message,/was on track/);
 for(const l of log)assert.equal(new Set(l.r.observations.filter(o=>Math.abs(o.time-l.t)<1e-6).map(o=>o.playerId)).size,l.r.observations.filter(o=>Math.abs(o.time-l.t)<1e-6).length,`duplicate playerId at ${l.t}`);
 const a9=snapshotOf(tracker,'A9');assert.equal(a9.anchored,true);assert.equal(a9.status,'active');
});

test('a camera cut ends every local track: identities become missing/offscreen, are not deleted, and are never guessed',()=>{
 const tracker=new SoccerTracker(),people=[...RED,...BLUE,REF];
 const closeUp=makeFrame(W,H,(x,y)=>[120+(x*y)%30,110,100]);
 let afterCut;
 const log=run(tracker,30,(t,k)=>k>=8&&k<12?[]:people.map(p=>k>=12?{...p,x:1-p.x}:p),{
  camera:(t,k)=>({...still,cut:k===8||k===12}),frame:(t,k)=>k>=8&&k<12?closeUp:undefined,
  each:l=>{if(l.k===8)afterCut=l.r.debug.time;}});
 const before=log[7];assert.equal(Object.values(before.who).filter(id=>id!=='-').length,9);
 const ids=Object.values(before.who);
 // Close-up: pitch unreliable, nothing tracked or promoted, all identities kept.
 for(const l of log.slice(8,12)){assert.equal(l.r.debug.pitch.reliable,false);assert.equal(l.r.observations.length,0);}
 const saved=tracker.snapshot();
 assert.equal(saved.length,9);
 assert.ok(afterCut>0);
 // After the second cut nothing is created anew: the unique referee kit re-identifies, same-kit players wait.
 assert.equal(events(log.slice(8),'new-identity').length,0);
 const refReid=events(log.slice(12),'reid').find(e=>e.playerId==='R1');assert.ok(refReid,'referee reconnected by kit');
 assert.equal(log[log.length-1].who.ref,'R1');
 for(const l of log.slice(12))for(const p of [...RED,...BLUE])assert.equal(l.who[p.key],'-',`${p.key} guessed at ${l.t}`);
 assert.ok(events(log.slice(12),'deferred').length>0);
 const statuses=tracker.snapshot().filter(s=>s.playerId!=='R1').map(s=>s.status);
 assert.ok(statuses.every(s=>s==='missing'||s==='offscreen'),statuses.join());
 assert.deepEqual(tracker.snapshot().map(s=>s.playerId).sort(),ids.sort());
});

test('snapshot() / constructor(saved) round trip keeps identities, galleries and roles; restored identities re-identify',()=>{
 const {tracker}=getA(),snap=JSON.parse(JSON.stringify(tracker.snapshot()));
 assert.equal(snap.length,10);
 for(const s of snap){
  assert.ok(s.gallery.length>=1&&s.gallery.length<=6);assert.ok(s.gallery.every(v=>v.length===DESCRIPTOR_LENGTH&&v.length<=80));
  assert.ok(s.lastBox.w>0&&s.lastBox.h>0);assert.ok(['player','referee','goalkeeper'].includes(s.role));
 }
 // Schema compatible with the saved document.
 const doc=tracking.ensureOfficials(tracking.initialTracking());
 assert.equal(tracking.cleanIdentities(doc,snap).length,snap.length);
 const restored=new SoccerTracker({},snap),again=restored.snapshot();
 assert.deepEqual(again.map(s=>[s.playerId,s.role,s.team,s.gallery,s.anchored]),snap.map(s=>[s.playerId,s.role,s.team,s.gallery,s.anchored]));
 for(const s of again)assert.ok(s.status==='missing'||s.status==='offscreen');
 // Long after the save: same-kit players stay deferred, the unique referee kit reconnects to R1.
 const log=run(restored,14,()=>[...RED,...BLUE,REF].map(p=>({...p,y:Math.min(.8,p.y+.05)})),{k0:300});
 assert.equal(log[log.length-1].who.ref,'R1');
 assert.equal(events(log,'new-identity').length,0);
 assert.ok(log.every(l=>RED.every(p=>l.who[p.key]==='-')));
});

test('roster changes: inactive slots are never observed or assigned; a substituted identity stops at once',()=>{
 const tracker=new SoccerTracker(),inactive=['A2','A3','B2'];
 let subbed;
 const log=run(tracker,16,()=>[...RED,...BLUE,REF],{roster:(_t,k)=>roster(k>=10&&subbed?[...inactive,subbed]:inactive),each:l=>{if(l.k===9)subbed=l.who.r1;}});
 for(const l of log){
  const active=new Set(roster(l.k>=10?[...inactive,subbed]:inactive).filter(s=>s.active).map(s=>s.id));
  for(const o of l.r.observations)assert.ok(active.has(o.playerId),`${o.playerId} observed while inactive at ${l.t}`);
 }
 assert.ok(!Object.values(log[9].who).some(id=>inactive.includes(id)));
 assert.match(subbed,/^A\d+$/);
 assert.equal(log[12].who.r1,'-');
 assert.equal(snapshotOf(tracker,subbed).status,'substituted');
 const d=log[12].r.debug.tracks.find(x=>iou(x.box,boxOf(RED[0]))>.6);assert.equal(d.label,'UNK');assert.match(d.reason,/substituted/);
});

test('rememberLabel seeds identities from earlier labels: kits are oriented by them; a kit alone never reconnects',()=>{
 const reds=[{key:'r1',x:.12,y:.5,kit:'red'},{key:'r2',x:.5,y:.5,kit:'red'},{key:'r3',x:.88,y:.5,kit:'red'}];
 const blues=[{key:'b1',x:.3,y:.35,kit:'blue'},{key:'b2',x:.7,y:.35,kit:'blue'},{key:'b3',x:.3,y:.75,kit:'blue'}];
 const ref={key:'ref',x:.7,y:.75,kit:'ref'},all=[...blues,...reds,ref]; // blue first: unlabelled, blue would become team A
 const label=(tracker,time)=>{const f=paint(all);tracker.rememberLabel('A7','A',f,boxOf(reds[0]),time);tracker.rememberLabel('B9','B',f,boxOf(blues[2]),time);tracker.rememberLabel('R3','ref',f,boxOf(ref),time);};
 // Labels from the previous frame: spatial evidence is usable, the labelled players reconnect at once.
 const fresh=new SoccerTracker();label(fresh,4.9);
 assert.deepEqual(fresh.snapshot().map(s=>[s.playerId,s.status,s.anchored]),[['A7','missing',true],['B9','missing',true],['R3','missing',true]]);
 const log=run(fresh,10,()=>all,{k0:25}),last=log[log.length-1];
 assert.equal(last.who.r1,'A7');assert.equal(last.who.b3,'B9');assert.equal(last.who.ref,'R3');
 for(const p of reds)assert.match(last.who[p.key],/^A\d+$/);
 for(const p of blues)assert.match(last.who[p.key],/^B\d+$/);
 assert.equal(fresh.teamModel.anchored,true);
 for(const id of ['A7','B9']){const e=events(log,'reid').find(x=>x.playerId===id);assert.ok(e,id);assert.ok(e.scores.spatial>.9);}
 // Labels from long ago: one red and one blue person later could be any unseen teammate -> deferred, never guessed.
 const old=new SoccerTracker();label(old,0);
 const later=[{...blues[2],x:.6},{...reds[0],x:.3},{...ref,x:.45}];
 const log2=run(old,14,()=>later,{k0:25});
 assert.ok(log2.every(l=>l.who.r1==='-'&&l.who.b3==='-'));
 assert.ok(events(log2,'deferred').some(e=>e.playerId==='A7'&&/unseen teammate/.test(e.message)));
 assert.equal(events(log2,'new-identity').length,0);
 assert.equal(log2[log2.length-1].who.ref,'R3','the official kit is matched by appearance (officials have no unseen-teammate hypothesis)');
});

test('goalkeeper: a keeper kit near a goal waits for a label, then keeps its identity (with pitch coordinates)',()=>{
 const tracker=new SoccerTracker(),H=[1,0,0,0,1,0,0,0,1]; // image -> pitch identity homography: x < .17 is near the left goal
 const log=run(tracker,70,(t,k)=>{const gk={key:'gk',x:k>=45?.02+.004*(k-45):.08,y:.5,kit:'keeper'};return [k>=30&&k<45?null:gk,...RED,...BLUE,REF];},
  {homography:H,anchors:(t,k,seen)=>k===12?[{playerId:'A1',box:boxOf(seen.find(p=>p.key==='gk'))}]:[]});
 const waiting=log.slice(0,12);
 assert.ok(waiting.every(l=>l.who.gk==='-'),'never guessed before the label');
 const gk=waiting[11].r.debug.tracks.find(d=>iou(d.box,boxOf({x:.08,y:.5}))>.6);
 assert.equal(gk.label,'GK-?');assert.equal(gk.state,'uncertain');
 assert.ok(events(waiting,'deferred').some(e=>/goalkeeper/.test(e.message)));
 assert.ok(log.slice(12,30).every(l=>l.who.gk==='A1'));
 assert.ok(events(log,'role').some(e=>e.playerId==='A1'&&/goalkeeper/.test(e.message)));
 const back=log.slice(52);
 assert.ok(back.every(l=>l.who.gk==='A1'),back.map(l=>l.who.gk).join());
 assert.ok(events(log,'reid').some(e=>e.playerId==='A1'));
 assert.equal(back[back.length-1].r.debug.tracks.find(d=>d.playerId==='A1').roleLabel,'GOALKEEPER_TEAM_A');
 assert.equal(events(log,'team-change').length,0,'a labelled keeper does not redefine team A');
 const saved=snapshotOf(tracker,'A1');assert.equal(saved.role,'goalkeeper');assert.equal(saved.team,'A');
 const o=log[20].r.observations.find(x=>x.playerId==='A1');
 assert.ok(o.pitch&&Math.abs(o.pitch.x-.08)<.01&&Math.abs(o.pitch.y-.5)<.01,JSON.stringify(o.pitch));
 for(const l of log)assert.ok(Object.entries(l.who).every(([k,id])=>id!=='A1'||k==='gk'),'A1 is never given to an outfield player');
});

test('assistant referee running the touchline is promoted only as REFEREE with the established referee kit',()=>{
 const tracker=new SoccerTracker(),o=defaultSoccerOptions;
 const log=run(tracker,50,(t,k)=>[...[...RED,...BLUE,REF].map((p,i)=>wobble(p,k,i)),k>=10&&{key:'ar',x:.1+.008*(k-10),y:.86,kit:'ref'}]);
 const first=log.find(l=>l.who.ar&&l.who.ar!=='-');
 assert.equal(first.who.ar,'R2');
 assert.ok(first.k>=10+o.boundaryHits-1,`promoted at step ${first.k}`);
 const d=log[log.length-1].r.debug.tracks.find(x=>x.playerId==='R2');
 assert.equal(d.zone,'boundary');assert.equal(d.roleLabel,'REFEREE');
 assert.ok(events(log,'role').some(e=>/touchline runner/.test(e.message)));
 assert.equal(log[log.length-1].who.ref,'R1');
});

test('a local-tracker swap between distinguishable teammates is corrected; no wrong identity reaches the data',()=>{
 const tracker=new SoccerTracker(),mates=[{key:'m1',x:.15,y:.75,kit:'red'},{key:'m2',x:.85,y:.75,kit:'red'},{key:'m3',x:.5,y:.8,kit:'red'},{key:'b1',x:.2,y:.35,kit:'blue'},{key:'b2',x:.4,y:.3,kit:'blue'},{key:'b3',x:.7,y:.3,kit:'blue'},{key:'b4',x:.9,y:.45,kit:'blue'}];
 // Same shirt (same team) but different shorts and socks. They meet and turn back; motion suggests a crossing.
 const log=run(tracker,30,(t,k)=>{const m=Math.max(0,k<=10?k:20-k),P={key:'P',x:.3+.02*m,y:.55,kit:'red'},Q={key:'Q',x:.7-.02*m,y:.57,kit:'redDark'};return [{...P,visible:iou(boxOf(P),boxOf(Q))<=.45},Q,...mates];});
 const p=log[6].who.P,q=log[6].who.Q;
 assert.match(p,/^A\d+$/);assert.match(q,/^A\d+$/);
 assert.equal(events(log,'swap-corrected').length,2);
 for(const l of log.slice(5)){assert.ok(l.who.P===p||l.who.P==='-'||l.who.P===undefined,`P ${l.t} ${l.who.P}`);assert.ok(l.who.Q===q||l.who.Q==='-',`Q ${l.t} ${l.who.Q}`);}
 assert.equal(log[log.length-1].who.P,p);assert.equal(log[log.length-1].who.Q,q);
 // Every observation of p (including held / predicted ones) sits on P's side of Q.
 for(const o of log.pool.filter(o=>o.playerId===p)){const k=Math.round(o.time/.2),m=Math.max(0,k<=10?k:20-k);assert.ok(Math.abs(o.box.x+o.box.w/2-(.3+.02*m))<.03,`p at ${o.time}`);}
});

test('a candidate that disappears before promotion is reported as REJECTED: LOW PLAYER CONFIDENCE',()=>{
 const tracker=new SoccerTracker();
 const log=run(tracker,16,(t,k)=>[...RED,...BLUE,REF,k>=3&&k<5&&{key:'blip',x:.8,y:.75,kit:'red'}]);
 const drops=log.flatMap(l=>l.r.debug.rejected.filter(r=>r.reason==='low-confidence'));
 assert.equal(drops.length,1);
 assert.ok(Number.isInteger(drops[0].track));assert.equal(drops[0].score,.85);
 assert.match(drops[0].detail,/before promotion/);
 assert.equal(rejectText[drops[0].reason],'REJECTED: LOW PLAYER CONFIDENCE');
 assert.ok(!log.pool.some(o=>iou(o.box,boxOf({x:.8,y:.75}))>.5));
});

test('team-change hysteresis: an identity is released only when >= 80% of the last 15 kit votes disagree',()=>{
 const m=new IdentityManager(defaultSoccerOptions,[]),model={a:[1,0],b:[0,1],keepers:[],anchored:false,spread:.1,samples:10};
 const vote=team=>({team,distA:team==='A'?.1:.9,distB:team==='A'?.9:.1,distRef:1,distKeeper:1,outlier:false,margin:.8,refLike:false,keeperLike:false,valid:true});
 const box={x:.4,y:.4,w:.03,h:.18},all=[];
 const step=(k,team)=>{const out=m.update({time:T(k),roster:roster(),samples:[{id:1,box,score:.9,evidence:'detection',zone:'inside',occluded:false,vote:vote(team),nearGoal:false,stab:{x:.415,y:.58,h:.18}}],ended:[],anchors:[],model,pitchReliable:true,filterEnabled:true,aspect:16/9,segment:0,drift:0,stabilize:b=>({x:b.x+b.w/2,y:b.y+b.h,h:b.h})});all.push(...out.events);return out;};
 for(let k=0;k<8;k++)step(k,'A');
 assert.equal(m.tracks.get(1).playerId,'A2');
 let changedAt;
 for(let k=8;k<30;k++){const out=step(k,'B');if(changedAt===undefined&&out.events.some(e=>e.kind==='team-change'))changedAt=k;}
 assert.equal(changedAt,19,'12 of the last 15 votes');
 assert.equal(m.registry.get('A2').status,'missing');
 assert.equal(m.tracks.get(1).playerId,'B2');
 assert.ok(all.some(e=>e.kind==='new-identity'&&e.playerId==='B2'));
});

test('diagnostics(): look-alike identities never seen together and too many identities per team',()=>{
 const m=new IdentityManager(defaultSoccerOptions,[]),model={a:[1,0],b:[0,1],keepers:[],anchored:false,spread:.1,samples:10};
 const zeros=n=>Array(n).fill(0),desc={jersey:[1,...zeros(23)],shorts:[1,...zeros(11)],socks:[1,...zeros(7)],layout:Array(24).fill(.5),quality:.9};
 const vote={team:'A',distA:.1,distB:.9,distRef:1,distKeeper:1,outlier:false,margin:.8,refLike:false,keeperLike:false,valid:true};
 const sample=(id,x)=>({id,box:{x,y:.4,w:.03,h:.18},score:.9,evidence:'detection',zone:'inside',occluded:false,descriptor:desc,vote,nearGoal:false,stab:{x:x+.015,y:.58,h:.18}});
 const ctx=(k,samples,ended=[])=>({time:T(k),roster:roster(),samples,ended,anchors:[],model,pitchReliable:true,filterEnabled:true,aspect:16/9,segment:0,drift:0,stabilize:b=>({x:b.x+b.w/2,y:b.y+b.h,h:b.h})});
 const evs=[];
 // Same kit, but the second person appears far away right after the first vanished: an implausible return,
 // so it becomes a new identity. Both were never visible together and look identical -> possible duplicate.
 for(let k=0;k<10;k++)evs.push(...m.update(ctx(k,[sample(1,.1)])).events);
 for(let k=10;k<22;k++)evs.push(...m.update(ctx(k,[sample(2,.85)],k===10?[{id:1,cut:false}]:[])).events);
 assert.deepEqual([...m.registry.keys()],['A2','A3']);
 assert.ok(evs.some(e=>e.kind==='reid-rejected'&&e.playerId==='A2'&&/spatially implausible/.test(e.message)));
 const diag=m.sanity(5);
 assert.ok(diag.some(e=>e.kind==='sanity'&&/A_Player_02 and TeamA_Player_03 were never visible together/.test(e.message)),diag.map(e=>e.message).join('\n'));
 // User labels can exceed the per-team reference: reported, never prevented.
 const tracker=new SoccerTracker({maxPerTeam:2});
 run(tracker,2,()=>RED.slice(0,3),{anchors:(t,k,seen)=>k===0?seen.map((p,i)=>({playerId:'A'+(i+2),box:boxOf(p)})):[]});
 assert.ok(tracker.diagnostics(1).some(e=>/Team A has 3 active identities \(expected at most 2\)/.test(e.message)));
});

test('no promotion while the pitch is unreliable (close-up / crowd shot)',()=>{
 const tracker=new SoccerTracker(),closeUp=makeFrame(W,H,(x,y)=>[120+(x*y)%30,110,100]);
 const log=run(tracker,12,()=>[...RED,...BLUE],{frame:()=>closeUp});
 for(const l of log){assert.equal(l.r.debug.pitch.reliable,false);assert.equal(l.r.observations.length,0);}
 assert.equal(events(log,'promotion').length,0);
});

test('explainFrame explains one frame without changing identities',()=>{
 const people=[...RED,...BLUE,REF],frame=paint([...people,...OUTSIDE]),dets=[...people,...OUTSIDE].map(detect);
 const cold=explainFrame(frame,dets,1,roster());
 assert.equal(cold.tracks.length,9);assert.equal(cold.rejected.length,OUTSIDE.length);
 assert.deepEqual([...new Set(cold.rejected.map(r=>r.reason))].sort(),['audience','outside-pitch']);
 assert.ok(cold.rejected.every(r=>r.detail.length>0));
 assert.ok(cold.tracks.every(t=>t.state==='candidate'&&/no team model/.test(t.reason)));
 const {tracker,log}=getA(),before=JSON.stringify(tracker.snapshot());
 const warm=explainFrame(frame,dets,log[log.length-1].t,roster(),{},tracker);
 assert.equal(JSON.stringify(tracker.snapshot()),before);
 const ref=warm.tracks.find(t=>iou(t.box,boxOf(REF))>.6);assert.equal(ref.label,'REF-1');
 const red=warm.tracks.find(t=>iou(t.box,boxOf(RED[0]))>.6);assert.match(red.label,/^A-\d\d$/);assert.match(red.reason,/kit vote A/);
 const loose=explainFrame(frame,dets,1,roster(),{field:{...defaultSoccerOptions.field,enabled:false}});
 assert.equal(loose.rejected.length,0);
});

test('re-ID score combination: weights, unknown components and the team gate',()=>{
 assert.equal(Object.values(REID_WEIGHTS).reduce((a,b)=>a+b,0).toFixed(6),'1.000000');
 assert.equal(combineReid({appearance:.9,uniform:.9,spatial:.9,movement:.9,temporal:.9,team:true}).toFixed(6),'0.900000');
 assert.equal(combineReid({appearance:1,uniform:1,spatial:1,movement:1,temporal:1,team:false}),0);
 // Unknown spatial (camera cut): its weight is removed, not counted as zero.
 assert.equal(combineReid({appearance:.8,uniform:.8,temporal:.8,team:true}).toFixed(6),'0.800000');
});

test('deterministic: the same input produces the same identities and events',()=>{
 const go=()=>{const tr=new SoccerTracker();const log=run(tr,16,(t,k)=>[...RED,...BLUE,REF].map((p,i)=>wobble(p,k,i)));return JSON.stringify(log.map(l=>[l.r.observations,l.r.events,l.r.debug.tracks]));};
 assert.equal(go(),go());
});

test('step stays fast: identity work well under 20 ms per step besides pitch analysis and descriptors',()=>{
 const tracker=new SoccerTracker(),many=Array.from({length:22},(_,i)=>({key:'p'+i,x:.05+.09*(i%11),y:i<11?.4:.7,kit:i%2?'red':'blue'}));
 const costs=[];
 run(tracker,25,(t,k)=>[...many,REF].map((p,i)=>wobble(p,k,i)),{each:l=>{if(l.k>=3){const c=tracker.lastTiming;costs.push(c.total-c.pitch-c.describe);}}});
 costs.sort((a,b)=>a-b);
 assert.ok(costs[Math.floor(costs.length/2)]<20,`median ${costs[Math.floor(costs.length/2)]} ms`);
});

test('LocalTracker replay over a real professional match: unique local ids per frame, confident spawns only',()=>{
 const fixture=JSON.parse(fs.readFileSync(__dirname+'/fixtures/pro-match-detections.json','utf8'));
 const lt=new LocalTracker(),lifetimes=new Map(),ended=new Set();let spawned=0;
 for(const f of fixture.frames){
  const dets=f.detections.filter(d=>d.pitch&&d.kind==='person');
  const r=lt.step(dets,f.time,f.camera);
  const ids=r.samples.map(s=>s.id);
  assert.equal(new Set(ids).size,ids.length,`duplicate local id at ${f.time}`);
  assert.ok(ids.every(id=>Number.isInteger(id)&&id>0));
  for(const s of r.samples){if(s.detection)assert.ok(dets.includes(s.detection));if(s.spawned){spawned++;assert.ok(s.detection.score>=.35);}lifetimes.set(s.id,(lifetimes.get(s.id)||0)+1);}
  const boxes=r.samples.filter(s=>s.detection).map(s=>s.detection);assert.equal(new Set(boxes).size,boxes.length,'a detection feeds one track');
  for(const e of r.ended){assert.ok(!ended.has(e.id),'ended once');ended.add(e.id);assert.ok(!ids.includes(e.id));}
  assert.equal(lt.tracks.length,new Set(lt.tracks.map(t=>t.id)).size);
 }
 assert.ok(spawned>10);
 assert.ok([...lifetimes.values()].filter(n=>n>=fixture.frames.length-2).length>=10,'many tracks persist through the clip');
 assert.equal(localId(localKey(42)),42);
 // A camera cut ends every track.
 const r=lt.step([],99,{...still,cut:true});assert.equal(lt.tracks.length,0);assert.ok(r.ended.length>0&&r.ended.every(e=>e.cut));
});

// ---------- review regressions ----------
test('existing identities orient the kits: a restored session never renames or duplicates them, whatever the detection order',()=>{
 const {describe}=require('../lib/soccer/appearance.ts'),{kitVote}=require('../lib/soccer/classify.ts');
 const first=new SoccerTracker();
 run(first,20,(t,k)=>[...RED,...BLUE].map((p,i)=>wobble(p,k,i)));
 const snap=JSON.parse(JSON.stringify(first.snapshot()));
 assert.ok(snap.every(s=>s.team==='A'?/^A/.test(s.playerId):/^B/.test(s.playerId)));
 // Next session, long after (no spatial evidence): blue players are listed first, which alone would make blue team A.
 const next=new SoccerTracker({},snap),log=run(next,30,(t,k)=>[...BLUE,...RED].map((p,i)=>wobble(p,k,i)),{k0:300});
 const redKit=kitVote(next.teamModel,describe(paint([RED[0]]),boxOf(RED[0]))).team;
 assert.equal(redKit,'A','red (team A in the first session, listed first there) is still team A');
 assert.equal(events(log,'new-identity').length,0);
 assert.equal(events(log,'team-change').length,0);
 assert.deepEqual(next.snapshot().map(s=>s.playerId+s.team).sort(),snap.map(s=>s.playerId+s.team).sort());
 for(const l of log){for(const p of RED)assert.ok(l.who[p.key]==='-'||/^A/.test(l.who[p.key]));for(const p of BLUE)assert.ok(l.who[p.key]==='-'||/^B/.test(l.who[p.key]));}
});

test('a late label orienting the kits the other way moves automatic identities to the other team instead of duplicating everyone',()=>{
 const tracker=new SoccerTracker();
 const log=run(tracker,50,(t,k)=>[...BLUE,...RED].map((p,i)=>wobble(p,k,i)),{anchors:(t,k,seen)=>k===20?[{playerId:'A7',box:boxOf(seen.find(p=>p.key==='r1'))}]:[]});
 const before=log[19].who,end=log[49].who;
 for(const p of BLUE)assert.match(before[p.key],/^A\d+$/,'unlabelled, blue listed first became team A');
 assert.equal(end.r1,'A7');
 for(const p of BLUE)assert.equal(end[p.key],before[p.key].replace('A','B'),'same number, other team');
 for(const p of RED.slice(1))assert.equal(end[p.key],before[p.key].replace('B','A'));
 assert.equal(events(log,'new-identity').filter(e=>e.time>1).length,0);
 assert.ok(events(log,'team-change').every(e=>/oriented the kits/.test(e.message)),'nobody released by the kit rule');
 assert.equal(new Set(Object.values(end)).size,8);
 const people=tracker.snapshot().filter(s=>s.status!=='unknown');
 assert.equal(people.length,8,'one identity per person');
 // r1's own automatic identity (created on that track) is kept but out of re-ID competition.
 const auto=before.r1.replace('B','A');assert.equal(snapshotOf(tracker,auto).status,'unknown');
});

test('a crop hidden by an undetected opponent gives the released identity back to the same track (no duplicate)',()=>{
 const tracker=new SoccerTracker();
 const log=run(tracker,50,(t,k)=>[...[...RED,...BLUE].map((p,i)=>({...p,x:p.x+.004*Math.sin(k*.7+i)})),k>=25&&k<30&&{key:'occ',x:.308,y:.615,kit:'blue',visible:false}]);
 const id=log[24].who.r2;assert.match(id,/^A\d+$/);
 assert.ok(events(log,'reid-rejected').some(e=>e.playerId===id&&/no longer matches/.test(e.message)),'released while hidden');
 assert.ok(events(log,'reid').some(e=>e.playerId===id&&e.time<7),'won back once the crop is clean');
 assert.ok(log.slice(36).every(l=>l.who.r2===id));
 assert.equal(events(log,'new-identity').filter(e=>e.time>1).length,0);
 assert.equal(tracker.snapshot().filter(s=>s.team==='A').length,4);
});

test('an identity without gallery or position still competes: the others are not chosen by elimination',()=>{
 const tracker=new SoccerTracker({maxPerTeam:3}),ids={};
 const P1={key:'P1',x:.3,y:.6,kit:'red'},P2={key:'P2',x:.6,y:.32,kit:'red',h:.075},P3={key:'P3',x:.75,y:.6,kit:'red'};
 const blue=[{key:'b1',x:.2,y:.45,kit:'blue'},{key:'b2',x:.45,y:.4,kit:'blue'},{key:'b3',x:.85,y:.4,kit:'blue'}];
 const log=run(tracker,45,(t,k)=>(k<15?[P1,P2,P3,...blue]:[{key:'Q',x:.5,y:.6,kit:'red'},{...P3,x:.2}]).map((p,i)=>({...p,x:p.x+.004*Math.sin(k*.7+i)})),
  {camera:(t,k)=>({...still,cut:k===15}),each:l=>{if(l.k===14)Object.assign(ids,l.who);},anchors:(t,k,seen)=>k===16?[{playerId:ids.P3,box:boxOf(seen.find(p=>p.key==='P3'))}]:[]});
 assert.equal(snapshotOf(tracker,ids.P2).gallery.length,0,'the far player never got a gallery sample');
 const after=log.slice(16);
 assert.ok(after.every(l=>l.who.Q==='-'),'Q (really P2) is not given P1 by elimination');
 assert.ok(events(after,'deferred').some(e=>/cannot be compared/.test(e.message)));
 assert.equal(events(after,'reid').filter(e=>e.playerId===ids.P1).length,0);
});

test('a labelled goalkeeper is re-identified after a camera cut (a team has no second keeper)',()=>{
 const tracker=new SoccerTracker(),GK={key:'gk',x:.08,y:.55,kit:'keeper'};
 const ros=()=>roster().map(s=>s.id==='A1'?{...s,role:'goalkeeper'}:s);
 const log=run(tracker,60,(t,k)=>[...RED,...BLUE,GK].map((p,i)=>wobble(k>=20?{...p,x:1-p.x}:p,k,i)),{roster:ros,camera:(t,k)=>({...still,cut:k===20}),anchors:(t,k,seen)=>k===2?[{playerId:'A1',box:boxOf(seen.find(p=>p.key==='gk'))}]:[]});
 assert.equal(log[19].who.gk,'A1');
 const reid=events(log.slice(20),'reid').find(e=>e.playerId==='A1');
 assert.ok(reid,'reconnected');assert.match(reid.message,/no spatial evidence/);
 assert.ok(log.slice(40).every(l=>l.who.gk==='A1'));
 assert.equal(events(log,'new-identity').filter(e=>e.playerId?.startsWith('A')&&e.time>4).length,0);
});

test('people outside both kits become referees only when it is unambiguous; a shaded player stays in his team',()=>{
 // An unlabelled goalkeeper (no goal line in view) and the referee: two different non-team kits, neither is guessed.
 const tracker=new SoccerTracker(),GK={key:'gk',x:.08,y:.55,kit:'keeper'};
 const log=run(tracker,30,(t,k)=>[...RED,...BLUE,GK,REF].map((p,i)=>wobble(p,k,i)));
 assert.equal(log[29].who.gk,'-');assert.equal(log[29].who.ref,'-');
 assert.ok(!tracker.snapshot().some(s=>s.role==='referee'));
 assert.match(log[29].r.debug.tracks.find(d=>iou(d.box,boxOf(REF))>.5).reason,/label the referee/);
 // A red player walking into deep shade (same hue, 35% darker) is not a referee and does not move the red prototype.
 KITS.redShade={shirt:KITS.red.shirt.map(v=>Math.round(v*.65)),shorts:KITS.red.shorts.map(v=>Math.round(v*.65)),socks:KITS.red.socks.map(v=>Math.round(v*.65))};
 const shade=new SoccerTracker(),log2=run(shade,50,(t,k)=>[...RED,...BLUE].map((p,i)=>wobble(p.key==='r2'&&k>=30?{...p,kit:'redShade'}:p,k,i)));
 assert.equal(events(log2,'role').length,0);
 assert.ok(!shade.snapshot().some(s=>s.role==='referee'));
 assert.ok(log2.slice(36).every(l=>l.who.r2==='-'||/^A/.test(l.who.r2)));
 const d=log2[49].r.debug.tracks.find(x=>iou(x.box,boxOf(RED[1]))>.5);assert.equal(d.team,'A');
});

test('a loose or click-sized label binds the person under it at once (no second track)',()=>{
 const tracker=new SoccerTracker();
 const loose=b=>{const w=b.w*1.6,h=b.h*1.25;return {x:b.x+b.w/2-w/2,y:b.y+b.h-h,w,h};},click=b=>({x:b.x+b.w/2-.0075,y:b.y+b.h*.35-.019,w:.015,h:.038});
 const log=run(tracker,20,(t,k)=>[...RED,...BLUE].map((p,i)=>wobble(p,k,i)),{anchors:(t,k,seen)=>{
  if(k===10)return [{playerId:'A9',box:loose(boxOf(seen.find(p=>p.key==='r2')))}];
  if(k===12)return [{playerId:'A10',box:click(boxOf(seen.find(p=>p.key==='r3')))}];
  return [];}});
 assert.ok(log.slice(10).every(l=>l.who.r2==='A9'));
 assert.ok(log.slice(12).every(l=>l.who.r3==='A10'));
 for(const l of log.slice(10)){const n=l.r.debug.tracks.filter(d=>iou(d.box,boxOf(RED[1]))>.3).length;assert.equal(n,1,`tracks on r2 at ${l.t}`);}
 assert.equal(observed(log[12].r.observations,boxOf(RED[2]),log[12].t).conf,1);
});

test('a label drawn between detection steps binds the moving player, never a box left behind on empty grass',()=>{
 const runner=t=>({key:'run',x:.2+.3*t,y:.6,kit:'red'}),others=[...RED.slice(0,1),...RED.slice(2),...BLUE];
 const tracker=new SoccerTracker();
 const log=run(tracker,14,t=>[runner(t),...others],{anchors:(t,k)=>k===6?[{playerId:'A7',box:boxOf(runner(1.05)),time:1.05}]:[]});
 for(const l of log.slice(6,9))assert.equal(l.who.run,'A7',`runner at ${l.t}`);
 const a7=log.pool.filter(o=>o.playerId==='A7');
 assert.ok(a7.length>=3);
 for(const o of a7)assert.ok(iou(o.box,boxOf(runner(o.time)))>.5,`A7 at ${o.time} is on the runner`);
});

test('a label replacing an automatic identity created on that track: the player is re-identified as the label after leaving',()=>{
 const tracker=new SoccerTracker();
 const log=run(tracker,60,(t,k)=>[...RED,...BLUE].map((p,i)=>wobble(p,k,i)).filter(p=>!(p.key==='r2'&&k>=30&&k<40)),{anchors:(t,k,seen)=>k===12?[{playerId:'A9',box:boxOf(seen.find(p=>p.key==='r2'))}]:[]});
 const auto=log[11].who.r2;assert.match(auto,/^A\d+$/);assert.notEqual(auto,'A9');
 assert.ok(events(log,'reid').some(e=>e.playerId==='A9'&&e.time>=8));
 assert.ok(log.slice(46).every(l=>l.who.r2==='A9'));
 const kept=snapshotOf(tracker,auto);assert.ok(kept,'kept, not deleted');assert.equal(kept.status,'unknown');
});

test('re-running from an earlier frame emits the back-filled observations of that pass',()=>{
 const scene=(t,k)=>[...RED,...BLUE,REF].map((p,i)=>wobble(p,k,i));
 const first=new SoccerTracker();run(first,40,scene);
 const again=new SoccerTracker({},JSON.parse(JSON.stringify(first.snapshot()))),log=run(again,20,scene);
 const r1=log.pool.filter(o=>o.playerId==='R1');
 assert.ok(r1.some(o=>o.evidence==='reidentified'));
 assert.equal(Math.min(...r1.map(o=>o.time)),0);
});

test('the team model accumulates a kit seen only a few players at a time',()=>{
 const reds=Array.from({length:9},(_,i)=>({key:'r'+i,x:.06+.1*i,y:[.35,.5,.65,.78][i%4],kit:'red'}));
 const blues=Array.from({length:6},(_,i)=>({key:'b'+i,x:.15+.3*(i%3)+.05*Math.floor(i/3),y:[.45,.68][Math.floor(i/3)],kit:'blue'}));
 // Three blue players for 1.6 s, none for 2 s, then three others: never more than three in one frame.
 const tracker=new SoccerTracker();
 const log=run(tracker,30,(t,k)=>[...reds,...blues.filter((_,i)=>i<3?k<8:k>=18&&k<26),{key:'ref',x:.5,y:.3,kit:'ref'}].map((p,i)=>({...p,x:p.x+.004*Math.sin(k*.7+i)})));
 assert.ok(tracker.teamModel.a&&tracker.teamModel.b);
 for(const k of ['b3','b4','b5'])assert.match(log[25].who[k],/^B\d+$/);
 for(const p of reds)assert.match(log[25].who[p.key],/^A\d+$/);
});

test('the size model is dropped at a camera cut: a closer view keeps its players',()=>{
 const tracker=new SoccerTracker(),wide=Array.from({length:10},(_,i)=>({key:'p'+i,x:.08+i*.09,y:.3+(i%5)*.11,kit:i%2?'red':'blue'}));
 const close=[{x:.3,y:.6,h:.45},{x:.5,y:.7,h:.5},{x:.7,y:.65,h:.47},{x:.45,y:.45,h:.38}].map((p,i)=>({key:'c'+i,...p,kit:i%2?'red':'blue'}));
 const log=run(tracker,8,(t,k)=>k<3?wide:close,{camera:(t,k)=>({...still,cut:k===3})});
 assert.ok(log[2].r.debug.rejected.every(r=>r.reason!=='implausible-size'));
 for(const l of log.slice(3))assert.deepEqual(l.r.debug.rejected.filter(r=>r.reason==='implausible-size'),[],`at ${l.t}`);
 assert.equal(log[7].r.debug.tracks.length,4);
});

test('explainFrame shows identities only for the tracker\'s own frame',()=>{
 const {tracker,log}=getA(),people=[...RED,...BLUE,REF],frame=paint(people),dets=people.map(detect);
 const other=explainFrame(frame,dets,0,roster(),{},tracker);
 assert.ok(other.tracks.every(t=>!t.playerId&&t.state==='candidate'),'no identity from the end of the run on another frame');
 assert.match(other.tracks.find(t=>iou(t.box,boxOf(RED[0]))>.6).reason,/kit vote A/);
 assert.equal(explainFrame(frame,dets,log[log.length-1].t,roster(),{},tracker).tracks.find(t=>iou(t.box,boxOf(REF))>.6).label,'REF-1');
});
