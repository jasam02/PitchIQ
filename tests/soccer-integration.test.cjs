// End-to-end seam between the soccer tracker and the saved document, mirroring the workspace's runSoccer loop:
// roster/anchors from the document -> SoccerTracker.step -> soccerPoints / appendTrackingLog / cleanIdentities ->
// trackingSchema (what the server stores) -> identities restored into a new tracker.
const test=require('node:test');
const assert=require('node:assert/strict');
require('./helpers/ts-require.cjs');
const {makeScene,FAR,NEAR}=require('./helpers/soccer-scene.cjs');
const {SoccerTracker,composeCamera}=require('../lib/soccer/pipeline.ts');
const {iou}=require('../lib/detection-core.ts');
const tracking=require('../lib/tracking.ts');

const scene=makeScene(640,360);
const still={scale:1,dx:0,dy:0,reliable:true,cut:false};
const T=k=>Number((k*.2).toFixed(4));
const RED=[[.15,.45],[.3,.62],[.45,.76],[.66,.4]].map(([x,y],i)=>({key:'r'+(i+1),x,y,kit:'red'}));
const BLUE=[[.22,.72],[.4,.35],[.74,.66],[.86,.3]].map(([x,y],i)=>({key:'b'+(i+1),x,y,kit:'blue'}));
const FANS=[.1,.3,.5,.7,.9].map((x,i)=>({key:'fan'+i,x,y:.11,kit:'fan',h:.07,score:.6}));
const BENCH=[.3,.36].map((x,i)=>({key:'bench'+i,x,y:.97,kit:'coat',score:.7}));
// r4 runs out of the right side of the view (steps 12-21 off screen) and comes back the same way.
function people(k){
 const wobble=i=>.01*Math.sin(k*.7+i);
 const out=[...RED.slice(0,3),...BLUE].map((p,i)=>({...p,x:p.x+wobble(i),y:p.y+wobble(i+3)/2}));
 const r4=RED[3],x=k<6?r4.x:k<12?r4.x+(k-5)*.055:k<22?undefined:Math.max(r4.x,1.02-(k-21)*.05);
 if(x!==undefined)out.push({...r4,x});
 out.push({key:'ref',x:.55+.1*Math.sin(k*.15),y:.55,kit:'ref'});
 return [...out,...FANS,...BENCH];
}
const SUB_TIME=4; // B4 is substituted off; the incoming B12 is not on the pitch yet

// suggestions: the start frame also takes roster suggestions as weak labels (as the workspace does).
function session(doc,steps,tracker,suggestions=false){
 const base=tracking.ensureOfficials(doc),keys=new Set(),additions=[],issues=[],ids=new Set(base.players.map(p=>p.id)),byKey=[];
 let log=[],last=-1;
 for(let k=0;k<steps;k++){
  const t=T(k),scenePeople=people(k).filter(p=>p.x+.02<1.04),detections=scenePeople.map(scene.detect);
  detections.push({id:'ball',kind:'ball',score:.7,box:{x:.5,y:.6,w:.008,h:.014}});
  const anchors=k?tracking.confirmedAnchors(base,last,t):tracking.confirmedAnchors(base,t-.125,t+.001,suggestions);last=t;
  const r=tracker.step({time:t,frame:scene.paint(scenePeople),detections,camera:still,anchors,roster:tracking.soccerRoster(base,t)});
  log=tracking.appendTrackingLog(log,r.events);
  for(const i of r.issues)if(ids.has(i.playerId))issues.push({playerId:i.playerId,time:i.time,reason:i.reason.slice(0,160)});
  additions.push(...tracking.soccerPoints(base,r.observations,keys));
  byKey.push({t,people:scenePeople});
 }
 const document={...base,points:[...base.points,...additions],issues:issues.slice(-500),identities:tracking.cleanIdentities(base,tracker.snapshot()),trackingLog:log,checkpoint:{time:T(steps-1),start:0}};
 return {document,byKey,log};
}
// Person -> playerIds of the points that sit on that person's box at that time.
function owners(document,byKey){
 const map=new Map();
 for(const {t,people} of byKey)for(const p of people){
  const b=scene.boxOf(p);
  for(const q of document.points)if(q.playerId!=='ball'&&Math.abs(q.time-t)<1e-6&&iou(q.box,b)>.5){if(!map.has(p.key))map.set(p.key,new Set());map.get(p.key).add(q.playerId);}
 }
 return map;
}

test('composeCamera chains motions, keeps cuts and never trusts an unknown half',()=>{
 const a={scale:1.02,dx:.01,dy:-.005,reliable:true,cut:false},b={scale:.99,dx:-.02,dy:.004,reliable:true,cut:false};
 const c=composeCamera(a,b),p={x:.4,y:.3},via={x:(p.x*a.scale+a.dx)*b.scale+b.dx,y:(p.y*a.scale+a.dy)*b.scale+b.dy};
 assert.ok(c.reliable&&!c.cut);
 assert.ok(Math.abs(p.x*c.scale+c.dx-via.x)<1e-12&&Math.abs(p.y*c.scale+c.dy-via.y)<1e-12);
 const unknown=composeCamera(a,{scale:1,dx:0,dy:0,reliable:false,cut:false});
 assert.deepEqual(unknown,{scale:1,dx:0,dy:0,reliable:false,cut:false});
 assert.equal(composeCamera({...a,cut:true},b).cut,true);
 assert.equal(composeCamera(a,{...b,cut:true}).reliable,false);
});

test('a soccer session produces a document the server accepts, and its identities restore',()=>{
 let doc=tracking.initialTracking('Reds','Blues');
 // One user label (A1 on r1 at the start), a reviewed observation that must survive, and a substitution.
 doc.points.push({playerId:'A1',time:0,box:scene.boxOf(RED[0]),score:1,source:'manual',reviewed:true});
 doc.points.push({playerId:'A1',time:T(30),box:{x:.1,y:.3,w:.02,h:.05},score:.9,source:'experimental',reviewed:true});
 doc.players.push({id:'B12',name:'Substitute',number:'12',team:'B',starter:false});
 doc.substitutions.push({id:'s1',time:SUB_TIME,off:'B4',on:'B12'});
 doc=tracking.trackingSchema.parse(doc);
 const tracker=new SoccerTracker({},doc.identities||[]);
 const {document,byKey,log}=session(doc,40,tracker);

 const parsed=tracking.trackingSchema.safeParse(document);
 assert.ok(parsed.success,parsed.success?'':JSON.stringify(parsed.error.issues.slice(0,3)));
 const body=JSON.stringify({videoId:'00000000-0000-4000-8000-000000000000',revision:3,document:parsed.data});
 assert.ok(body.length<900000,`body ${body.length} chars`);
 assert.ok(document.players.some(p=>p.id==='R1'&&p.team==='ref'),'officials added to the roster');

 // Only people on the pitch become points: no fan in the stands, nobody on the bench.
 const pts=document.points.filter(p=>p.source==='experimental'&&!p.reviewed);
 assert.ok(pts.length>100,`${pts.length} automatic points`);
 for(const p of pts){const foot=p.box.y+p.box.h;assert.ok(foot>FAR-.03&&foot<NEAR+.04,`point for ${p.playerId} at foot ${foot.toFixed(3)}`);}
 assert.ok(pts.every(p=>Number.isInteger(p.track)&&p.conf!==undefined),'track and confidence stored');
 // Substitution: nothing for B4 after it left, nothing for B12 at all (never on the pitch).
 assert.ok(!pts.some(p=>p.playerId==='B4'&&p.time>SUB_TIME+.001),'no B4 point after the substitution');
 assert.ok(!pts.some(p=>p.playerId==='B12'),'the bench substitute never gets a point');
 // One point per player and time; two players never share one box (no duplicate identity on one person).
 const seen=new Set();
 for(const p of document.points){const k=p.playerId+':'+Math.round(p.time*100);assert.ok(!seen.has(k),'duplicate '+k);seen.add(k);}
 for(const a of pts)for(const b of pts)if(a!==b&&a.playerId!==b.playerId&&Math.abs(a.time-b.time)<1e-6)assert.ok(iou(a.box,b.box)<.5,`${a.playerId} and ${b.playerId} share a box at ${a.time}`);
 // The user label wins: r1 is A1; every person keeps one identity; kits map to their teams; the referee is R1.
 const who=owners(document,byKey);
 assert.deepEqual([...(who.get('r1')||[])],['A1']);
 for(const [key,set] of who){
  assert.equal(set.size,1,`${key} was ${[...set].join(', ')}`);
  const id=[...set][0];
  if(/^r\d/.test(key))assert.match(id,/^A\d+$/);
  if(/^b\d/.test(key))assert.match(id,/^B\d+$/);
  if(key==='ref')assert.equal(id,'R1');
 }
 for(const key of ['fan0','fan1','fan2','fan3','fan4','bench0','bench1'])assert.ok(!who.has(key),key+' tracked');
 // r4 left the view and came back on a new local track: reconnected to its identity, never a second one.
 const r4=[...who.get('r4')][0];
 assert.ok(log.some(e=>e.kind==='reid'&&e.playerId===r4&&e.scores&&e.scores.missingSeconds>1),'re-identified after leaving the view');
 // The reviewed observation is preserved and no automatic point replaces it.
 assert.ok(document.points.some(p=>p.playerId==='A1'&&p.time===T(30)&&p.reviewed&&p.box.x===.1));
 assert.ok(!document.points.some(p=>p.playerId==='A1'&&Math.abs(p.time-T(30))<.05&&!p.reviewed));
 assert.ok(log.length>0&&log.length<=300&&log.some(e=>e.kind==='new-identity'));

 // Saved identities restore into a fresh tracker (next pass) and snapshot back to the same roster ids.
 const ids=parsed.data.identities.map(i=>i.playerId).sort();
 assert.ok(ids.includes('A1')&&ids.includes('R1'));
 const again=new SoccerTracker({},parsed.data.identities);
 assert.deepEqual(again.snapshot().map(i=>i.playerId).sort(),ids);
 assert.ok(again.snapshot().every(i=>i.status!=='active'),'restored identities wait for re-identification');
 assert.ok(tracking.trackingSchema.safeParse({...parsed.data,identities:tracking.cleanIdentities(parsed.data,again.snapshot())}).success);
});

test('kit-colour roster suggestions on the start frame seed the pass; a reviewed one is never doubled by an automatic identity',()=>{
 for(const reviewed of [false,true]){
  let doc=tracking.initialTracking('Reds','Blues');
  doc.points.push({playerId:'A7',time:0,box:scene.boxOf(RED[1]),score:.8,source:'experimental',reviewed,evidence:'reidentified'});
  doc=tracking.trackingSchema.parse(doc);
  const {document,byKey}=session(doc,20,new SoccerTracker(),true),who=owners(document,byKey);
  assert.deepEqual([...who.get('r2')],['A7'],`reviewed ${reviewed}`);
  for(const [key,set] of who)if(key!=='r2')assert.ok(!set.has('A7'),key);
 }
});
