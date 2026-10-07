const {test}=require('node:test');
const assert=require('node:assert/strict');
require('./helpers/ts-require.cjs');
const tracking=require('../lib/tracking.ts');
const native=require('../lib/native-tracking.ts');
const box={x:.3,y:.3,w:.05,h:.1};
const clone=v=>JSON.parse(JSON.stringify(v));
// Exact shape of a match saved before soccer identity tracking existed.
function legacyDoc(){
 const d=tracking.initialTracking('Home','Away');
 d.players.push({id:'sub-1',name:'Sub',number:'12',team:'A',starter:false});
 d.substitutions.push({id:'s1',time:5,off:'A2',on:'sub-1'});
 d.points=[{playerId:'A1',time:1,box,score:1,source:'manual',reviewed:true},{playerId:'A1',time:1.2,box,score:.8,source:'experimental',reviewed:false,evidence:'predicted'},{playerId:'ball',time:1,box:{x:.5,y:.5,w:.01,h:.01},score:.9,source:'experimental',reviewed:false,evidence:'appearance'}];
 d.issues=[{playerId:'A1',time:1.2,reason:'Searching for this player.'}];
 d.playerMemory=[{playerId:'A1',box,appearance:[1,0],time:1,lastSeen:1.2,edge:'',offscreen:false}];
 d.fieldMarks=[{id:'m',name:'far-left',time:1,points:[{x:.1,y:.2}]}];
 d.fieldCamera=[{time:1,segment:'one',scale:1,dx:0,dy:0}];
 d.checkpoint={start:0,time:2};
 return clone(d);
}
function identity(playerId,extra={}){return {playerId,role:'player',team:'A',status:'missing',gallery:[Array.from({length:80},(_,i)=>i%7?0:.25)],lastSeen:3.2,lastBox:box,exitEdge:'left',velocity:{x:.01,y:0},identityConfidence:.93,teamConfidence:.9,roleConfidence:.8,anchored:false,...extra};}
const reid={time:14.4,kind:'reid',track:91,playerId:'A7',message:'Reconnected after 11.2 s',scores:{appearance:.94,jersey:.92,uniform:.9,team:true,spatial:.87,movement:.7,temporal:.6,final:.93,secondBest:.41,missingSeconds:11.2}};

test('legacy saved documents parse unchanged (version 1, no new fields required)',()=>{
 const d=legacyDoc(),parsed=tracking.trackingSchema.parse(d);
 assert.deepEqual(parsed,d);assert.equal(parsed.version,1);
 assert.equal(parsed.identities,undefined);assert.equal(parsed.trackingLog,undefined);
 assert.ok(parsed.players.every(p=>p.role===undefined));
 assert.equal(tracking.trackingSchema.safeParse({...d,version:2}).success,false);
});

test('new optional fields parse: roles, officials, point track/conf/pitch, identities and tracking log',()=>{
 const d=tracking.ensureOfficials(legacyDoc());
 d.players=d.players.map(p=>p.id==='A1'?{...p,role:'goalkeeper'}:p.id==='A3'?{...p,role:'player'}:p);
 d.points.push({playerId:'A7',time:2,box,score:.7,source:'experimental',reviewed:false,evidence:'reidentified',track:91,conf:.93,pitch:{x:.42,y:-.05}},{playerId:'R1',time:2,box,score:.8,source:'experimental',reviewed:false,evidence:'detection',track:12,conf:1});
 d.identities=[identity('A7'),identity('R1',{role:'referee',team:undefined,status:'active'}),identity('A1',{role:'goalkeeper',lastPitch:{x:.02,y:.5},anchored:true,identityConfidence:1})];
 d.trackingLog=[reid,{time:15,kind:'sanity',message:'Team A has 12 identities.'},{time:15.2,kind:'promotion',track:7,message:'CANDIDATE -> PLAYER_TEAM_B'}];
 const parsed=tracking.trackingSchema.parse(clone(d));
 assert.equal(parsed.points.find(p=>p.playerId==='A7').track,91);assert.equal(parsed.points.find(p=>p.playerId==='A7').pitch.y,-.05);
 assert.equal(parsed.identities.length,3);assert.equal(parsed.trackingLog[0].scores.missingSeconds,11.2);
 assert.equal(parsed.players.find(p=>p.id==='A1').role,'goalkeeper');
 for(const bad of [{conf:1.2},{track:-1},{track:1.5},{pitch:{x:2,y:.5}}])assert.equal(tracking.trackingSchema.safeParse({...d,points:[{...d.points.at(-1),...bad}]}).success,false,JSON.stringify(bad));
});

test('officials: at most four, not counted as starters, referee role only for officials',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking());
 d.players.push({id:'R4',name:'Fourth official',number:'',team:'ref',starter:true,role:'referee'});
 assert.equal(d.players.filter(p=>p.team==='A'&&p.starter).length,11);
 assert.ok(tracking.trackingSchema.safeParse(d).success);
 const five={...d,players:[...d.players,{id:'R5',name:'VAR',number:'',team:'ref',starter:true}]};
 assert.match(tracking.trackingSchema.safeParse(five).error.issues[0].message,/4 match officials/);
 const role=(id,r)=>({...d,players:d.players.map(p=>p.id===id?{...p,role:r}:p)});
 assert.equal(tracking.trackingSchema.safeParse(role('A4','referee')).success,false);
 assert.equal(tracking.trackingSchema.safeParse(role('R2','goalkeeper')).success,false);
 assert.equal(tracking.trackingSchema.safeParse(role('ball','player')).success,false);
 assert.ok(tracking.trackingSchema.safeParse(role('R2',undefined)).success);
 assert.equal(tracking.trackingSchema.safeParse({...d,substitutions:[{id:'x',time:3,off:'R1',on:'A1'}]}).success,false);
});

test('ensureOfficials adds R1-R3 once, persists through the schema and respects roster limits',()=>{
 const base=tracking.initialTracking(),once=tracking.ensureOfficials(base);
 assert.deepEqual(once.players.filter(p=>p.team==='ref').map(p=>[p.id,p.name,p.role,p.starter]),[['R1','Referee','referee',true],['R2','Assistant referee 1','referee',true],['R3','Assistant referee 2','referee',true]]);
 assert.equal(base.players.length,23);
 assert.equal(tracking.ensureOfficials(once),once);
 assert.deepEqual(tracking.ensureOfficials(tracking.trackingSchema.parse(clone(once))).players,once.players);
 const partial={...base,players:[...base.players,{id:'R2',name:'Linesman',number:'',team:'ref',starter:true,role:'referee'}]};
 const filled=tracking.ensureOfficials(partial);assert.deepEqual(filled.players.filter(p=>p.team==='ref').map(p=>p.id+':'+p.name),['R2:Linesman','R1:Referee','R3:Assistant referee 2']);
 const crowded={...base,players:[...base.players,...Array.from({length:37},(_,i)=>({id:'s'+i,name:'Sub',number:'',team:i%2?'A':'B',starter:false}))]};
 assert.equal(tracking.ensureOfficials(crowded).players.length,61);assert.ok(tracking.trackingSchema.safeParse(tracking.ensureOfficials(crowded)).success);
 const fourRefs={...base,players:[...base.players,...['X1','X2','X3','X4'].map(id=>({id,name:'Official',number:'',team:'ref',starter:true}))]};
 assert.equal(tracking.ensureOfficials(fourRefs),fourRefs);
});

test('points for officials are valid labels and follow official substitutions',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking());
 d.players.push({id:'R4',name:'Reserve official',number:'',team:'ref',starter:false,role:'referee'});
 d.substitutions.push({id:'s',time:10,off:'R2',on:'R4'});
 d.points=[{playerId:'R1',time:1,box,score:1,source:'manual',reviewed:true},{playerId:'R2',time:2,box,score:.8,source:'experimental',reviewed:false,evidence:'detection',track:4,conf:.9},{playerId:'R4',time:11,box,score:.8,source:'experimental',reviewed:false,evidence:'detection'}];
 assert.ok(tracking.trackingSchema.safeParse(d).success);
 assert.equal(tracking.trackingSchema.safeParse({...d,points:[...d.points,{...d.points[1],time:12}]}).success,false);
 assert.equal(tracking.isActive(d,'R4',11),true);assert.equal(tracking.isActive(d,'R2',11),false);
});

test('splitParticipants keeps referees out of tactical players and separates goalkeepers',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking());
 d.players=d.players.map(p=>p.id==='A1'?{...p,role:'goalkeeper'}:p.id==='B5'?{...p,role:'player'}:p);
 d.identities=[identity('B1',{role:'goalkeeper',team:'B'}),identity('B5',{role:'goalkeeper',team:'B'})];
 const {players,goalkeepers,referees}=tracking.splitParticipants(d);
 assert.deepEqual(goalkeepers.map(p=>p.id),['A1','B1']);
 assert.deepEqual(referees.map(p=>p.id),['R1','R2','R3']);
 assert.equal(players.length,20);assert.ok(players.some(p=>p.id==='B5'));
 assert.ok(!players.some(p=>p.team==='ref'||p.team==='ball'));
 assert.deepEqual(tracking.splitParticipants(tracking.initialTracking()).referees,[]);
});

test('soccerRoster resolves substitutions, roles and excludes the ball',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking());
 d.players.push({id:'sub',name:'Sub',number:'12',team:'A',starter:false,role:'goalkeeper'});
 d.substitutions.push({id:'s',time:30,off:'A1',on:'sub'});
 const early=tracking.soccerRoster(d,10),late=tracking.soccerRoster(d,31);
 assert.ok(!early.some(r=>r.id==='ball'));assert.equal(early.length,26);
 assert.deepEqual(early.find(r=>r.id==='R1'),{id:'R1',team:'ref',number:'',role:'referee',active:true});
 assert.equal(early.find(r=>r.id==='A1').active,true);assert.equal(late.find(r=>r.id==='A1').active,false);
 assert.equal(late.find(r=>r.id==='sub').active,true);assert.equal(late.find(r=>r.id==='sub').role,'goalkeeper');
 assert.equal(early.find(r=>r.id==='A2').role,undefined);
});

test('identity and log limits are enforced',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking()),ok=v=>tracking.trackingSchema.safeParse({...d,...v}).success;
 assert.ok(ok({identities:[identity('A1'),identity('R1',{role:'referee',team:undefined})]}));
 assert.equal(ok({identities:[identity('A1'),identity('A1')]}),false);
 assert.equal(ok({identities:[identity('nobody')]}),false);
 assert.equal(ok({identities:[identity('ball')]}),false);
 assert.equal(ok({identities:[identity('A1',{gallery:Array.from({length:7},()=>[0])})]}),false);
 assert.equal(ok({identities:[identity('A1',{gallery:[Array(81).fill(0)]})]}),false);
 assert.equal(ok({identities:[identity('A1',{identityConfidence:1.1})]}),false);
 const players=[...d.players,...Array.from({length:35},(_,i)=>({id:'p'+i,name:'Sub',number:'',team:i%2?'A':'B',starter:false}))],ids=players.filter(p=>p.team!=='ball').map(p=>p.id);
 assert.equal(ids.length,60);
 assert.ok(ok({players,identities:ids.slice(0,40).map(id=>identity(id))}));
 assert.equal(ok({players,identities:ids.slice(0,41).map(id=>identity(id))}),false);
 assert.ok(ok({trackingLog:Array(300).fill(reid)}));
 assert.equal(ok({trackingLog:Array(301).fill(reid)}),false);
 assert.equal(ok({trackingLog:[{...reid,message:'x'.repeat(241)}]}),false);
 assert.equal(ok({trackingLog:[{...reid,kind:'guess'}]}),false);
});

test('cleanIdentities and appendTrackingLog make tracker output storable',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking());
 const raw=[identity('A1',{gallery:Array.from({length:9},(_,i)=>Array.from({length:90},()=>i/9+1e-9)),identityConfidence:1.4,lastSeen:-3,lastBox:{x:-.1,y:.2,w:0,h:.1},lastPitch:{x:NaN,y:1}}),identity('ball'),identity('ghost'),identity('A2',{status:'offscreen'}),identity('A2',{status:'active'}),identity('R1',{role:'referee',team:undefined})];
 const clean=tracking.cleanIdentities(d,raw);
 assert.deepEqual(clean.map(i=>i.playerId+':'+i.status),['A1:missing','A2:active','R1:missing']);
 assert.equal(clean[0].gallery.length,6);assert.ok(clean[0].gallery.every(v=>v.length===80));assert.equal(clean[0].gallery[0][0],.333);
 assert.equal(clean[0].identityConfidence,1);assert.equal(clean[0].lastSeen,0);assert.equal(clean[0].lastBox.w,.001);assert.equal(clean[0].lastPitch,undefined);
 assert.ok(tracking.trackingSchema.safeParse({...d,identities:clean}).success);
 const events=[{...reid,message:'m'.repeat(500),scores:{...reid.scores,final:.93333333}},{time:-1,kind:'sanity',message:'x'},{time:2,kind:'deferred',track:2.5,message:'tie 45/48/51%'}];
 const log=tracking.appendTrackingLog([],events);
 assert.equal(log.length,3);assert.equal(log[0].message.length,240);assert.equal(log[0].scores.final,.9333);assert.equal(log[1].time,0);assert.equal(log[2].track,undefined);
 assert.equal(tracking.appendTrackingLog(Array(300).fill(clone(log[1])),[reid]).at(-1).kind,'reid');
 assert.equal(tracking.appendTrackingLog(Array(300).fill(clone(log[1])),[reid]).length,300);
 assert.deepEqual(tracking.appendTrackingLog(log,[reid],0),[]);
 assert.ok(tracking.trackingSchema.safeParse({...d,trackingLog:tracking.appendTrackingLog(undefined,events)}).success);
});

test('native Norfair setup and merge still accept documents with officials and identities',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking());
 d.points=['A1','ball','R1'].map(playerId=>({playerId,time:0,box,score:1,source:'manual',reviewed:true}));
 d.identities=[identity('A1')];d.trackingLog=[reid];
 const video={id:'v',filename:'v.mp4',width:1280,height:720,duration:30},setup=native.makeNativeSetup(d,video,0,10);
 assert.equal(setup.seeds.length,3);
 const result={format:'pitchiq-norfair-result',version:1,video,segment:setup.segment,setupKey:setup.setupKey,points:[{playerId:'R1',time:1,box,score:.7,source:'experimental',reviewed:false,evidence:'detection'}],issues:[],summary:{elapsedSeconds:1,processedSeconds:10,observed:1,predicted:0,stopReason:'Complete'}};
 const merged=native.mergeNativeResult(d,result,'v').document;
 assert.equal(merged.points.length,4);assert.equal(merged.identities.length,1);assert.equal(merged.trackingLog.length,1);
});

test('confirmedAnchors returns the latest confirmed box per active player inside the step window',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking());
 d.players.push({id:'sub',name:'Sub',number:'12',team:'A',starter:false});d.substitutions.push({id:'s',time:5,off:'A2',on:'sub'});
 const p=(playerId,time,source='manual',x=.3)=>({playerId,time,box:{...box,x},score:1,source,reviewed:source!=='experimental'});
 d.points=[p('A1',1.05),p('A1',1.15,'detector',.4),p('A3',1.1,'experimental'),p('ball',1.1),p('R1',1.2),p('A1',1.3),p('B1',1)];
 assert.deepEqual(tracking.confirmedAnchors(d,1,1.2).map(a=>a.playerId+':'+a.box.x),['A1:0.4','R1:0.3']);
 assert.deepEqual(tracking.confirmedAnchors(d,.875,1.001).map(a=>a.playerId),['B1']);
 d.points.push(p('A2',6),p('sub',6));
 assert.deepEqual(tracking.confirmedAnchors(d,5.8,6).map(a=>a.playerId),['sub']);
});

test('observationPoint clips boxes, bounds numbers and always yields schema-valid experimental points',()=>{
 const d=tracking.ensureOfficials(tracking.initialTracking());
 const o={playerId:'A7',time:2.123456,box:{x:-.02,y:.95,w:.05,h:.1},score:1.3,evidence:'predicted',track:91,conf:.934567,pitch:{x:.5,y:1.2}};
 const point=tracking.observationPoint(o);
 assert.deepEqual(point,{playerId:'A7',time:2.1235,box:{x:0,y:.95,w:.03,h:.05},score:1,source:'experimental',reviewed:false,evidence:'predicted',track:91,conf:.935,pitch:{x:.5,y:1.2}});
 assert.equal(tracking.observationPoint({...o,pitch:{x:3,y:.5}}).pitch,undefined);
 assert.equal(tracking.observationPoint({...o,track:-4}).track,undefined);
 assert.equal(tracking.observationPoint({...o,conf:NaN}).conf,undefined);
 assert.equal(tracking.observationPoint({...o,box:{x:1.2,y:.5,w:.05,h:.1}}),undefined);
 assert.equal(tracking.observationPoint({...o,box:{x:NaN,y:.5,w:.05,h:.1}}),undefined);
 assert.equal(tracking.observationPoint({...o,time:-1}),undefined);
 const odd=[{x:.99,y:.99,w:.5,h:.5},{x:.3333333333,y:.1234567891,w:.6666666667,h:.3},{x:0,y:0,w:1,h:1}].map(b=>tracking.observationPoint({...o,box:b}));
 assert.ok(tracking.trackingSchema.safeParse({...d,points:odd.map((q,i)=>({...q,time:i}))}).success);
});
