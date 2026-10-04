const {test}=require('node:test'),assert=require('node:assert/strict'),load=require('./load-ts.cjs');
const {SoccerPipeline,GlobalIdentityManager}=load('lib/soccer-pipeline.ts');
const {initialTracking,trackingSchema,correctPoint}=load('lib/tracking.ts');
const {detectPitch}=load('lib/pitch-detector.ts');
const {homography,projectPosition}=load('lib/pitch-coordinates.ts');
const {validHistoryPoint}=load('lib/tracking-history.ts');
const camera={scale:1,dx:0,dy:0,reliable:true,cut:false};
const feature=i=>Array.from({length:50},(_,j)=>i===j?1:0);
const embed=i=>Array.from({length:512},(_,j)=>i===j?1:0);
const box={x:.25,y:.4,w:.04,h:.15};
const detection=(id='d',x=.25,person=1,team=1)=>({id,kind:'person',score:.95,box:{...box,x},appearance:feature(team),kit:[...feature(team),...feature(3),...feature(4)],embedding:embed(person),quality:.95,fieldScore:1,fieldReliable:true});
const noMap=()=>undefined;
function seeded(){const p=new SoccerPipeline(initialTracking(),0,true);p.seed('A1',box,0,detection());p.seed('A2',{...box,x:.6},0,detection('a2',.6,2));p.seed('B1',{...box,x:.8},0,detection('b1',.8,3,10));return p;}

test('persistent global identity survives a cut, ten seconds missing, and sparse embeddings',()=>{
 const p=seeded(),old=p.state.identities.find(g=>g.id==='A1').localId;p.cut();
 let observations;
 for(let step=0;step<=15;step++){
  const d=detection('return',.4);if(step%5!==0)delete d.embedding;
  observations=p.step([d],10+step*.2,camera,noMap);
 }
 assert.equal(observations[0].globalId,'A1');assert.notEqual(observations[0].localId,old);
 assert.equal(p.state.identities.length,3);assert.ok(p.state.events.some(e=>e.accepted&&e.globalId==='A1'));
 assert.ok(trackingSchema.safeParse({...initialTracking(),soccer:p.state}).success);
});
test('same-kit ambiguity produces UNKNOWN_PLAYER rather than a forced match',()=>{
 const p=seeded();p.state.identities.find(g=>g.id==='A2').gallery=p.state.identities.find(g=>g.id==='A1').gallery;p.cut();
 for(let i=0;i<30;i++){const o=p.step([detection('return',.4)],10+i*.2,camera,noMap);assert.equal(o[0].globalId,undefined);}
 assert.equal(p.state.identities.length,3);
});
test('missing embeddings cannot reconnect a player by position and shirt alone',()=>{
 const p=seeded();p.cut();for(let i=0;i<30;i++){const d=detection();delete d.embedding;assert.equal(p.step([d],10+i*.2,camera,noMap)[0].globalId,undefined);}
});
test('conflicting teams, impossible pitch travel, and substitutions are rejected',()=>{
 const p=seeded(),g=p.state.identities[0],manager=new GlobalIdentityManager(p.state);
 assert.match(manager.candidate(g,detection(),'B','PLAYER_TEAM_B',1).reason,/team/);
 g.pitch={x:0,y:0};g.pitchTime=0;assert.match(manager.candidate(g,detection(),'A','PLAYER_TEAM_A',1,{x:1,y:1}).reason,/implausible/);
 g.state='SUBSTITUTED';assert.match(manager.candidate(g,detection(),'A','PLAYER_TEAM_A',10).reason,/Substituted/);
});
test('referee seed remains separate and role survives a poor frame',()=>{
 const p=seeded(),d=detection('ref',.45,8,15);p.seed('REF-1',d.box,0,d,'REFEREE');
 for(let i=1;i<9;i++)p.step([{...d,appearance:feature(1)}],i*.2,camera,noMap);
 const ref=p.state.identities.find(g=>g.id==='REF-1');assert.equal(ref.role,'REFEREE');assert.equal(ref.team,'UNKNOWN');
 assert.ok(!initialTracking().players.some(g=>g.id==='REF-1'));
});
test('outside-field people are excluded; uncertain views retain local boxes without roster admission',()=>{
 const p=seeded();p.cut();for(let i=0;i<20;i++){const d={...detection(),fieldScore:0};assert.equal(p.step([d],i*.2,camera,noMap).length,0);}
 const d={...detection(),fieldReliable:false};const observations=p.step([d],5,camera,noMap);assert.equal(observations.length,1);assert.equal(observations[0].globalId,undefined);assert.equal(p.state.identities.length,3);
});
test('pitch mask rejects stands, disconnected green patches and fails closed',()=>{
 const width=320,height=180,data=new Uint8ClampedArray(width*height*4);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){const i=(y*width+x)*4,on=y>70&&y<170&&x>20&&x<300;data.set(on?[40,115,45,255]:[120,90,80,255],i);}
 const mask=detectPitch({width,height,data});assert.equal(mask.reliable,true);
 assert.ok(mask.score({x:.4,y:.45,w:.04,h:.2})>.7);assert.equal(mask.score({x:.4,y:.05,w:.04,h:.1}),0);
 assert.equal(detectPitch({width,height,data:new Uint8ClampedArray(data.length)}).score(box),0);
});
test('homography maps real corners and rejects collinear calibration',()=>{
 const corners=[{x:.1,y:.2},{x:.9,y:.2},{x:.8,y:.9},{x:.2,y:.9}],h=homography(corners);
 for(const [i,c] of corners.entries()){const p=projectPosition(h,c);assert.ok(Math.abs(p.x-([0,1,1,0][i]))<1e-8);assert.ok(Math.abs(p.y-([0,0,1,1][i]))<1e-8);}
 assert.equal(homography([{x:0,y:0},{x:.2,y:0},{x:.4,y:0},{x:.6,y:0}]),undefined);
});
test('manual correction invalidates old archived identity without invalidating new generations',()=>{
 const point={playerId:'A1',time:10,box,score:1,source:'experimental',reviewed:false};
 const doc=correctPoint(initialTracking(),{...point,source:'manual',reviewed:true});
 assert.equal(validHistoryPoint(doc,{generation:0},point),false);assert.equal(validHistoryPoint(doc,{generation:1},point),true);
});
