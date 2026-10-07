// Headless timing of the soccer tracking layer on a synthetic 1280x720 broadcast frame.
// Usage: node scripts/benchmark-soccer.cjs [steps=150] [width=1280] [height=720]
// Times analyzePitch, describe (30 boxes) and SoccerTracker.step (detection steps every 0.2 s). Detection itself
// (YOLOX in the browser) is not included; see scripts/benchmark-tracking.cjs for that.
const path=require('node:path');
require(path.join(__dirname,'../tests/helpers/ts-require.cjs'));
const {makeScene}=require(path.join(__dirname,'../tests/helpers/soccer-scene.cjs'));
const {analyzePitch}=require(path.join(__dirname,'../lib/soccer/pitch.ts'));
const {describe}=require(path.join(__dirname,'../lib/soccer/appearance.ts'));
const {SoccerTracker}=require(path.join(__dirname,'../lib/soccer/pipeline.ts'));
const tracking=require(path.join(__dirname,'../lib/tracking.ts'));

const steps=Math.max(10,Number(process.argv[2])||150),W=Number(process.argv[3])||1280,H=Number(process.argv[4])||720;
const scene=makeScene(W,H,11);
let seed=3;const rnd=()=>{seed=(seed*1103515245+12345)%2147483648;return seed/2147483648;};
// 11 + 11 players, a referee, five fans in the stands and two people on the bench: 30 person detections.
const field=[];
for(let i=0;i<22;i++)field.push({key:'p'+i,kit:i<11?'red':'blue',x:.06+.88*((i%11)+.5)/11,y:.3+.45*(((i*7)%11)/10),vx:0,vy:0});
field.push({key:'ref',kit:'ref',x:.5,y:.55,vx:0,vy:0});
const fans=[.1,.3,.5,.7,.9].map((x,i)=>({key:'fan'+i,kit:'fan',x,y:.11,h:.07,score:.6}));
const bench=[.3,.36].map((x,i)=>({key:'bench'+i,kit:'coat',x,y:.97,score:.7}));
function advance(){
 for(const p of field){
  p.vx=.8*p.vx+.004*(rnd()-.5);p.vy=.8*p.vy+.003*(rnd()-.5);
  p.x=Math.min(.95,Math.max(.05,p.x+p.vx));p.y=Math.min(.82,Math.max(.27,p.y+p.vy));
 }
 return [...field,...fans,...bench];
}
const stats=v=>{const s=[...v].sort((a,b)=>a-b),q=f=>s[Math.min(s.length-1,Math.floor(f*s.length))];return {mean:+(v.reduce((a,b)=>a+b,0)/v.length).toFixed(2),p50:+q(.5).toFixed(2),p95:+q(.95).toFixed(2),max:+s[s.length-1].toFixed(2)};};
const now=()=>performance.now();

const doc=tracking.ensureOfficials(tracking.initialTracking());
const still={scale:1,dx:0,dy:0,reliable:true,cut:false};
const frames=[];
for(let k=0;k<steps;k++){const people=advance();frames.push({t:Number((k*.2).toFixed(4)),frame:scene.paint(people),detections:people.map(scene.detect)});}

// analyzePitch (with the previous model, as the tracker calls it)
const pitchMs=[];let model;
for(const [i,f] of frames.entries()){const t0=now();model=analyzePitch(f.frame,f.t,model,still);if(i>=3)pitchMs.push(now()-t0);}
// describe: 30 boxes per frame, each with the others as occluders
const describeMs=[];
for(const [i,f] of frames.entries()){const boxes=f.detections.map(d=>d.box),t0=now();for(const b of boxes)describe(f.frame,b,boxes);if(i>=3)describeMs.push(now()-t0);}
// SoccerTracker.step: full layer per detection step
const tracker=new SoccerTracker(),stepMs=[],inner={pitch:[],describe:[]};let observations=0,last;
for(const [i,f] of frames.entries()){
 const t0=now(),r=tracker.step({time:f.t,frame:f.frame,detections:f.detections,camera:still,anchors:[],roster:tracking.soccerRoster(doc,f.t)}),ms=now()-t0;
 observations+=r.observations.length;last=r;
 if(i>=3){stepMs.push(ms);inner.pitch.push(tracker.lastTiming.pitch);inner.describe.push(tracker.lastTiming.describe);}
}
const confirmed=last.debug.tracks.filter(d=>d.state==='confirmed').length;if(process.env.BENCH_DEBUG)for(const d of last.debug.tracks)if(d.state!=='confirmed')console.error(d.label,d.state,d.reason);
console.log(JSON.stringify({frame:`${W}x${H}`,steps,people:frames[0].detections.length,
 analyzePitch_ms:stats(pitchMs),describe30_ms:stats(describeMs),
 soccerStep_ms:stats(stepMs),soccerStep_pitch_ms:stats(inner.pitch),soccerStep_describe_ms:stats(inner.describe),
 confirmedAtEnd:confirmed,rejectedAtEnd:last.debug.rejected.length,observations},null,1));
