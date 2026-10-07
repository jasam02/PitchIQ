const test=require('node:test');
const assert=require('node:assert/strict');
const {makeFrame}=require('./helpers/ts-require.cjs');
const pitch=require('../lib/soccer/pitch.ts');
const relevance=require('../lib/soccer/relevance.ts');
const homography=require('../lib/soccer/homography.ts');
const {analyzePitch,warpPitch,zoneOf,pointInPolygon,signedDistance,nearestBoundary,defaultFieldFilter,LINE_MAX_AGE}=pitch;
const {assessDetections,fitSizeModel}=relevance;
const {solveHomography,applyHomography,invertHomography,pitchHomography,toPitch}=homography;

// Synthetic broadcast view in 320x180 "scene units" (scaled to any frame size): stands at the top, an
// advertising board band, thin far run-off, far touchline, pitch with halfway line and a left penalty box
// (goal line off-frame on the left), near touchline, run-off + bench strip below, right goal line with
// run-off and non-grass beyond.
const LINE=[230,235,230];
const nearY=X=>140+.03*(X-160),goalX=Y=>280-.3*(Y-100),boxX=Y=>50+.4*(Y-100);
function scene(W=320,H=180,o={}){
 const s=W/320,near=o.near!==false,shift=o.shift||0,players=o.players!==false;
 return makeFrame(W,H,(x,y)=>{const X=x/s,Y=y/s-shift;
  if(Y<40)return ((Math.floor(X)*7+Math.floor(Y)*13)%23<2)?[220,220,220]:[95,95,105];
  if(Y<48)return Math.floor(X/40)%3===0?[235,235,235]:Math.floor(X/40)%3===1?[40,60,200]:[200,40,40];
  const gx=goalX(Y);
  if(X>gx+14)return [60,60,70];
  if(Y>=158)return Math.floor(X/25)%2?[80,60,40]:[150,150,155];
  if(players){
   if(X>=200&&X<205&&Y>=126&&Y<143)return Y<133?[200,30,30]:[20,20,20]; // on the near touchline
   if(X>=120&&X<124&&Y>=78&&Y<92)return Y<84?[235,235,235]:[15,15,15]; // white shirt
   if(X>=60&&X<63&&Y>=98&&Y<108)return [30,40,200];
  }
  if(Math.abs(Y-52.5)<1&&X<=gx)return LINE;
  if(near&&Math.abs(Y-nearY(X))<1&&X<=gx)return LINE;
  if(Y>52&&Y<nearY(X)&&Math.abs(X-gx)<1)return LINE;
  if(Y>=62&&Y<=125&&Math.abs(X-boxX(Y))<1)return LINE;
  if((Math.abs(Y-62)<1&&X<boxX(62))||(Math.abs(Y-125)<1&&X<boxX(125)))return LINE;
  if(Y>52&&Y<nearY(X)&&Math.abs(X-(160+.1*(Y-100)))<1)return LINE;
  return Math.floor(X/30)%2?[40,140,50]:[52,162,62];});
}
const px=(x,y)=>({x:x/320,y:y/180});
// Box whose foot point (bottom centre) is at scene pixel (x,y).
const footBox=(x,y,h=.12,w=.025)=>({x:x/320-w/2,y:y/180-h,w,h});
const person=(box,score=.8,id='p')=>({id,kind:'person',score,box});
const still={scale:1,dx:0,dy:0,reliable:true,cut:false};
const lineY=(l,x)=>l.a.y+(l.b.y-l.a.y)*(x-l.a.x)/(l.b.x-l.a.x);
const lineX=(l,y)=>l.a.x+(l.b.x-l.a.x)*(y-l.a.y)/(l.b.y-l.a.y);
const side=(m,s)=>m.lines.find(l=>l.side===s);
const base=analyzePitch(scene(),0);

test('pitch polygon follows the playable area and clips at boundary lines',()=>{
 assert.equal(base.reliable,true);
 assert.equal(base.aspect,320/180);
 assert.ok(base.coverage>.35&&base.coverage<.55,`coverage ${base.coverage}`);
 const sides=base.lines.map(l=>l.side).sort();
 assert.deepEqual(sides,['far','near','right']);
 assert.ok(Math.abs(lineY(side(base,'near'),.5)-px(0,140.5).y)<.01);
 assert.ok(Math.abs(lineY(side(base,'far'),.5)-px(0,53).y)<.01);
 assert.ok(Math.abs(lineX(side(base,'right'),px(0,90.5).y)-px(goalX(90)+.5,0).x)<.01);
 for(const l of base.lines){assert.equal(l.age,0);assert.ok(l.support>.3&&l.support<=1);}
 const inside=[px(160,90),px(10,100),px(30,70),px(250,135),px(200,60)];
 const outside=[px(160,20),px(160,44),px(160,150),px(40,170),px(300,100),px(160,50)];
 for(const p of inside)assert.ok(pointInPolygon(p,base.polygon),`inside ${JSON.stringify(p)}`);
 for(const p of outside)assert.ok(!pointInPolygon(p,base.polygon),`outside ${JSON.stringify(p)}`);
 // run-off grass is part of the grass hull but not of the playable polygon
 assert.ok(pointInPolygon(px(160,150),base.grassPolygon));
 // convex
 const P=base.polygon;let pos=0,neg=0;
 for(let i=0;i<P.length;i++){const a=P[i],b=P[(i+1)%P.length],c=P[(i+2)%P.length],cr=(b.x-a.x)*(c.y-b.y)-(b.y-a.y)*(c.x-b.x);if(cr>1e-9)pos++;else if(cr<-1e-9)neg++;}
 assert.ok(pos===0||neg===0);
});

test('penalty-box and halfway lines are never boundaries',()=>{
 assert.ok(!base.lines.some(l=>l.side==='left'));
 // players in the left penalty area and between halfway and goal line stay inside
 assert.equal(zoneOf(base,footBox(12,100)).zone,'inside');
 assert.equal(zoneOf(base,footBox(30,122)).zone,'inside');
 assert.equal(zoneOf(base,footBox(220,100)).zone,'inside');
 // without the right goal line the halfway line still is not a boundary
 const noGoal=analyzePitch(makeFrame(320,180,(x,y)=>{if(y<40)return [95,95,105];if(y<48)return [40,60,200];if(y>=158)return [80,60,40];if(Math.abs(y-nearY(x))<1)return LINE;if(Math.abs(x-(160+.1*(y-100)))<1&&y>52)return LINE;if(Math.abs(y-52.5)<1)return LINE;return [45,150,55];}),0);
 assert.ok(!noGoal.lines.some(l=>l.side==='left'||l.side==='right'));
 assert.equal(zoneOf(noGoal,footBox(300,100)).zone,'inside');
 // grass bounded by non-grass on both sides: only the thin-band test keeps the halfway line out
 const boxed=analyzePitch(makeFrame(320,180,(x,y)=>{if(y<40||x<12||x>=308)return [95,95,105];if(y<48)return [40,60,200];if(y>=158)return [80,60,40];if(Math.abs(y-nearY(x))<1)return LINE;if(Math.abs(x-(150+.1*(y-100)))<1&&y>52)return LINE;if(Math.abs(y-52.5)<1)return LINE;return [45,150,55];}),0);
 assert.ok(!boxed.lines.some(l=>l.side==='left'||l.side==='right'),JSON.stringify(boxed.lines));
 assert.equal(zoneOf(boxed,footBox(60,100)).zone,'inside');
 assert.equal(zoneOf(boxed,footBox(260,100)).zone,'inside');
});

test('near/far side rules: run-off to the frame bottom, long interior lines near either side',()=>{
 const G=[45,150,55];
 // near touchline close to the bottom with run-off grass running off-frame: still a boundary (thin band)
 const lowLine=analyzePitch(makeFrame(320,180,(x,y)=>y<40?[95,95,105]:y<48?[40,60,200]:Math.abs(y-52.5)<1||Math.abs(y-165.5)<1?LINE:G),0);
 assert.ok(side(lowLine,'near'));
 assert.equal(zoneOf(lowLine,footBox(160,176)).zone,'outside');
 assert.equal(zoneOf(lowLine,footBox(160,150)).zone,'inside');
 // a long interior line (zoomed penalty box) with deep pitch beyond running off-frame is not a boundary
 const zoomed=analyzePitch(makeFrame(320,180,(x,y)=>y<10?[95,95,105]:y<14?[40,60,200]:Math.abs(y-16.5)<1||Math.abs(y-150.5)<1?LINE:G),0);
 assert.ok(!side(zoomed,'near'));
 assert.equal(zoneOf(zoomed,footBox(160,172)).zone,'inside');
 // far touchline with boards right behind it (no run-off): an interior line 17px below is not a boundary
 const farBox=analyzePitch(makeFrame(320,180,(x,y)=>y<40?[95,95,105]:y<52?[40,60,200]:Math.abs(y-52.5)<1||Math.abs(y-70.5)<1||Math.abs(y-nearY(x))<1?LINE:y>=158?[80,60,40]:G),0);
 assert.ok(!side(farBox,'far'));
 assert.ok(side(farBox,'near'));
 assert.equal(zoneOf(farBox,footBox(160,62,.06)).zone,'inside');
});

test('zones: inside, boundary tolerance beyond the touchline, outside',()=>{
 assert.deepEqual(zoneOf(base,footBox(140,90)).zone,'inside');
 assert.ok(zoneOf(base,footBox(140,90)).outsideBy<0);
 const line=nearY(160)+.5;
 for(const d of [2,4]){const z=zoneOf(base,footBox(160,line+d));assert.equal(z.zone,'boundary',`+${d}px`);assert.ok(z.outsideBy>0&&z.outsideBy<=.03);}
 assert.equal(zoneOf(base,footBox(160,line+14)).zone,'outside');
 // tolerance scales with the person's height, within [minMargin,maxMargin]
 assert.equal(zoneOf(base,footBox(160,line+3,.04)).zone,'outside'); // .35*.04=.014 < 3px=.0167
 assert.equal(zoneOf(base,footBox(160,line+3,.04),{...defaultFieldFilter,boundaryMargin:.6}).zone,'boundary');
 // a few px beyond the far touchline (assistant referee) is boundary as well
 assert.equal(zoneOf(base,footBox(160,51,.06)).zone,'boundary');
 assert.equal(zoneOf(base,footBox(160,20,.05)).zone,'outside');
 // disabled config never filters
 const off={...defaultFieldFilter,enabled:false};
 assert.deepEqual(zoneOf(base,footBox(160,20,.05),off),{zone:'inside',outsideBy:0});
});

test('relevance: coach beyond the near touchline, audience in the stands, crowd cluster',()=>{
 const dets=[
  person(footBox(140,90),.9,'player'),
  person(footBox(160,nearY(160)+4),.9,'touchline'),
  person(footBox(100,168,.13),.9,'coach'),
  person(footBox(160,152,.13),.9,'runoff'),
  person(footBox(150,30,.05),.7,'spectator'),
  person(footBox(300,100,.1),.8,'goal-runoff'),
  {id:'ball',kind:'ball',score:.6,box:{x:.5,y:.5,w:.01,h:.01}},
 ];
 const r=assessDetections(dets,base,defaultFieldFilter);
 assert.deepEqual(r.accepted.map(d=>d.id).sort(),['player','touchline']);
 assert.equal(r.accepted.find(d=>d.id==='touchline').zone,'boundary');
 assert.equal(r.accepted.find(d=>d.id==='player').zone,'inside');
 assert.deepEqual(r.accepted.find(d=>d.id==='player').foot,{x:140/320,y:90/180});
 const reason=id=>r.rejected.find(x=>x.box===dets.find(d=>d.id===id).box);
 assert.equal(reason('coach').reason,'outside-pitch');
 assert.match(reason('coach').detail,/beyond near touchline/);
 assert.equal(reason('runoff').reason,'outside-pitch');
 assert.equal(reason('spectator').reason,'audience');
 assert.equal(reason('goal-runoff').reason,'outside-pitch');
 assert.match(reason('goal-runoff').detail,/right goal line/);
 assert.equal(r.balls.length,1);assert.equal(r.balls[0],dets[6]);
 // a single person off the grass on the right is outside-pitch; a dense group there is a crowd
 const lone=assessDetections([person(footBox(308,110,.06))],base,defaultFieldFilter);
 assert.equal(lone.rejected[0].reason,'outside-pitch');
 const group=[0,1,2,3].map(i=>person(footBox(300+i*4,108+i*2,.06),.7,'g'+i));
 const crowd=assessDetections(group,base,defaultFieldFilter);
 assert.equal(crowd.accepted.length,0);
 assert.ok(crowd.rejected.every(x=>x.reason==='audience'),JSON.stringify(crowd.rejected.map(x=>x.detail)));
 assert.equal(nearestBoundary(base,px(160,170)).side,'near');
 assert.equal(nearestBoundary(base,px(160,170)).label,'near touchline');
});

test('unreliable frame: zone unknown and nothing rejected by zone',()=>{
 const crowdShot=makeFrame(320,180,(x,y)=>x>40&&x<100&&y>120&&y<160?[40,140,50]:[(x*3)%120+60,90,(y*5)%100+60]);
 const m=analyzePitch(crowdShot,1);
 assert.equal(m.reliable,false);
 assert.deepEqual(m.polygon,[]);
 assert.equal(m.coverage,0);
 assert.deepEqual(zoneOf(m,footBox(160,30,.05)),{zone:'unknown',outsideBy:0});
 const dets=[person(footBox(160,30,.05)),person(footBox(60,150,.3)),person(footBox(200,100,.02))];
 const r=assessDetections(dets,m,defaultFieldFilter);
 assert.equal(r.rejected.length,0);
 assert.equal(r.accepted.length,3);
 assert.ok(r.accepted.every(d=>d.zone==='unknown'));
 // tiny frames are unreliable rather than throwing
 assert.equal(analyzePitch(makeFrame(8,8,()=>[40,140,50]),0).reliable,false);
});

test('disabled field filter accepts every person',()=>{
 const dets=[person(footBox(150,30,.05)),person(footBox(100,168,.13)),person(footBox(140,90,.6))];
 const r=assessDetections(dets,base,{...defaultFieldFilter,enabled:false});
 assert.equal(r.rejected.length,0);
 assert.ok(r.accepted.every(d=>d.zone==='inside'));
});

test('warpPitch moves the region with the camera and ages lines',()=>{
 const cam={scale:1,dx:.05,dy:-.02,reliable:true,cut:false};
 const w=warpPitch(base,cam,.4);
 assert.equal(w.time,.4);
 assert.equal(w.reliable,true);
 for(const l of w.lines){const o=side(base,l.side);assert.ok(Math.abs(l.a.x-o.a.x-.05)<1e-9&&Math.abs(l.a.y-o.a.y+.02)<1e-9);assert.ok(Math.abs(l.age-.4)<1e-9);}
 // a point inside near the left edge moves right with the camera
 assert.ok(pointInPolygon({x:.5+.05,y:.5-.02},w.polygon));
 assert.ok(w.polygon.every(p=>p.x>=-1e-9&&p.x<=1+1e-9&&p.y>=-1e-9&&p.y<=1+1e-9));
 assert.ok(!pointInPolygon({x:.02,y:.5},w.polygon)); // region left of the shifted left edge is gone
 const zoom=warpPitch(base,{scale:1.05,dx:-.025,dy:-.025,reliable:true,cut:false},.2);
 const near=side(zoom,'near'),orig=side(base,'near');
 assert.ok(Math.abs(near.b.y-(orig.b.y*1.05-.025))<1e-9);
 const old=warpPitch(base,still,LINE_MAX_AGE+.1);
 assert.equal(old.lines.length,0);
 const cut=warpPitch(base,{...still,cut:true},.2);
 assert.equal(cut.reliable,false);assert.deepEqual(cut.polygon,[]);assert.deepEqual(cut.lines,[]);
});

test('boundary line is carried with the camera while missing, then dropped',()=>{
 const shift=3,cam={scale:1,dx:0,dy:shift/180,reliable:true,cut:false};
 const missing=scene(320,180,{near:false,shift});
 const fresh=analyzePitch(missing,.2);
 assert.ok(!side(fresh,'near'));
 assert.ok(pointInPolygon(px(160,150+shift),fresh.polygon)); // without memory the run-off looks playable
 const carried=analyzePitch(missing,.2,base,cam);
 const near=side(carried,'near');
 assert.ok(near,'near touchline carried');
 assert.ok(Math.abs(near.age-.2)<1e-9);
 assert.ok(Math.abs(lineY(near,.5)-px(0,nearY(160)+.5+shift).y)<.01);
 assert.ok(!pointInPolygon(px(160,150+shift),carried.polygon));
 assert.equal(zoneOf(carried,footBox(160,165)).zone,'outside');
 // re-detected lines restart at age 0
 const back=analyzePitch(scene(320,180,{shift}),.4,carried,still);
 assert.equal(side(back,'near').age,0);
 // keeps carrying for ~2 s, then the line is dropped
 let prev=carried,t=.2;const ages=[];
 while(t<3){t=+(t+.2).toFixed(2);prev=analyzePitch(missing,t,prev,still);ages.push([t,side(prev,'near')?.age]);}
 assert.ok(ages.filter(([tt])=>tt<=1.8).every(([,a])=>a!==undefined),JSON.stringify(ages));
 assert.ok(ages.filter(([tt])=>tt>=2.4).every(([,a])=>a===undefined),JSON.stringify(ages));
 assert.ok(pointInPolygon(px(160,150+shift),prev.polygon));
 // a camera cut drops all memory; unknown camera motion keeps lines only briefly
 assert.ok(!side(analyzePitch(missing,.2,base,{...cam,cut:true}),'near'));
 const loose={...still,reliable:false};
 assert.ok(side(analyzePitch(missing,.4,{...base,time:0},loose),'near'));
 assert.ok(!side(analyzePitch(missing,.8,{...base,time:0},loose),'near'));
});

test('size model: robust perspective fit and implausible size rejection',()=>{
 const people=[.35,.42,.5,.58,.66,.74,.82,.9].map((y,i)=>({box:{x:.1*i,y:y-(.04+.15*y),w:.02,h:.04+.15*y}}));
 people.push({box:{x:.5,y:.1,w:.1,h:.5}},{box:{x:.6,y:.69,w:.01,h:.01}}); // outliers
 const m=fitSizeModel(people);
 assert.equal(m.reliable,true);
 assert.ok(Math.abs(m.a-.04)<.01&&Math.abs(m.b-.15)<.02,JSON.stringify(m));
 assert.equal(fitSizeModel(people.slice(0,5)).reliable,false);
 assert.equal(fitSizeModel([1,2,3,4,5,6,7].map(i=>({box:{x:i*.1,y:.4,w:.02,h:.1}}))).reliable,false);
 assert.ok(fitSizeModel([{box:{x:0,y:.7,w:.02,h:.08}},{box:{x:0,y:.3,w:.02,h:.12}}]).b>=0);
 // detections on the synthetic pitch: expected height .02 + .1 * footY
 const feet=[[60,70],[100,80],[140,95],[180,105],[220,115],[250,125],[90,130],[200,62]];
 const h=y=>.02+.1*y/180;
 const dets=feet.map(([x,y],i)=>person(footBox(x,y,h(y)),.8,'n'+i));
 const giant=person(footBox(160,110,.45),.8,'giant'),tiny=person(footBox(170,120,.012),.8,'tiny');
 const r=assessDetections([...dets,giant,tiny],base,defaultFieldFilter);
 assert.equal(r.size.reliable,true);
 assert.deepEqual(r.rejected.map(x=>x.reason),['implausible-size','implausible-size']);
 assert.equal(r.accepted.length,8);
 assert.ok(r.accepted.every(d=>Math.abs(d.sizeRatio-1)<.15),JSON.stringify(r.accepted.map(d=>d.sizeRatio)));
 assert.match(r.rejected[0].detail,/expected/);
 // too few people: no model, no size rejections; a previous reliable model is reused
 const few=assessDetections([...dets.slice(0,3),giant],base,defaultFieldFilter);
 assert.equal(few.rejected.length,0);assert.equal(few.size.reliable,false);
 assert.equal(few.accepted.find(d=>d.id==='giant').sizeRatio,1);
 const reuse=assessDetections([...dets.slice(0,3),giant],base,defaultFieldFilter,r.size);
 assert.deepEqual(reuse.rejected.map(x=>x.reason),['implausible-size']);
 assert.equal(reuse.size,r.size);
});

test('low-score detections inside are kept down to .08',()=>{
 const r=assessDetections([person(footBox(140,90),.1,'weak'),person(footBox(150,90),.05,'noise')],base,defaultFieldFilter);
 assert.deepEqual(r.accepted.map(d=>d.id),['weak']);
 assert.equal(r.rejected[0].reason,'low-confidence');
});

test('geometry helpers',()=>{
 const sq=[{x:.2,y:.2},{x:.8,y:.2},{x:.8,y:.8},{x:.2,y:.8}];
 assert.ok(pointInPolygon({x:.5,y:.5},sq));
 assert.ok(!pointInPolygon({x:.9,y:.5},sq));
 assert.ok(Math.abs(signedDistance({x:.5,y:.5},sq,1)+.3)<1e-9);
 assert.ok(Math.abs(signedDistance({x:.9,y:.5},sq,2)-.2)<1e-9); // dx measured in height units
 assert.ok(Math.abs(signedDistance({x:.5,y:.9},sq,2)-.1)<1e-9);
 assert.equal(signedDistance({x:.5,y:.5},[],1),Infinity);
});

test('analyzePitch scales to full HD within budget and agrees with the low-res result',()=>{
 const big=scene(1920,1080);
 for(let i=0;i<3;i++)analyzePitch(big,0); // warm up the JIT
 const times=[];let m;
 for(let i=0;i<5;i++){const t=performance.now();m=analyzePitch(big,0);times.push(performance.now()-t);}
 times.sort((a,b)=>a-b);
 assert.ok(times[2]<40,`median ${times[2].toFixed(1)} ms`);
 assert.deepEqual(m.lines.map(l=>l.side).sort(),['far','near','right']);
 assert.ok(Math.abs(lineY(side(m,'near'),.5)-lineY(side(base,'near'),.5))<.01);
 assert.ok(Math.abs(m.coverage-base.coverage)<.03);
});

// ---------- boundary edge cases (640x360 scenes) ----------
const G2=x=>Math.floor(x/40)%2?[40,140,50]:[52,162,62];
const foot640=(x,y,h,w=.02)=>({x:x/640-w/2,y:y/360-h,w,h});
const zones=(m,boxes)=>boxes.map(b=>zoneOf(m,b).zone);

test('green seats behind advertising boards never join the pitch',()=>{
 for(const boards of [6,10,16]){
  const m=analyzePitch(makeFrame(640,360,(x,y)=>y<70?(((x*7+y*13)%23<3)?[200,200,200]:[30,110,60]):y<70+boards?[40,60,200]:y>=330?[80,60,40]:Math.abs(y-96)<1.2||Math.abs(y-300)<1.2?LINE:G2(x)),0);
  assert.ok(Math.min(...m.grassPolygon.map(p=>p.y))>.18,`boards ${boards}: grass hull reaches the stands`);
  assert.ok(Math.abs(lineY(side(m,'far'),.5)-96/360)<.01,`boards ${boards}: far touchline ${JSON.stringify(m.lines)}`);
  const fans=[[300,40],[300,60]].map(([x,y])=>person(foot640(x,y,.06,.015)));
  assert.deepEqual(assessDetections(fans,m,defaultFieldFilter).accepted,[],`boards ${boards}`);
 }
});

test('a short marking or an oblique goal line outside a touchline never hides the touchline',()=>{
 // Technical-area marking 1 m beyond the near touchline along the dugout (30% of the width).
 const tech=analyzePitch(makeFrame(640,360,(x,y)=>{if(y<70)return [95,95,105];if(y<85)return [40,60,200];if(y>=330)return [80,60,40];
  if(Math.abs(y-96)<1.2||Math.abs(y-280)<1.2)return LINE;if(Math.abs(y-292)<1.2&&x>=200&&x<390)return LINE;if(y>280&&y<292&&(Math.abs(x-200)<1.2||Math.abs(x-390)<1.2))return LINE;return G2(x);}),0);
 assert.ok(Math.abs(lineY(side(tech,'near'),.5)-280/360)<.01,JSON.stringify(tech.lines));
 assert.deepEqual(zones(tech,[[300,318],[350,322],[100,320]].map(([x,y])=>foot640(x,y,.2,.03))),['outside','outside','outside']);
 // Camera on the left corner: the goal line runs down-left at a shallow angle, farther from the centre than the far touchline.
 const farY=x=>100-.02*(x-320),goalY=x=>farY(260)+.33*(260-x);
 const corner=analyzePitch(makeFrame(640,360,(x,y)=>{if(y<60)return [95,95,105];if(y<86)return [40,60,200];if(y>=335)return [80,60,40];
  if(y>farY(x)-2&&x<260&&Math.abs(y-goalY(x))<1.3)return LINE;if(Math.abs(y-farY(x))<1.3&&x>=258)return LINE;if(Math.abs(y-310)<1.3&&y>goalY(x))return LINE;return G2(x);}),0);
 assert.ok(side(corner,'far')&&Math.abs(lineY(side(corner,'far'),.7)-farY(448)/360)<.01,JSON.stringify(corner.lines));
 const runoff=[[380,88],[460,87.5],[540,87]].map(([x,y])=>person({x:x/640-.006,y:y/360-.07,w:.012,h:.07}));
 assert.deepEqual(assessDetections(runoff,corner,defaultFieldFilter).rejected.map(r=>r.reason),['audience','audience','audience']);
});

test('deep grass beyond the far touchline (park ground) keeps the touchline; a painted line at the grass edge makes it interior',()=>{
 for(const trees of [70,60,40]){
  const m=analyzePitch(makeFrame(640,360,(x,y)=>y<trees?[70,70,60]:y>=330?[80,60,40]:Math.abs(y-96)<1.2||Math.abs(y-300)<1.2?LINE:G2(x)),0);
  assert.ok(side(m,'far')&&Math.abs(lineY(side(m,'far'),.5)-96/360)<.01,`trees at ${trees}: ${JSON.stringify(m.lines)}`);
  const beyond=[[200,80],[300,70]].filter(([,y])=>y>trees).map(([x,y])=>foot640(x,y,.07,.015));
  assert.ok(zones(m,beyond).every(z=>z==='outside'),`trees at ${trees}`);
 }
});

test('interior goal-side lines: pillarbox bars are the frame edge, a goal line at the grass edge makes the 18-yard line interior',()=>{
 // 4:3 footage in a 16:9 frame: the halfway line near the right bar is not a goal line.
 const bars=analyzePitch(makeFrame(640,360,(x,y)=>{if(x<80||x>=560)return [0,0,0];if(y<70)return [95,95,105];if(y<85)return [40,60,200];if(y>=330)return [80,60,40];
  if(Math.abs(y-96)<1.2||Math.abs(y-300)<1.2)return LINE;if(y>96&&y<300&&Math.abs(x-(470+.12*(y-200)))<1.2)return LINE;return G2(x);}),0);
 assert.ok(!bars.lines.some(l=>l.side==='left'||l.side==='right'),JSON.stringify(bars.lines));
 assert.deepEqual(zones(bars,[[500,200],[500,280],[510,150]].map(([x,y])=>foot640(x,y,.1))),['inside','inside','inside']);
 // Artificial pitch with terracotta run-off: the goal line has no grass beyond it, the 18-yard line in front of it does.
 const goalX=y=>70-.25*(y-200),boxX=y=>170-.12*(y-200),RUN=[150,70,55];
 const turf=analyzePitch(makeFrame(640,360,(x,y)=>{if(y<60)return [95,95,105];if(y<92||y>=306||x<goalX(y)-1.5)return RUN;
  if(Math.abs(y-93)<1.5||Math.abs(y-305)<1.5||Math.abs(x-goalX(y))<1.5)return LINE;if(y>=125&&y<=275&&Math.abs(x-boxX(y))<1.2)return LINE;
  if((Math.abs(y-125)<1.2||Math.abs(y-275)<1.2)&&x<boxX(y))return LINE;return G2(x);}),0);
 assert.ok(!turf.lines.some(l=>l.side==='left'),JSON.stringify(turf.lines));
 assert.deepEqual(zones(turf,[[110,200],[140,160]].map(([x,y])=>foot640(x,y,.1))),['inside','inside'],'goalkeeper and defender in the box');
});

test('size plausibility measures a lying or diving player along the body',()=>{
 const people=[.32,.38,.44,.5,.56,.62,.68,.74].map((y,i)=>person(footBox(20+35*i,y*180,.02+.1*y),.8,'p'+i));
 const e=y=>.02+.1*y,lying=person({x:.4,y:.62-.025,w:e(.62)/base.aspect,h:.025},.8,'lying'),diving=person({x:.2,y:.5-.035,w:e(.5)*1.1/base.aspect,h:.035},.8,'dive');
 const r=assessDetections([...people,lying,diving],base,defaultFieldFilter);
 assert.equal(r.size.reliable,true);
 assert.deepEqual(r.rejected,[]);
 assert.ok(r.accepted.find(d=>d.id==='lying').sizeRatio>.8);
});

test('marginFilter: the touchline tolerance slider scales floor and ceiling',()=>{
 const {marginFilter}=pitch;
 assert.deepEqual(marginFilter(defaultFieldFilter.boundaryMargin),defaultFieldFilter);
 const line=nearY(160)+.5,tall=footBox(160,line+6,.12),small=footBox(160,line+6,.05);
 assert.equal(zoneOf(base,tall).zone,'outside');
 assert.equal(zoneOf(base,tall,marginFilter(.6)).zone,'boundary');
 assert.equal(zoneOf(base,small,marginFilter(.6)).zone,'outside');
 assert.equal(zoneOf(base,small,marginFilter(1)).zone,'boundary');
 assert.equal(zoneOf(base,footBox(160,line+1,.12),marginFilter(0)).zone,'outside','0% means no tolerance');
});

// ---------- homography ----------
const H0=[1.2,.1,-.1,.05,1.5,-.2,.1,.3,1];
const apply0=p=>applyHomography(H0,p);
const close=(a,b,eps=1e-6)=>Math.abs(a.x-b.x)<eps&&Math.abs(a.y-b.y)<eps;
const probe=[{x:.3,y:.6},{x:.9,y:.1},{x:.5,y:.5}];

test('solveHomography: exact DLT round trip from 4 and more points',()=>{
 const src4=[{x:.1,y:.1},{x:.9,y:.15},{x:.85,y:.9},{x:.05,y:.8}];
 const h4=solveHomography(src4,src4.map(apply0));
 assert.ok(h4);
 for(const p of probe)assert.ok(close(applyHomography(h4,p),apply0(p)));
 const src8=[...src4,{x:.5,y:.4},{x:.3,y:.7},{x:.7,y:.2},{x:.6,y:.65}];
 const h8=solveHomography(src8,src8.map(apply0));
 for(const p of probe)assert.ok(close(applyHomography(h8,p),apply0(p)));
 // noisy observations still fit closely
 const noisy=src8.map(apply0).map((p,i)=>({x:p.x+((i*37)%7-3)*3e-4,y:p.y+((i*53)%5-2)*3e-4}));
 const hn=solveHomography(src8,noisy);
 assert.ok(hn);for(const p of probe)assert.ok(close(applyHomography(hn,p),apply0(p),.01));
 // w is normalized positive over the source points
 for(const p of src8)assert.ok(h8[6]*p.x+h8[7]*p.y+h8[8]>0);
 // inverse round trip
 const inv=invertHomography(h8);
 for(const p of probe)assert.ok(close(applyHomography(inv,applyHomography(h8,p)),p,1e-9));
});

test('solveHomography rejects degenerate, folded and inconsistent sets',()=>{
 const sq=[{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}];
 assert.equal(solveHomography(sq.slice(0,3),sq.slice(0,3)),undefined);
 assert.equal(solveHomography([{x:0,y:0},{x:.5,y:.5},{x:1,y:1},{x:0,y:1}],sq),undefined); // 3 collinear
 assert.equal(solveHomography(sq,[{x:0,y:0},{x:.5,y:0},{x:1,y:0},{x:0,y:1}]),undefined); // 3 collinear targets
 assert.equal(solveHomography([0,1,2,3,4,5].map(i=>({x:i/5,y:i/10})),[0,1,2,3,4,5].map(i=>({x:i,y:i*i}))),undefined); // all collinear
 assert.equal(solveHomography([{x:0,y:0},{x:0,y:0},{x:1,y:1},{x:0,y:1}],sq),undefined); // duplicate
 assert.equal(solveHomography(sq,[{x:0,y:0},{x:1,y:0},{x:0,y:1},{x:1,y:1}]),undefined); // bow-tie fold
 assert.equal(solveHomography([...sq,{x:NaN,y:0}],[...sq,{x:0,y:0}]),undefined);
 const src=[...sq,{x:.5,y:.2},{x:.3,y:.7}],dst=src.map(apply0);dst[5]={x:dst[5].x+.5,y:dst[5].y-.4}; // gross outlier
 assert.equal(solveHomography(src,dst),undefined);
 assert.ok(solveHomography(src,dst,10)); // accepted when the caller tolerates the error
 assert.equal(invertHomography([1,2,3,2,4,6,0,0,1]),undefined);
 assert.equal(applyHomography([1,0,0,0,1,0,1,0,0],{x:0,y:.5}),undefined); // w = 0
});

// Camera: pitch (normalized) -> image (normalized) for a broadcast-like trapezoid.
const imageCorners=[{x:.2,y:.3},{x:.8,y:.3},{x:.95,y:.85},{x:.05,y:.85}];
const pitchCorners=[{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}];
const G=solveHomography(pitchCorners,imageCorners);
const img=(x,y)=>applyHomography(G,{x,y});
const names=['far-left','far-right','near-right','near-left'];
const cornerMarks=(t=5,pts=imageCorners)=>names.map((name,i)=>({id:'m'+i,name,time:t,points:[pts[i]]}));
const lineMarks=t=>[
 {id:'l1',name:'far-touchline',time:t,points:[img(.3,0),img(.7,0)]},
 {id:'l2',name:'near-touchline',time:t,points:[img(.2,1),img(.9,1)]},
 {id:'l3',name:'left-goal-line',time:t,points:[img(0,.2),img(0,.8)]},
 {id:'l4',name:'right-goal-line',time:t,points:[img(1,.3),img(1,.7)]},
];
const boxAt=p=>({x:p.x-.01,y:p.y-.08,w:.02,h:.08});

test('pitchHomography from corner marks and toPitch',()=>{
 const r=pitchHomography(cornerMarks(),5,[]);
 assert.ok(r);assert.equal(r.points,4);assert.ok(r.error<1e-9);
 for(const [x,y] of [[.5,.5],[.1,.9],[.75,.2]])assert.ok(close(toPitch(r.h,boxAt(img(x,y))),{x,y}));
 assert.ok(close(toPitch(r.h,boxAt(img(1.1,.5))),{x:1.1,y:.5}));
 assert.equal(toPitch(r.h,boxAt(img(.5,1.5))),undefined); // far outside the pitch
 assert.equal(toPitch(r.h,boxAt(img(-.4,.5))),undefined);
 assert.equal(toPitch(r.h,{x:.49,y:0,w:.02,h:.05}),undefined); // stands above the far touchline
 // the closest mark in time is used for each name
 const stale={id:'old',name:'far-left',time:60,points:[{x:.6,y:.6}]};
 assert.ok(close(toPitch(pitchHomography([stale,...cornerMarks()],5,[]).h,boxAt(img(.5,.5))),{x:.5,y:.5}));
 assert.equal(pitchHomography(cornerMarks().slice(0,3),5,[]),undefined);
 // swapped left/right labels would mirror the pitch
 const mirrored=cornerMarks(5,[imageCorners[1],imageCorners[0],imageCorners[3],imageCorners[2]]);
 assert.equal(pitchHomography(mirrored,5,[]),undefined);
});

test('pitchHomography from boundary line intersections and mixed marks',()=>{
 const r=pitchHomography(lineMarks(5),5,[]);
 assert.ok(r);assert.equal(r.points,4);
 for(const [x,y] of [[.5,.5],[.2,.3],[.9,.95]])assert.ok(close(toPitch(r.h,boxAt(img(x,y))),{x,y},1e-6));
 const mixed=pitchHomography([...cornerMarks().slice(0,2),...lineMarks(5).filter(m=>m.name!=='far-touchline')],5,[]);
 assert.ok(mixed);assert.ok(close(toPitch(mixed.h,boxAt(img(.4,.6))),{x:.4,y:.6},1e-6));
 // only two lines: not enough corners
 assert.equal(pitchHomography(lineMarks(5).slice(0,2),5,[]),undefined);
});

test('pitchHomography projects marks through the camera chain',()=>{
 const cameras=[{time:0,segment:'s1',scale:1,dx:0,dy:0},{time:2,segment:'s1',scale:1.1,dx:.02,dy:-.01},{time:4,segment:'s2',scale:1,dx:0,dy:0}];
 const r=pitchHomography(cornerMarks(0),2,cameras);
 assert.ok(r);
 const cam=p=>({x:p.x*1.1+.02,y:p.y*1.1-.01});
 assert.ok(close(toPitch(r.h,boxAt(cam(img(.5,.5)))),{x:.5,y:.5},1e-6));
 assert.equal(pitchHomography(cornerMarks(0),4,cameras),undefined); // different camera segment
 assert.equal(pitchHomography(cornerMarks(0),3,cameras),undefined); // no camera estimate at that time
});
