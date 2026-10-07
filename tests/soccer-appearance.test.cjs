const test=require('node:test');
const assert=require('node:assert/strict');
const {makeFrame}=require('./helpers/ts-require.cjs');
const ap=require('../lib/soccer/appearance.ts');
const cl=require('../lib/soccer/classify.ts');
const {describe,encodeDescriptor,decodeDescriptor,descriptorDistance,addToGallery,galleryDistance,bhattacharyya,DESCRIPTOR_LENGTH,MIN_GALLERY_QUALITY}=ap;
const {fitTeamModel,kitVote,emptyRoleEvidence,accumulateRole,decideRole,emptyTeamModel}=cl;

// Deterministic noise so every run sees the same pixels.
let seed=12345;
const rnd=()=>{seed=(seed*1103515245+12345)%2147483648;return seed/2147483648;};
const RED={shirt:[200,30,35],shorts:[240,240,240],socks:[200,30,35]};
const BLUE={shirt:[30,60,190],shorts:[20,20,30],socks:[240,240,240]};
const REF={shirt:[235,215,40],shorts:[15,15,15],socks:[15,15,15]};
const KEEPER={shirt:[245,130,20],shorts:[245,130,20],socks:[245,130,20]};
const SKIN=[205,160,130],BOOT=[25,25,25];
const grassAt=x=>Math.floor(x/40)%2?[40,140,50]:[52,162,62];
// Person painted inside its box: head, jersey (with arms), shorts, two socked legs and boots; grass elsewhere.
function personPixel(p,x,y){
 const u=(x-p.x)/p.w,v=(y-p.y)/p.h;if(u<0||u>=1||v<0||v>=1)return null;
 const k=p.kit;let c=null;
 if(v<.15){if(u>.35&&u<.65)c=SKIN;}
 else if(v<.5){if(u>.15&&u<.85)c=k.shirt;}
 else if(v<.7){if(u>.2&&u<.8)c=k.shorts;}
 else if(v<.94){if((u>.25&&u<.45)||(u>.55&&u<.75))c=k.socks;}
 else if((u>.22&&u<.46)||(u>.54&&u<.78))c=BOOT;
 return c&&c.map(ch=>ch*(p.light??1));
}
// players: [{x,y,w,h (pixels), kit, light}], later entries are painted on top (in front).
function scene(W,H,players,noise=10){
 return makeFrame(W,H,(x,y)=>{
  let c=null;for(const p of players){const q=personPixel(p,x,y);if(q)c=q;}
  return (c||grassAt(x,y)).map(ch=>Math.max(0,Math.min(255,ch+(rnd()-.5)*2*noise)));
 });
}
const boxOf=(p,W,H)=>({x:p.x/W,y:p.y/H,w:p.w/W,h:p.h/H});
// One player in a small frame (with a grass margin so the box is not truncated): fast way to get many descriptors.
function solo(kit,light=1,o={}){const W=60,H=120,p={x:10,y:10,w:40,h:100,kit,light};return describe(scene(W,H,[p],o.noise??10),boxOf(p,W,H));}
const proto=kit=>solo(kit).jersey;
const lights=n=>Array.from({length:n},(_,i)=>.75+.35*i/Math.max(1,n-1));
const sum=v=>v.reduce((a,b)=>a+b,0);

// ---------------- Appearance ----------------
test('same kit is close, different kit is far',()=>{
 const W=640,H=360,ps=[{x:40,y:100,w:30,h:80,kit:RED},{x:200,y:120,w:36,h:96,kit:RED},{x:380,y:100,w:30,h:80,kit:BLUE},{x:520,y:110,w:30,h:84,kit:REF}];
 const f=scene(W,H,ps),[r1,r2,b,ref]=ps.map(p=>describe(f,boxOf(p,W,H)));
 const same=descriptorDistance(r1,r2),diff=descriptorDistance(r1,b);
 assert.ok(same.total<.12,`same kit total ${same.total}`);
 assert.ok(same.jersey<.15&&same.shorts<.15&&same.socks<.15,JSON.stringify(same));
 assert.ok(diff.total>.8,`different kit total ${diff.total}`);
 assert.ok(diff.jersey>.9,JSON.stringify(diff));
 assert.ok(descriptorDistance(r1,ref).total>.8);
 assert.equal(descriptorDistance(r1,r1).total,0);
 // Mild shade keeps the same kit much closer than another kit.
 const shade=descriptorDistance(solo(RED,1),solo(RED,.8));
 assert.ok(shade.total<.35&&shade.total<diff.total-.4,JSON.stringify(shade));
 for(const k of ['total','jersey','shorts','socks','layout'])assert.ok(diff[k]>=0&&diff[k]<=1,k);
});

test('descriptor parts and normalisation',()=>{
 const d=solo(RED);
 assert.equal(d.jersey.length,24);assert.equal(d.shorts.length,12);assert.equal(d.socks.length,8);assert.equal(d.layout.length,24);
 for(const part of [d.jersey,d.shorts,d.socks])assert.ok(Math.abs(sum(part)-1)<.01,'L1 normalised');
 // White shorts land in the achromatic bins (first three), red shirt in chromatic bins.
 assert.ok(sum(d.shorts.slice(0,3))>.95,JSON.stringify(d.shorts));
 assert.ok(sum(d.jersey.slice(0,4))<.05,JSON.stringify(d.jersey));
 // Black shorts vs white shorts differ strongly even though both are achromatic.
 assert.ok(bhattacharyya(d.shorts,solo(BLUE).shorts)>.9);
});

test('grass is excluded from the descriptor',()=>{
 const W=320,H=240,p={x:140,y:60,w:30,h:90,kit:RED},f=scene(W,H,[p]);
 const tight=describe(f,boxOf(p,W,H)),wide=describe(f,{x:(p.x-30)/W,y:(p.y-5)/H,w:(p.w+60)/W,h:(p.h+10)/H});
 // Green hue bins (~90-160 degrees) carry no mass even for a box that is mostly grass.
 const greenBins=d=>sum(d.jersey.slice(8,12));
 assert.equal(greenBins(tight),0);assert.equal(greenBins(wide),0);
 assert.ok(descriptorDistance(tight,wide).jersey<.15,JSON.stringify(descriptorDistance(tight,wide)));
 // A box with only grass yields empty parts and zero quality.
 const grass=describe(f,{x:10/W,y:150/H,w:30/W,h:80/H});
 assert.equal(sum(grass.jersey),0);assert.equal(sum(grass.shorts),0);assert.equal(grass.quality,0);
 assert.ok(grass.layout.every(v=>v===-1));
 // Missing parts are neutral rather than counted as a match.
 assert.equal(descriptorDistance(grass,tight).total,.5);
});

test('quality drops for tiny, occluded, truncated and odd boxes',()=>{
 const W=640,H=360;
 const big={x:300,y:100,w:40,h:110,kit:RED},tiny={x:100,y:100,w:7,h:18,kit:RED},edge={x:0,y:150,w:24,h:110,kit:BLUE};
 const behind={x:450,y:100,w:40,h:110,kit:RED},front={x:462,y:112,w:40,h:110,kit:BLUE};
 const f=scene(W,H,[big,tiny,edge,behind,front]);
 const qBig=describe(f,boxOf(big,W,H)).quality,qTiny=describe(f,boxOf(tiny,W,H)).quality;
 const qEdge=describe(f,boxOf(edge,W,H)).quality;
 const bBox=boxOf(behind,W,H),fBox=boxOf(front,W,H);
 const qAlone=describe(f,bBox).quality,qOcc=describe(f,bBox,[fBox]).quality;
 assert.ok(qBig>=.85,`big ${qBig}`);
 assert.ok(qTiny<.35,`tiny ${qTiny}`);
 assert.ok(qTiny<MIN_GALLERY_QUALITY);
 assert.ok(qEdge<qBig*.7,`edge ${qEdge}`);
 assert.ok(qOcc<qAlone*.5,`occluded ${qOcc} vs ${qAlone}`);
 // The person in front loses less than the one behind (only part of the overlap counts).
 assert.ok(describe(f,fBox,[bBox]).quality>qOcc);
 // Passing the box itself (or an identical copy) among `others` is not occlusion.
 assert.equal(describe(f,bBox,[bBox,{...bBox}]).quality,qAlone);
 // Pixels of the occluder are not sampled: the hidden red player's jersey does not turn blue.
 const occ=describe(f,bBox,[fBox]);
 assert.ok(bhattacharyya(occ.jersey,proto(RED))<bhattacharyya(occ.jersey,proto(BLUE)));
 // Lying / squashed box.
 const flat=describe(f,{x:big.x/W,y:big.y/H,w:110/W,h:40/H}).quality;
 assert.ok(flat<qBig*.6,`flat ${flat}`);
 // Degenerate input.
 assert.equal(describe(f,{x:.5,y:.5,w:0,h:.1}).quality,0);
 assert.equal(describe(f,{x:1.2,y:.5,w:.1,h:.1}).quality,0);
});

test('encode / decode round trip',()=>{
 assert.ok(DESCRIPTOR_LENGTH>0&&DESCRIPTOR_LENGTH<=80);
 for(const d of [solo(RED),solo(BLUE,.8),describe(scene(60,120,[]),{x:.2,y:.1,w:.6,h:.8})]){
  const v=encodeDescriptor(d);
  assert.equal(v.length,DESCRIPTOR_LENGTH);
  assert.ok(v.every(Number.isFinite));
  assert.ok(v.every(n=>Math.round(n*1000)/1000===n),'3 decimals');
  const back=decodeDescriptor(v);
  assert.deepEqual(back,d);
  assert.deepEqual(encodeDescriptor(back),v);
  if(d.quality>0)assert.equal(descriptorDistance(back,d).total,0);
 }
 assert.equal(decodeDescriptor([1,2,3]),undefined);
 assert.equal(decodeDescriptor(Array(DESCRIPTOR_LENGTH).fill(NaN)),undefined);
 assert.equal(decodeDescriptor(Array(DESCRIPTOR_LENGTH+1).fill(0)),undefined);
 // JSON (as stored in TrackingDoc.identities) survives too.
 const d=solo(KEEPER);assert.deepEqual(decodeDescriptor(JSON.parse(JSON.stringify(encodeDescriptor(d)))),d);
});

test('gallery keeps at most max samples, ignores low quality, respects spacing',()=>{
 const d=solo(RED);let g=[];
 for(let t=0;t<10;t++)g=addToGallery(g,d,t,5);
 assert.equal(g.length,5);
 assert.ok(g.every((s,i)=>i===0||s.time>g[i-1].time),'sorted by time');
 const low={...d,quality:MIN_GALLERY_QUALITY-.01};
 assert.deepEqual(addToGallery([],low,0),[]);
 assert.equal(addToGallery(g,low,20).length,5);
 assert.ok(!addToGallery(g,low,20).some(s=>s.time===20));
 // Too close in time: ignored unless clearly better, then it replaces the close sample.
 let h=addToGallery([],{...d,quality:.8},0);
 h=addToGallery(h,{...d,quality:.85},.4);
 assert.equal(h.length,1);assert.equal(h[0].time,0);
 h=addToGallery(h,{...d,quality:.95},.5);
 assert.equal(h.length,1);assert.equal(h[0].time,.5);assert.equal(h[0].d.quality,.95);
 h=addToGallery(h,{...d,quality:.6},1.3);
 assert.equal(h.length,2);
 // Between two samples that are both within the spacing: never added.
 h=addToGallery(h,{...d,quality:1},.9);
 assert.equal(h.length,2);
 for(let i=1;i<h.length;i++)assert.ok(h[i].time-h[i-1].time>=.8-1e-9);
 // Input is not mutated.
 const before=JSON.stringify(g);addToGallery(g,d,30);assert.equal(JSON.stringify(g),before);
});

test('gallery prefers quality and diversity when full',()=>{
 const d=solo(RED);
 // Same appearance, different qualities: the lowest-quality sample is replaced.
 let g=[];[.5,.9,.8,.7,.6].forEach((q,i)=>{g=addToGallery(g,{...d,quality:q},i);});
 assert.equal(g.length,5);
 g=addToGallery(g,{...d,quality:.95},10);
 assert.equal(g.length,5);
 assert.ok(!g.some(s=>s.d.quality===.5),'lowest quality dropped');
 assert.ok(g.some(s=>s.time===10));
 // Five near-identical samples plus a new look (shade): the new look is kept, a redundant one goes.
 let r=[];for(let t=0;t<5;t++)r=addToGallery(r,{...solo(RED,1),quality:.9},t);
 const shaded={...solo(RED,.75),quality:.75};
 assert.ok(descriptorDistance(shaded,r[0].d).total>.15);
 r=addToGallery(r,shaded,10);
 assert.equal(r.length,5);
 assert.ok(r.some(s=>s.time===10),'diverse sample kept');
 // A redundant, lower-quality sample does not displace anything.
 const copy=JSON.stringify(r.map(s=>s.time));
 r=addToGallery(r,{...solo(RED,1),quality:.7},20);
 assert.equal(JSON.stringify(r.map(s=>s.time)),copy);
});

test('galleryDistance is robust to one bad probe',()=>{
 let g=[];[1,.9,.85].forEach((l,i)=>{g=addToGallery(g,solo(RED,l),i);});
 assert.equal(g.length,3);
 const good=[solo(RED,.95),solo(RED,1)],bad=solo(BLUE);
 const clean=galleryDistance(g,good),mixed=galleryDistance(g,[...good,bad]);
 assert.ok(clean.total<.15,JSON.stringify(clean));
 assert.ok(mixed.total<.15,JSON.stringify(mixed));
 assert.ok(Math.abs(mixed.total-clean.total)<.02);
 // Naive averaging would have been fooled.
 const naive=[...good,bad].flatMap(p=>g.map(s=>descriptorDistance(s.d,p).total));
 assert.ok(sum(naive)/naive.length>.3);
 // Other kit vs the gallery is far; an occluded (low-quality) probe is skipped when better ones exist.
 assert.ok(galleryDistance(g,[bad]).total>.8);
 assert.ok(galleryDistance(g,[{...bad,quality:.1},...good]).total<.15);
 assert.equal(galleryDistance([],good),undefined);
 assert.equal(galleryDistance(g,[]),undefined);
 // Per-part values are reported too.
 for(const k of ['jersey','shorts','socks','layout'])assert.ok(mixed[k]>=0&&mixed[k]<=1);
});

test('describe stays cheap for a crowded frame',()=>{
 const W=1280,H=720,ps=Array.from({length:30},(_,i)=>({x:20+(i%10)*125,y:80+Math.floor(i/10)*200,w:36+(i%4)*8,h:100+(i%3)*20,kit:i%2?RED:BLUE}));
 const f=scene(W,H,ps,4),boxes=ps.map(p=>boxOf(p,W,H));
 const t=process.hrtime.bigint();
 for(let k=0;k<5;k++)for(const b of boxes)describe(f,b,boxes);
 const ms=Number(process.hrtime.bigint()-t)/1e6/5;
 assert.ok(ms<60,`30 descriptors took ${ms.toFixed(1)} ms`);
});

// ---------------- Teams and roles ----------------
// Field players under varied lighting plus two referees; unlabelled like the automatic refit sees them.
function fieldSamples(){
 const s=[];
 for(const l of lights(9))s.push({jersey:solo(RED,l).jersey});
 for(const l of lights(8))s.push({jersey:solo(BLUE,l).jersey});
 s.splice(4,0,{jersey:solo(REF,1).jersey});s.push({jersey:solo(REF,.85).jersey});
 return s;
}
const near=(a,b,limit=.3)=>bhattacharyya(a,b)<limit;

test('fitTeamModel: unsupervised two kits, referee outliers trimmed',()=>{
 const m=fitTeamModel(fieldSamples());
 assert.equal(m.anchored,false);
 assert.ok(m.a&&m.b);
 assert.ok(near(m.a,proto(RED)),`a ${bhattacharyya(m.a,proto(RED))}`);
 assert.ok(near(m.b,proto(BLUE)),`b ${bhattacharyya(m.b,proto(BLUE))}`);
 // Referee yellow did not leak into either prototype.
 const yellowA=bhattacharyya(m.a,proto(REF)),yellowB=bhattacharyya(m.b,proto(REF));
 assert.ok(yellowA>.85&&yellowB>.85,`${yellowA} ${yellowB}`);
 assert.ok(m.spread>0&&m.spread<.35);
 assert.equal(m.samples,19);
 assert.equal(m.referee,undefined,'no referee prototype without a role hint');
 const v=kitVote(m,solo(REF));
 assert.equal(v.team,undefined);assert.equal(v.outlier,true);
});

test('fitTeamModel: a dense referee trio is not chosen as a team, whatever the order',()=>{
 const base=[...lights(5).map(l=>({jersey:solo(RED,l).jersey})),...lights(4).map(l=>({jersey:solo(BLUE,l).jersey})),...[1,.9,.8].map(l=>({jersey:solo(REF,l).jersey}))];
 for(let r=0;r<4;r++){
  const s=base.map((_,i)=>base[(i*5+r*7)%base.length]);
  assert.equal(new Set(s).size,base.length);
  const m=fitTeamModel(s),kits=[m.a,m.b].map(p=>bhattacharyya(p,proto(RED))<.3?'red':bhattacharyya(p,proto(BLUE))<.3?'blue':'other');
  assert.deepEqual(kits.slice().sort(),['blue','red'],`order ${r}: ${kits}`);
  assert.equal(kitVote(m,solo(REF)).outlier,true);
 }
});

test('fitTeamModel: too few samples or a single kit gives no prototypes',()=>{
 const few=fitTeamModel(lights(5).map(l=>({jersey:solo(l>.9?RED:BLUE,l).jersey})));
 assert.equal(few.a,undefined);assert.equal(few.b,undefined);
 const one=fitTeamModel(lights(12).map(l=>({jersey:solo(RED,l).jersey})));
 assert.equal(one.a,undefined,'one kit is not split into two teams');
 // One team plus a group of officials is not two teams.
 const officials=fitTeamModel([...lights(12).map(l=>({jersey:solo(RED,l).jersey})),...[1,.9,.8].map(l=>({jersey:solo(REF,l).jersey}))]);
 assert.equal(officials.a,undefined);assert.equal(officials.b,undefined);
 // An existing model is kept rather than replaced by nothing.
 const prev=fitTeamModel(fieldSamples());
 const kept=fitTeamModel(lights(3).map(l=>({jersey:solo(RED,l).jersey})),prev);
 assert.deepEqual(kept.a,prev.a);assert.deepEqual(kept.b,prev.b);
 // Empty / malformed jerseys are ignored.
 assert.equal(fitTeamModel([{jersey:[]},{jersey:Array(24).fill(0)}]).samples,0);
});

test('fitTeamModel: anchored model from labelled samples',()=>{
 const unl=fieldSamples();
 const labelled=[{jersey:solo(BLUE,1).jersey,team:'A'},{jersey:solo(BLUE,.9).jersey,team:'A'},{jersey:solo(RED,1).jersey,team:'B'}];
 const m=fitTeamModel([...labelled,...unl]);
 assert.equal(m.anchored,true);
 assert.ok(near(m.a,proto(BLUE)),'labels decide the mapping');
 assert.ok(near(m.b,proto(RED)));
 assert.equal(kitVote(m,solo(RED,.9)).team,'B');
 // A stray wrong label among correct ones does not move the prototype (medoid + trimming).
 const noisy=fitTeamModel([...labelled,{jersey:solo(BLUE,1).jersey,team:'A'},{jersey:solo(REF).jersey,team:'A'}]);
 assert.ok(near(noisy.a,proto(BLUE)));
 // Only one team labelled: the other kit is found among the unlabelled samples.
 const half=fitTeamModel([{jersey:solo(RED).jersey,team:'A'},...unl]);
 assert.equal(half.anchored,true);
 assert.ok(near(half.a,proto(RED)));
 assert.ok(half.b&&near(half.b,proto(BLUE)),'other kit from unlabelled samples');
 // Labelled keepers / referees do not become team prototypes.
 const roles=fitTeamModel([...labelled,{jersey:solo(KEEPER).jersey,team:'A',role:'goalkeeper'},{jersey:solo(REF).jersey,role:'referee'}]);
 assert.ok(near(roles.a,proto(BLUE)));
 assert.ok(roles.referee&&near(roles.referee,proto(REF)));
 assert.equal(roles.keepers.length,1);assert.equal(roles.keepers[0].team,'A');
});

test('fitTeamModel: A/B mapping is stable across refits',()=>{
 const s=fieldSamples(),first=fitTeamModel(s);
 assert.ok(near(first.a,proto(RED)));
 const reversed=s.slice().reverse();
 // Without history the arbitrary order would flip the labels...
 assert.ok(near(fitTeamModel(reversed).a,proto(BLUE)));
 // ...with the previous model the mapping is kept.
 const again=fitTeamModel(reversed,first);
 assert.ok(near(again.a,proto(RED)));assert.ok(near(again.b,proto(BLUE)));
 // Changed lighting in the next window still maps to the same teams.
 const darker=[...lights(7).map(l=>({jersey:solo(BLUE,l*.85).jersey})),...lights(7).map(l=>({jersey:solo(RED,l*.85).jersey}))];
 const next=fitTeamModel(darker,again);
 assert.ok(near(next.a,proto(RED),.4));assert.ok(near(next.b,proto(BLUE),.4));
 // Referee and keeper prototypes carry over when the window has none.
 const withRoles=fitTeamModel([...s,{jersey:solo(REF).jersey,role:'referee'},{jersey:solo(KEEPER).jersey,role:'goalkeeper',team:'B'}]);
 const carried=fitTeamModel(s,withRoles);
 assert.deepEqual(carried.referee,withRoles.referee);
 assert.equal(carried.keepers.length,1);assert.equal(carried.keepers[0].team,'B');
});

test('kitVote: team, outlier and ref-like',()=>{
 const m=fitTeamModel(fieldSamples());
 const a=kitVote(m,solo(RED,.85)),b=kitVote(m,solo(BLUE,.95));
 assert.equal(a.team,'A');assert.equal(a.outlier,false);assert.ok(a.margin>=.06);assert.ok(a.distA<a.distB);
 assert.equal(b.team,'B');
 const ref=kitVote(m,solo(REF));
 assert.equal(ref.outlier,true);assert.equal(ref.team,undefined);assert.equal(ref.refLike,false);
 // With an official prototype the same kit becomes ref-like.
 const withRef=fitTeamModel([...fieldSamples(),{jersey:solo(REF,.95).jersey,role:'referee'}]);
 const r=kitVote(withRef,solo(REF,.9));
 assert.equal(r.refLike,true);assert.equal(r.outlier,true);assert.ok(r.distRef<.3);
 assert.equal(kitVote(withRef,solo(RED)).refLike,false);
 // Keeper kit with a labelled keeper prototype.
 const withKeeper=fitTeamModel([...fieldSamples(),{jersey:solo(KEEPER).jersey,role:'goalkeeper',team:'B'}]);
 const k=kitVote(withKeeper,solo(KEEPER,.9));
 assert.equal(k.keeperLike,true);assert.equal(k.keeper,'B');assert.equal(k.outlier,true);assert.equal(k.team,undefined);
 // Ambiguous: equally close to both kits -> no team.
 const mix=solo(RED);mix.jersey=mix.jersey.map((v,i)=>(v+proto(BLUE)[i])/2);
 const amb=kitVote(m,mix);
 assert.equal(amb.team,undefined);
 // No model / no jersey pixels: an invalid, uninformative vote.
 const none=kitVote(emptyTeamModel,solo(RED));
 assert.equal(none.valid,false);assert.equal(none.team,undefined);assert.equal(none.outlier,false);
 const empty=kitVote(m,{...solo(RED),jersey:Array(24).fill(0)});
 assert.equal(empty.valid,false);assert.equal(empty.outlier,false);
});

// Feeds `n` observations of a descriptor through kitVote + accumulateRole.
function observe(e,model,d,n,context={nearGoal:false,zone:'inside'}){for(let i=0;i<n;i++)e=accumulateRole(e,kitVote(model,d),context);return e;}

test('decideRole: PLAYER_TEAM_A only after enough consistent votes',()=>{
 const m=fitTeamModel(fieldSamples());
 let e=observe(emptyRoleEvidence(),m,solo(RED,.9),1);
 assert.equal(decideRole(e,m).label,'CANDIDATE','never from a single vote');
 assert.equal(decideRole(e,m,1).label,'CANDIDATE','even when asked to');
 e=observe(e,m,solo(RED,.95),3);
 assert.equal(e.hits,4);
 assert.equal(decideRole(e,m).label,'CANDIDATE');
 e=observe(e,m,solo(RED,1),1);
 const d=decideRole(e,m);
 assert.equal(d.label,'PLAYER_TEAM_A');assert.equal(d.role,'player');assert.equal(d.team,'A');
 assert.ok(d.teamConfidence>.6&&d.teamConfidence<=1);
 assert.ok(d.roleConfidence>.6);
 // Confidence grows with more agreeing votes.
 const more=decideRole(observe(e,m,solo(RED,.9),5),m);
 assert.ok(more.teamConfidence>d.teamConfidence);
 // A couple of odd votes (occlusion) do not prevent the decision.
 let f=observe(emptyRoleEvidence(),m,solo(BLUE,.9),8);f=observe(f,m,solo(REF),2);
 assert.equal(decideRole(f,m).label,'PLAYER_TEAM_B');
});

test('decideRole: mixed or uninformative votes stay CANDIDATE',()=>{
 const m=fitTeamModel(fieldSamples());
 let e=emptyRoleEvidence();
 for(let i=0;i<6;i++){e=observe(e,m,solo(RED),1);e=observe(e,m,solo(BLUE),1);}
 const d=decideRole(e,m);
 assert.equal(d.label,'CANDIDATE');assert.equal(d.role,'unknown');assert.equal(d.team,undefined);
 // Votes without a team model are not votes at all.
 const blind=observe(emptyRoleEvidence(),emptyTeamModel,solo(RED),20);
 assert.equal(blind.hits,20);assert.equal(blind.votes.length,0);
 assert.equal(decideRole(blind,emptyTeamModel).label,'CANDIDATE');
 // Evidence is not mutated and the vote history is bounded.
 const before=JSON.stringify(e);accumulateRole(e,kitVote(m,solo(RED)),{nearGoal:false,zone:'inside'});
 assert.equal(JSON.stringify(e),before);
 assert.ok(observe(emptyRoleEvidence(),m,solo(RED),100).votes.length<=40);
});

test('decideRole: REFEREE for ref-like outliers and for central runners in no kit',()=>{
 const m=fitTeamModel([...fieldSamples(),{jersey:solo(REF,.95).jersey,role:'referee'}]);
 const few=observe(emptyRoleEvidence(),m,solo(REF),3);
 assert.equal(decideRole(few,m).label,'CANDIDATE');
 const e=observe(emptyRoleEvidence(),m,solo(REF,.9),6);
 const d=decideRole(e,m);
 assert.equal(d.label,'REFEREE');assert.equal(d.role,'referee');assert.equal(d.team,undefined);
 assert.ok(d.roleConfidence>.5);
 // Ref-like wins even near a goal (referees do go there).
 assert.equal(decideRole(observe(emptyRoleEvidence(),m,solo(REF),6,{nearGoal:true,zone:'inside'}),m).label,'REFEREE');
 // Assistant referee: no referee prototype yet, outlier kit running the touchline.
 const plain=fitTeamModel(fieldSamples());
 const ar=decideRole(observe(emptyRoleEvidence(),plain,solo(REF),8,{nearGoal:false,zone:'boundary'}),plain);
 assert.equal(ar.label,'REFEREE');
 assert.ok(ar.roleConfidence<d.roleConfidence,'positional evidence is weaker than a kit prototype');
});

test('decideRole: GOALKEEPER near goal with a distinct kit',()=>{
 const m=fitTeamModel(fieldSamples());
 const atGoal={nearGoal:true,zone:'inside'};
 const e=observe(emptyRoleEvidence(),m,solo(KEEPER,.9),7,atGoal);
 const d=decideRole(e,m);
 assert.equal(d.role,'goalkeeper');
 assert.equal(d.team,undefined,'team unknown without a keeper prototype');
 assert.equal(d.label,'UNKNOWN');
 assert.equal(d.teamConfidence,0);
 // With a user-labelled keeper prototype the team follows it.
 const mk=fitTeamModel([...fieldSamples(),{jersey:solo(KEEPER).jersey,role:'goalkeeper',team:'A'}]);
 const k=decideRole(observe(emptyRoleEvidence(),mk,solo(KEEPER,.9),7,atGoal),mk);
 assert.equal(k.label,'GOALKEEPER_TEAM_A');assert.equal(k.role,'goalkeeper');assert.equal(k.team,'A');
 assert.ok(k.teamConfidence>.5);
 // A keeper prototype of the other team does not hand out a team to a different keeper kit.
 const PURPLE={shirt:[150,40,200],shorts:[150,40,200],socks:[150,40,200]};
 const other=decideRole(observe(emptyRoleEvidence(),mk,solo(PURPLE),7,atGoal),mk);
 assert.equal(other.role,'goalkeeper');assert.equal(other.team,undefined);
 // Too few observations near the goal: still a candidate.
 assert.equal(decideRole(observe(emptyRoleEvidence(),m,solo(KEEPER),2,atGoal),m).label,'CANDIDATE');
 // An outfield kit near the goal is a player, not a keeper.
 assert.equal(decideRole(observe(emptyRoleEvidence(),m,solo(RED),7,atGoal),m).label,'PLAYER_TEAM_A');
});
