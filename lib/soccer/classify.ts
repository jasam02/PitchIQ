// Team prototypes from jersey colour, per-observation kit votes, and temporal role decisions.
// Nothing here decides from one frame: decideRole returns CANDIDATE until enough consistent votes exist.
import type {Descriptor,FieldZone,KitVote,Role,RoleDecision,RoleEvidence,Team,TeamModel} from './types';
import {roleLabel} from './types';
import {bhattacharyya} from './appearance';
export const emptyTeamModel:TeamModel={keepers:[],anchored:false,spread:.2,samples:0};
export const KIT_MARGIN=.06; // required lead of the nearer prototype
export const MIN_ROLE_VOTES=5; // default for decideRole (matches defaultSoccerOptions.minHits)
const MAX_VOTES=40,MIN_SAMPLES=6,MIN_KIT_SEPARATION=.35,MIN_KIT_GAP=.3; // kits closer than this are not told apart
// kitVote returns these extra fields; accumulateRole reads them when present (plain KitVotes still work).
export type KitVoteDetail=KitVote&{refLike:boolean;keeperLike:boolean;keeper?:Team;distKeeper:number;valid:boolean};
export type RoleVote=RoleEvidence['votes'][number]&{keeperLike?:boolean;keeper?:Team};
type Sample={jersey:number[];team?:Team;role?:Role;weight?:number};

const r3=(v:number)=>Math.round(v*1000)/1000;
const dist=(a:number[],b:number[])=>bhattacharyya(a,b)??1;
// Jersey histogram without its two brightness levels per hue (4 greys + 10 hues x 2 -> 14 bins): shade and
// floodlights move mass between the levels of one hue, a different kit changes the hue (or the greys).
const hueOnly=(v:number[])=>v.length===24?[...v.slice(0,4),...Array.from({length:10},(_,i)=>(v[4+2*i]||0)+(v[5+2*i]||0))]:v;
const mass=(v:number[])=>v.reduce((a,x)=>a+(x>0?x:0),0);
const median=(v:number[])=>{if(!v.length)return 0;const s=[...v].sort((a,b)=>a-b),m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2;};
const weightOf=(s:Sample)=>s.weight!==undefined&&Number.isFinite(s.weight)&&s.weight>0?s.weight:1;
// Same-kit crops (lighting, pose) stay within ~.4 while other kits sit beyond ~.6, hence the floor and ceiling.
const trimLimit=(d:number[])=>Math.min(.65,Math.max(.3,2.5*median(d)+.1));
function weightedMean(xs:number[][],ws:number[]){
 const n=Math.max(0,...xs.map(x=>x.length)),out=Array(n).fill(0);let total=0;
 xs.forEach((x,i)=>{const s=mass(x);if(s<=0)return;total+=ws[i];for(let k=0;k<n;k++)out[k]+=ws[i]*Math.max(0,x[k]||0)/s;});
 return total?out.map(v=>v/total):out;
}
// Medoid, then the weighted mean of the points near it: the densest group wins, stray labels are ignored.
function robustMean(xs:number[][],ws:number[]){
 if(xs.length<=2)return weightedMean(xs,ws);
 const D=xs.map(a=>xs.map(b=>dist(a,b)));let m=0,best=Infinity;
 D.forEach((row,i)=>{const s=row.reduce((a,v,j)=>a+v*ws[j],0);if(s<best){best=s;m=i;}});
 const limit=trimLimit(D[m]),keep=xs.map((_,i)=>D[m][i]<=limit);
 return weightedMean(xs.filter((_,i)=>keep[i]),ws.filter((_,i)=>keep[i]));
}
// 2-means with per-cluster outlier trimming (referees, keepers, occluded crops do not move a team prototype).
function refine(xs:number[][],ws:number[],c:number[][],fixed:boolean[]){
 let label:number[]=[],inlier:boolean[]=[],d:number[]=[];
 for(let it=0;it<20;it++){
  const nl=xs.map(x=>dist(x,c[0])<=dist(x,c[1])?0:1),nd=xs.map((x,i)=>dist(x,c[nl[i]]));
  const limit=[0,1].map(k=>trimLimit(nd.filter((_,i)=>nl[i]===k))),ni=nd.map((v,i)=>v<=limit[nl[i]]);
  const stable=nl.every((v,i)=>v===label[i]&&ni[i]===inlier[i]);
  label=nl;inlier=ni;d=nd;if(stable)break;
  for(const k of [0,1])if(!fixed[k]){const idx=xs.map((_,i)=>i).filter(i=>label[i]===k&&inlier[i]);if(idx.length)c[k]=weightedMean(idx.map(i=>xs[i]),idx.map(i=>ws[i]));}
 }
 const members=[0,1].map(k=>xs.map((_,i)=>i).filter(i=>label[i]===k&&inlier[i]));
 return {c,members,d,inlier};
}
// Density modes: samples with two close neighbours, densest first, at most one per kit. A lone referee or
// keeper never seeds a team cluster.
function modes(D:number[][],max=5){
 if(D.length<3)return [];
 const knn=D.map((row,i)=>row.filter((_,j)=>j!==i).sort((a,b)=>a-b)[1]),out:number[]=[];
 for(const i of D.map((_,i)=>i).sort((a,b)=>knn[a]-knn[b]||a-b)){
  if(knn[i]>MIN_KIT_GAP||out.length>=max)break;
  if(out.every(j=>D[i][j]>=MIN_KIT_SEPARATION))out.push(i);
 }
 return out;
}
// Two kits leave a gap between the groups; one kit under changing light is a continuum that 2-means would cut.
const kitGap=(members:number[][],D:number[][])=>median(members.flatMap((m,k)=>m.map(i=>Math.min(...members[1-k].map(j=>D[i][j])))));
// The smaller kit must hold a real share of the samples, so a team plus a few officials is never two teams.
const minGroup=(n:number)=>Math.max(n>=10?3:2,Math.ceil(n*.25));
// Floor keeps the vote limit (2.5*spread+.1) >= .3 so lighting changes beyond the sampled range still vote.
function spreadOf(d:number[],fallback:number){return d.length>=4?Math.min(.35,Math.max(.08,median(d))):fallback;}
function fitKeepers(samples:Sample[],previous:TeamModel['keepers']){
 const out:TeamModel['keepers']=[];
 for(const team of ['A','B',undefined] as (Team|undefined)[]){
  const groups:Sample[][]=[];
  for(const s of samples.filter(s=>s.team===team)){const g=groups.find(g=>dist(g[0].jersey,s.jersey)<.3);if(g)g.push(s);else groups.push([s]);}
  for(const g of groups.sort((a,b)=>b.length-a.length).slice(0,team?1:2))out.push({...(team?{team}:{}),jersey:robustMean(g.map(s=>s.jersey),g.map(weightOf))});
 }
 for(const k of previous)if(k.team&&!out.some(o=>o.team===k.team))out.push(k);
 return out.slice(0,4);
}

// Anchored prototypes when team-labelled samples exist; otherwise 2-means with outlier trimming.
// Keeps the A/B mapping stable relative to `previous`. Role hints: role 'referee' samples build the official
// prototype, role 'goalkeeper' samples build keeper prototypes; neither moves a team prototype.
export function fitTeamModel(samples:{jersey:number[];team?:Team;role?:Role;weight?:number}[],previous?:TeamModel):TeamModel{
 const usable=samples.filter(s=>Array.isArray(s.jersey)&&s.jersey.every(Number.isFinite)&&mass(s.jersey)>0);
 const refs=usable.filter(s=>s.role==='referee'),field=usable.filter(s=>s.role!=='referee'&&s.role!=='goalkeeper');
 const referee=refs.length?robustMean(refs.map(s=>s.jersey),refs.map(weightOf)):previous?.referee;
 const keepers=fitKeepers(usable.filter(s=>s.role==='goalkeeper'),previous?.keepers??[]);
 const fallback=():TeamModel=>({...(previous??emptyTeamModel),referee,keepers,samples:field.length});
 const la=field.filter(s=>s.team==='A'),lb=field.filter(s=>s.team==='B');
 if(la.length||lb.length){
  let a=la.length?robustMean(la.map(s=>s.jersey),la.map(weightOf)):undefined,b=lb.length?robustMean(lb.map(s=>s.jersey),lb.map(weightOf)):undefined;
  const free=field.filter(s=>!s.team);
  if(!a||!b){
   // One kit labelled: find the other among unlabelled samples, the labelled prototype stays fixed.
   const known=(a??b)!,xs=free.map(s=>s.jersey),ws=free.map(weightOf);let other:number[]|undefined;
   if(xs.length>=3){
    const D=xs.map(p=>xs.map(q=>dist(p,q)));let best=0;
    for(const seed of modes(D).filter(i=>dist(xs[i],known)>MIN_KIT_GAP)){
     const fit=refine(xs,ws,[known,xs[seed]],[true,false]),m=fit.members[1],score=m.reduce((t,i)=>t+ws[i],0);
     if(m.length>=minGroup(xs.length)&&dist(fit.c[1],known)>=MIN_KIT_SEPARATION&&median(m.map(i=>dist(xs[i],known)))>=MIN_KIT_GAP&&score>best){best=score;other=fit.c[1];}
    }
   }
   const prevOther=a?previous?.b:previous?.a;
   if(!other&&prevOther&&dist(prevOther,known)>=MIN_KIT_SEPARATION)other=prevOther;
   if(a)b=other;else a=other;
  }
  const d:number[]=[];
  for(const s of field){
   const da=a?dist(s.jersey,a):1,db=b?dist(s.jersey,b):1;
   if(s.team)d.push(s.team==='A'?da:db);else if(Math.abs(da-db)>=KIT_MARGIN&&Math.min(da,db)<.6)d.push(Math.min(da,db));
  }
  return {a,b,referee,keepers,anchored:true,spread:spreadOf(d,previous?.spread??emptyTeamModel.spread),samples:field.length};
 }
 if(field.length<MIN_SAMPLES)return fallback();
 const pick=field.length>200?field.filter((_,i)=>i%Math.ceil(field.length/200)===0):field;
 const xs=pick.map(s=>s.jersey),ws=pick.map(weightOf),D=xs.map(p=>xs.map(q=>dist(p,q))),seeds=modes(D),need=minGroup(xs.length);
 // Try every pair of density modes; keep the valid split that explains the most samples.
 let best:{fit:ReturnType<typeof refine>;spread:number;score:number}|undefined;
 for(let x=0;x<seeds.length;x++)for(let y=x+1;y<seeds.length;y++){
  const fit=refine(xs,ws,[xs[seeds[x]],xs[seeds[y]]],[false,false]),spread=spreadOf(fit.d.filter((_,i)=>fit.inlier[i]),emptyTeamModel.spread);
  if(fit.members.some(m=>m.length<need)||dist(fit.c[0],fit.c[1])<Math.max(MIN_KIT_SEPARATION,2*spread+.1)||kitGap(fit.members,D)<Math.max(MIN_KIT_GAP,1.5*spread))continue;
  const score=fit.members.flat().reduce((t,i)=>t+ws[i],0);
  if(!best||score>best.score+1e-9)best={fit,spread,score};
 }
 if(!best)return fallback();
 const {fit,spread}=best;
 // Without history, A is the cluster holding the earliest sample; with history, nearest-centroid matching.
 let [a,b]=fit.members[0][0]<fit.members[1][0]?[fit.c[0],fit.c[1]]:[fit.c[1],fit.c[0]];
 if(previous?.a&&previous.b){
  if(dist(a,previous.b)+dist(b,previous.a)<dist(a,previous.a)+dist(b,previous.b))[a,b]=[b,a];
  // Both clusters are shades of one known kit (the other team left the view, part of the pitch in shadow): a
  // lighting split, not two teams. The established model stays.
  const hue=(x:number[],y:number[])=>dist(hueOnly(x),hueOnly(y)),pa=previous.a,pb=previous.b,near=(x:number[])=>hue(x,pa)<=hue(x,pb)?pa:pb;
  if(near(a)===near(b)&&Math.max(hue(a,near(a)),hue(b,near(b)))<MIN_KIT_GAP)return fallback();
 }
 return {a,b,referee,keepers,anchored:false,spread,samples:field.length};
}

// team: nearer prototype when it leads by KIT_MARGIN and is within 2.5*spread+.1. outlier: far from both kits,
// or clearly closer to the official / a keeper prototype (refLike / keeperLike). Ties stay undecided.
export function kitVote(model:TeamModel,d:Descriptor):KitVoteDetail{
 const has=Array.isArray(d.jersey)&&mass(d.jersey)>0,to=(p?:number[])=>p&&has?dist(d.jersey,p):1;
 const distA=to(model.a),distB=to(model.b),distRef=to(model.referee);
 let distKeeper=1,keeper:Team|undefined;for(const k of model.keepers){const x=to(k.jersey);if(x<distKeeper){distKeeper=x;keeper=k.team;}}
 const vote:KitVoteDetail={distA:r3(distA),distB:r3(distB),distRef:r3(distRef),distKeeper:r3(distKeeper),outlier:false,margin:r3(Math.abs(distA-distB)),refLike:false,keeperLike:false,valid:false};
 if(!has||!model.a||!model.b)return {...vote,margin:0};
 vote.valid=true;
 const limit=2.5*model.spread+.1,near=Math.min(distA,distB),special=Math.min(distRef,distKeeper);
 if(special<limit&&special<near+KIT_MARGIN){
  if(special+KIT_MARGIN<=near){vote.outlier=true;if(distRef<=distKeeper)vote.refLike=true;else{vote.keeperLike=true;if(keeper)vote.keeper=keeper;}}
 }else if(near>=limit){
  // A darker or brighter crop of one team's hue (stadium shadow) is that team, unless an official/keeper kit fits as well.
  const h=(p?:number[])=>p?dist(hueOnly(d.jersey),hueOnly(p)):1,ha=h(model.a),hb=h(model.b),hs=Math.min(h(model.referee),...model.keepers.map(k=>h(k.jersey)));
  if(Math.min(ha,hb)<limit&&Math.abs(ha-hb)>=KIT_MARGIN&&hs>=Math.min(ha,hb)+KIT_MARGIN){vote.team=ha<hb?'A':'B';vote.margin=r3(Math.abs(ha-hb));}
  else vote.outlier=true;
 }else if(Math.abs(distA-distB)>=KIT_MARGIN)vote.team=distA<distB?'A':'B';
 return vote;
}
export function emptyRoleEvidence():RoleEvidence{return {hits:0,votes:[],nearGoal:0,central:0,boundary:0};}
// Returns a new evidence object; votes from an empty/missing model or jersey count as hits but not as votes.
export function accumulateRole(e:RoleEvidence,vote:KitVote,context:{nearGoal:boolean;zone:FieldZone}):RoleEvidence{
 const v=vote as KitVote&Partial<KitVoteDetail>,votes=e.votes.slice();
 if(v.valid??(!!vote.team||vote.outlier||vote.distA<1||vote.distB<1)){
  const refLike=v.refLike??(vote.outlier&&vote.distRef<.35&&vote.distRef+KIT_MARGIN<=Math.min(vote.distA,vote.distB));
  const record:RoleVote={team:vote.team,outlier:vote.outlier,margin:vote.margin,refLike,keeperLike:!!v.keeperLike,...(v.keeper?{keeper:v.keeper}:{})};
  votes.push(record);if(votes.length>MAX_VOTES)votes.splice(0,votes.length-MAX_VOTES);
 }
 return {hits:e.hits+1,votes,nearGoal:e.nearGoal+(context.nearGoal?1:0),central:e.central+(context.zone==='inside'&&!context.nearGoal?1:0),boundary:e.boundary+(context.zone==='boundary'?1:0)};
}

// Temporal decision; returns CANDIDATE until the evidence is strong enough.
// PLAYER: >=70% of votes for one kit with mean margin >= .05. Non-player roles need an outlier majority (>=60%):
// REFEREE when ref-like (or a central / boundary runner in no kit), GOALKEEPER when repeatedly near a goal or in a
// keeper kit. A goalkeeper's team comes only from keeper prototypes in the model; otherwise it stays undefined.
export function decideRole(e:RoleEvidence,model:TeamModel,minVotes=MIN_ROLE_VOTES):RoleDecision{
 const votes=e.votes as RoleVote[],n=votes.length,min=Math.max(3,minVotes); // never from one or two votes
 const support=n>=min?Math.min(1,.75+.25*(n-min)/min):.75*n/min;
 const count=(f:(v:RoleVote)=>boolean)=>votes.filter(f).length;
 const nA=count(v=>v.team==='A'),nB=count(v=>v.team==='B'),nO=count(v=>v.outlier);
 const candidate=():RoleDecision=>({label:'CANDIDATE',role:'unknown',teamConfidence:n?r3(Math.max(nA,nB)/n*support):0,roleConfidence:n?r3(Math.max(nA+nB,nO)/n*support):0});
 if(n<min)return candidate();
 for(const team of ['A','B'] as Team[]){
  const mine=votes.filter(v=>v.team===team),share=mine.length/n,margin=mine.reduce((s,v)=>s+v.margin,0)/Math.max(1,mine.length);
  if(share>=.7&&margin>=.05)return {label:roleLabel('player',team),role:'player',team,teamConfidence:r3(share*support),roleConfidence:r3((nA+nB)/n*support)};
 }
 const outShare=nO/n;
 if(outShare<.6)return candidate();
 const out=votes.filter(v=>v.outlier),refShare=count(v=>v.outlier&&v.refLike)/nO,keeperShare=count(v=>v.outlier&&!!v.keeperLike)/nO;
 const hits=Math.max(1,e.hits),goal=e.nearGoal/hits,central=e.central/hits,boundary=e.boundary/hits;
 const referee=(c:number):RoleDecision=>({label:'REFEREE',role:'referee',teamConfidence:0,roleConfidence:r3(Math.min(1,c)*support)});
 if(refShare>=.5)return referee(outShare*(.5+.5*refShare));
 if((goal>=.5||keeperShare>=.6)&&refShare<.3){
  const keeperVotes=out.filter(v=>v.keeper),teams=(['A','B'] as Team[]).map(t=>({t,k:keeperVotes.filter(v=>v.keeper===t).length})).sort((x,y)=>y.k-x.k);
  const top=teams[0],team=top.k&&top.k>=.6*keeperVotes.length&&top.k>=.5*nO&&model.keepers.some(k=>k.team===top.t)?top.t:undefined;
  return {label:roleLabel('goalkeeper',team),role:'goalkeeper',...(team?{team}:{}),teamConfidence:team?r3(top.k/nO*support):0,roleConfidence:r3(outShare*Math.max(goal,keeperShare)*support)};
 }
 // A person in neither kit who runs the middle or the touchline is an official, but less certain without a prototype.
 if(model.a&&model.b&&(central>=.5||boundary>=.5)&&goal<.5&&keeperShare<.3)return referee(outShare*.8);
 return candidate();
}
