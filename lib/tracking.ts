import {z} from 'zod';
import type {ReidEvent,RosterSlot,SavedIdentity,SoccerObservation} from './soccer/types';
export const boxSchema=z.object({x:z.number().min(0).max(1),y:z.number().min(0).max(1),w:z.number().min(.001).max(1),h:z.number().min(.001).max(1)}).refine(b=>b.x+b.w<=1.001&&b.y+b.h<=1.001,'Box must fit inside video');
const xy=z.object({x:z.number().min(0).max(1),y:z.number().min(0).max(1)});
const memorySchema=z.object({playerId:z.string().max(80),box:z.object({x:z.number().finite(),y:z.number().finite(),w:z.number().positive(),h:z.number().positive()}),appearance:z.array(z.number().finite()).max(64),time:z.number().min(0).max(14400),lastSeen:z.number().min(0).max(14400),edge:z.string().max(8),offscreen:z.boolean()});
const fieldMarkSchema=z.object({id:z.string().max(80),name:z.enum(['far-left','far-right','near-right','near-left','far-touchline','near-touchline','left-goal-line','right-goal-line']),time:z.number().min(0).max(14400),points:z.array(xy).min(1).max(2)}).refine(m=>m.points.length===(m.name.includes('line')?2:1),'A line needs two endpoints; a corner needs one');
const fieldCameraSchema=z.object({time:z.number().min(0).max(14400),segment:z.string().max(80),scale:z.number().positive().max(100),dx:z.number().finite(),dy:z.number().finite()});
// team 'ref' = match officials (not counted as starters). role is optional; absent means outfield player for A/B.
const playerSchema=z.object({id:z.string().min(1).max(80),name:z.string().min(1).max(80),number:z.string().max(5),team:z.enum(['A','B','ball','ref']),starter:z.boolean(),role:z.enum(['player','goalkeeper','referee']).optional()});
const pointSchema=z.object({playerId:z.string().max(80),time:z.number().min(0).max(14400),box:boxSchema,score:z.number().min(0).max(1),source:z.enum(['manual','detector','experimental']),reviewed:z.boolean(),evidence:z.enum(['detection','predicted','appearance','reidentified']).optional(),track:z.number().int().min(0).max(1e9).optional(),conf:z.number().min(0).max(1).optional(),pitch:z.object({x:z.number().min(-.5).max(1.5),y:z.number().min(-.5).max(1.5)}).optional()});
const finite=z.number().finite(),fxy=z.object({x:finite,y:finite});
export const reidKinds=['reid','reid-rejected','deferred','promotion','new-identity','role','team-change','swap-corrected','swap-uncertain','sanity','anchor'] as const;
const identitySchema=z.object({playerId:z.string().min(1).max(80),role:z.enum(['player','goalkeeper','referee','unknown']),team:z.enum(['A','B']).optional(),status:z.enum(['active','missing','offscreen','unknown','substituted']),gallery:z.array(z.array(finite).max(80)).max(6),lastSeen:z.number().min(0).max(14400),lastBox:z.object({x:finite,y:finite,w:z.number().positive().max(10),h:z.number().positive().max(10)}),lastPitch:fxy.optional(),exitEdge:z.string().max(8),velocity:fxy,identityConfidence:z.number().min(0).max(1),teamConfidence:z.number().min(0).max(1),roleConfidence:z.number().min(0).max(1),anchored:z.boolean()});
const scoresSchema=z.object({appearance:finite,jersey:finite,uniform:finite,team:z.boolean(),spatial:finite,movement:finite,temporal:finite,final:finite,secondBest:finite.optional(),missingSeconds:z.number().min(0).max(14400).optional()});
const logSchema=z.object({time:z.number().min(0).max(14400),kind:z.enum(reidKinds),track:z.number().int().min(0).max(1e9).optional(),playerId:z.string().max(80).optional(),message:z.string().max(240),scores:scoresSchema.optional()});
export const trackingSchema=z.object({version:z.literal(1),teamA:z.string().min(1).max(80),teamB:z.string().min(1).max(80),players:z.array(playerSchema).max(61),points:z.array(pointSchema).max(6000),issues:z.array(z.object({playerId:z.string().max(80),time:z.number().min(0).max(14400),reason:z.string().max(160)})).max(500),substitutions:z.array(z.object({id:z.string().max(80),time:z.number().min(0).max(14400),off:z.string().max(80),on:z.string().max(80)})).max(100),corners:z.array(z.object({x:z.number().min(0).max(1),y:z.number().min(0).max(1)})).max(4),playerMemory:z.array(memorySchema).max(60).optional(),fieldMarks:z.array(fieldMarkSchema).max(200).optional(),fieldCamera:z.array(fieldCameraSchema).max(1800).optional(),checkpoint:z.object({time:z.number().min(0).max(14400),start:z.number().min(0).max(14400)}).optional(),calibrationTime:z.number().min(0).max(14400),identities:z.array(identitySchema).max(40).optional(),trackingLog:z.array(logSchema).max(300).optional()}).superRefine((s,ctx)=>{
 const ids=new Set(s.players.map(p=>p.id)); if(ids.size!==s.players.length)ctx.addIssue({code:'custom',message:'Duplicate player ID'});
 if(s.players.filter(p=>p.team==='ball'&&p.id==='ball'&&p.starter).length!==1||s.players.some(p=>p.team==='ball'&&p.id!=='ball'))ctx.addIssue({code:'custom',message:'One active ball with the ball identifier is required'});
 for(const team of ['A','B'])if(s.players.filter(p=>p.team===team&&p.starter).length>11)ctx.addIssue({code:'custom',message:'At most 11 starters per team'});
 if(s.players.filter(p=>p.team==='ref').length>4)ctx.addIssue({code:'custom',message:'At most 4 match officials'});
 if(s.players.some(p=>p.team==='ball'?!!p.role:p.team==='ref'?!!p.role&&p.role!=='referee':p.role==='referee'))ctx.addIssue({code:'custom',message:'Referee role belongs to match officials only'});
 const active=new Set(s.players.filter(p=>p.starter).map(p=>p.id));
 for(const sub of [...s.substitutions].sort((a,b)=>a.time-b.time)){
 const off=s.players.find(p=>p.id===sub.off),on=s.players.find(p=>p.id===sub.on);
 if(!off||!on||off.team==='ball'||off.team!==on.team||!active.has(sub.off)||active.has(sub.on)||sub.off===sub.on)ctx.addIssue({code:'custom',message:'Invalid substitution order'});
 active.delete(sub.off);active.add(sub.on);
 }
 if(s.playerMemory?.some(m=>!ids.has(m.playerId)||m.playerId==='ball')||new Set(s.playerMemory?.map(m=>m.playerId)).size!==(s.playerMemory?.length||0))ctx.addIssue({code:'custom',message:'Invalid remembered player'});
 if(s.identities?.some(m=>!ids.has(m.playerId)||m.playerId==='ball')||new Set(s.identities?.map(m=>m.playerId)).size!==(s.identities?.length||0))ctx.addIssue({code:'custom',message:'Invalid saved identity'});
 if(s.issues.some(p=>!ids.has(p.playerId)))ctx.addIssue({code:'custom',message:'Tracking issue belongs to a missing player'});
 if(s.points.some(p=>!ids.has(p.playerId)||!isActive(s,p.playerId,p.time)))ctx.addIssue({code:'custom',message:'A label belongs to an inactive or missing player'});
});
export type PlayerMemory=z.infer<typeof memorySchema>;
export type FieldMark=z.infer<typeof fieldMarkSchema>;
export type FieldCamera=z.infer<typeof fieldCameraSchema>;
export type TrackingDoc=z.infer<typeof trackingSchema>;
export type Player=z.infer<typeof playerSchema>;
export type Point=z.infer<typeof pointSchema>;
export type Box=z.infer<typeof boxSchema>;
export function initialTracking(teamA='Team A',teamB='Team B'):TrackingDoc{return {version:1,teamA,teamB,players:[...(['A','B'] as const).flatMap(team=>Array.from({length:11},(_,i)=>({id:team+(i+1),name:team==='A'?`Player ${i+1}`:`Opponent ${i+1}`,number:String(i+1),team,starter:true}))),{id:'ball',name:'Ball',number:'',team:'ball',starter:true}],points:[],issues:[],substitutions:[],corners:[],calibrationTime:0};}
export function isActive(s:{players:Player[];substitutions:{time:number;off:string;on:string}[]},id:string,time:number){let active=s.players.find(p=>p.id===id)?.starter??false;for(const sub of [...s.substitutions].sort((a,b)=>a.time-b.time)){if(sub.time>time+.001)break;if(sub.off===id)active=false;if(sub.on===id)active=true;}return active;}
// Display only nearby observations. Never bridge an unseen interval with a made-up path.
export function pointAt(points:Point[],id:string,time:number):Point|undefined {let best:Point|undefined;let gap=.125;for(const p of points){const d=Math.abs(p.time-time);if(p.playerId===id&&d<=gap){gap=d;best=p;}}return best;}
export function correctPoint(s:TrackingDoc,p:Point):TrackingDoc{return {...s,issues:s.issues.filter(q=>q.playerId!==p.playerId||q.time<p.time-.125),points:[...s.points.filter(q=>q.playerId!==p.playerId||(q.source!=='experimental'?Math.abs(q.time-p.time)>.125:q.time<p.time-.125)),p]};}
export type Identity=z.infer<typeof identitySchema>;
export type TrackingLogEntry=z.infer<typeof logSchema>;
type Roster={players:Player[];substitutions:{time:number;off:string;on:string}[];identities?:{playerId:string;role:string}[]};
// Tactical analysis must use `players` (and goalkeepers when relevant), never referees.
export function splitParticipants(s:Roster){
 const role=(p:Player)=>p.team==='ref'?'referee':p.role??s.identities?.find(i=>i.playerId===p.id)?.role;
 const field=s.players.filter(p=>p.team==='A'||p.team==='B');
 return {players:field.filter(p=>role(p)!=='goalkeeper'&&role(p)!=='referee'),goalkeepers:field.filter(p=>role(p)==='goalkeeper'),referees:s.players.filter(p=>p.team==='ref')};
}
export const officials=[{id:'R1',name:'Referee'},{id:'R2',name:'Assistant referee 1'},{id:'R3',name:'Assistant referee 2'}];
// Adds the referee and both assistants as roster slots R1..R3 when missing. Idempotent; respects roster limits.
export function ensureOfficials<T extends {players:Player[]}>(s:T):T{
 const ids=new Set(s.players.map(p=>p.id));let refs=s.players.filter(p=>p.team==='ref').length,room=61-s.players.length;const add:Player[]=[];
 for(const o of officials)if(!ids.has(o.id)&&refs<4&&room>0){add.push({id:o.id,name:o.name,number:'',team:'ref',starter:true,role:'referee'});refs++;room--;}
 return add.length?{...s,players:[...s.players,...add]}:s;
}
// Roster slots for the soccer identity tracker at `time` (substitutions resolved; ball excluded).
export function soccerRoster(s:Roster,time:number):RosterSlot[]{return s.players.flatMap(p=>p.team==='ball'?[]:[{id:p.id,team:p.team,number:p.number,role:p.team==='ref'?'referee' as const:p.role,active:isActive(s,p.id,time)}]);}
const clamp=(v:number,lo:number,hi:number)=>Number.isFinite(v)?Math.min(hi,Math.max(lo,v)):lo,round=(v:number,d=4)=>Number(clamp(v,-1e6,1e6).toFixed(d));
// Boxes you confirmed (labels and reviewed observations) are anchors. With suggestions, unreviewed roster suggestions
// (kit-colour proposals: experimental 're-identified' boxes without a tracker id) count too, for the start frame.
export const isConfirmed=(p:Point)=>p.source!=='experimental'||p.reviewed;
const isSuggestion=(p:Point)=>p.source==='experimental'&&!p.reviewed&&p.evidence==='reidentified'&&p.track===undefined&&p.conf===undefined;
// Confirmed boxes in (from,to], latest per active player, with the time they were drawn; they override automatic
// identities. Reviewed automatic boxes and suggestions are weak: they bind the person under them, never a new track.
export function confirmedAnchors(s:Roster&{points:Point[]},from:number,to:number,suggestions=false):{playerId:string;box:Box;time:number;weak?:boolean}[]{
 const latest=new Map<string,Point>();
 for(const p of s.points)if((isConfirmed(p)||suggestions&&isSuggestion(p))&&p.playerId!=='ball'&&p.time>from&&p.time<=to&&isActive(s,p.playerId,p.time)&&(latest.get(p.playerId)?.time??-Infinity)<p.time)latest.set(p.playerId,p);
 return [...latest.values()].map(p=>({playerId:p.playerId,box:p.box,time:p.time,...(p.source==='experimental'?{weak:true}:{})}));
}
// Experimental point from an automatic observation: box clipped to the frame, numbers bounded; undefined when unusable.
export function observationPoint(o:{playerId:string;time:number;box:{x:number;y:number;w:number;h:number};score:number;evidence?:Point['evidence'];track?:number;conf?:number;pitch?:{x:number;y:number}}):Point|undefined{
 const unit=(n:number)=>clamp(n,0,1),r=(n:number,d=6)=>Number(n.toFixed(d)),b=o.box,x=unit(b.x),y=unit(b.y),w=Math.min(1,b.x+b.w)-x,h=Math.min(1,b.y+b.h)-y;
 if(!(w>=.001&&h>=.001)||!(o.time>=0&&o.time<=14400)||!o.playerId)return undefined;
 const {track,conf,pitch}=o;
 return {playerId:o.playerId,time:r(o.time,4),box:{x:r(x),y:r(y),w:r(w),h:r(h)},score:r(unit(o.score),4),source:'experimental',reviewed:false,...(o.evidence?{evidence:o.evidence}:{}),...(track!==undefined&&Number.isInteger(track)&&track>=0&&track<=1e9?{track}:{}),...(conf!==undefined&&Number.isFinite(conf)?{conf:r(unit(conf),3)}:{}),...(pitch&&Math.abs(pitch.x-.5)<=1&&Math.abs(pitch.y-.5)<=1?{pitch:{x:r(pitch.x,4),y:r(pitch.y,4)}}:{})};
}
// Points for one soccer tracker step: active roster players only, never within .05 s of a box you confirmed (labels
// and reviewed observations win), at most one per player per 10 ms (`seen` carries keys across steps). Observations
// may be for earlier times (re-identified back-fill, held crossings). boxFor may replace a box (motion smoothing).
export function soccerPoints(s:Roster&{points:Point[]},observations:SoccerObservation[],seen:Set<string>,boxFor?:(o:SoccerObservation)=>Box|undefined):Point[]{
 const ids=new Set(s.players.map(p=>p.id)),mine=s.points.filter(isConfirmed),out:Point[]=[];
 for(const o of observations){
  if(!ids.has(o.playerId)||o.playerId==='ball'||!isActive(s,o.playerId,o.time)||mine.some(q=>q.playerId===o.playerId&&Math.abs(q.time-o.time)<.05))continue;
  const key=o.playerId+':'+Math.round(o.time*100);if(seen.has(key))continue;
  const point=observationPoint({...o,box:boxFor?.(o)??o.box});if(!point)continue;
  seen.add(key);out.push(point);
 }
 return out;
}
// Makes tracker snapshots storable: only roster identities, bounded galleries, finite numbers.
export function cleanIdentities(s:{players:Player[]},list:SavedIdentity[]):Identity[]{
 const ids=new Set(s.players.filter(p=>p.team!=='ball').map(p=>p.id)),seen=new Set<string>(),out:Identity[]=[];
 for(const m of [...list].reverse()){
  if(!ids.has(m.playerId)||seen.has(m.playerId))continue;seen.add(m.playerId);
  const xy=(p:{x:number;y:number})=>({x:round(p.x),y:round(p.y)});
  const next={playerId:m.playerId,role:m.role,team:m.team,status:m.status,gallery:m.gallery.slice(-6).map(v=>v.slice(0,80).map(n=>round(n,3))),lastSeen:clamp(m.lastSeen,0,14400),lastBox:{x:round(m.lastBox.x),y:round(m.lastBox.y),w:clamp(round(m.lastBox.w),.001,10),h:clamp(round(m.lastBox.h),.001,10)},...(m.lastPitch&&Number.isFinite(m.lastPitch.x)&&Number.isFinite(m.lastPitch.y)?{lastPitch:xy(m.lastPitch)}:{}),exitEdge:String(m.exitEdge||'').slice(0,8),velocity:xy(m.velocity||{x:0,y:0}),identityConfidence:round(clamp(m.identityConfidence,0,1),3),teamConfidence:round(clamp(m.teamConfidence,0,1),3),roleConfidence:round(clamp(m.roleConfidence,0,1),3),anchored:!!m.anchored};
  const parsed=identitySchema.safeParse(next);if(parsed.success)out.push(parsed.data);
 }
 return out.reverse().slice(0,40);
}
// Saved identities for a pass starting at `start`: an identity last seen elsewhere in time (a re-run from an earlier
// frame) restarts from its saved box at the start frame, so the tracker has its position to re-identify it.
export function identitiesAt(s:{points:Point[];identities?:Identity[]},start:number):Identity[]{
 return (s.identities||[]).map(i=>{
  const p=Math.abs(i.lastSeen-start)>.3&&i.status!=='substituted'&&i.status!=='unknown'?pointAt(s.points,i.playerId,start):undefined;if(!p)return i;
  const next:Identity={...i,status:'missing',lastSeen:p.time,lastBox:{...p.box},exitEdge:'',velocity:{x:0,y:0}};delete next.lastPitch;return next;
 });
}
// Appends tracker events to the saved log (newest kept), bounding count and message length.
export function appendTrackingLog(log:TrackingLogEntry[]|undefined,events:ReidEvent[],max=300):TrackingLogEntry[]{
 const out=[...(log||[])];
 for(const e of events){
  const sc=e.scores,n=(v:number)=>round(v,4);
  const next={time:clamp(e.time,0,14400),kind:e.kind,...(e.track!==undefined&&Number.isInteger(e.track)&&e.track>=0?{track:e.track}:{}),...(e.playerId?{playerId:e.playerId.slice(0,80)}:{}),message:String(e.message||'').slice(0,240),...(sc?{scores:{appearance:n(sc.appearance),jersey:n(sc.jersey),uniform:n(sc.uniform),team:!!sc.team,spatial:n(sc.spatial),movement:n(sc.movement),temporal:n(sc.temporal),final:n(sc.final),...(sc.secondBest!==undefined?{secondBest:n(sc.secondBest)}:{}),...(sc.missingSeconds!==undefined?{missingSeconds:clamp(round(sc.missingSeconds,2),0,14400)}:{})}}:{})};
  const parsed=logSchema.safeParse(next);if(parsed.success)out.push(parsed.data);
 }
 return max>0?out.slice(-max):[];
}
