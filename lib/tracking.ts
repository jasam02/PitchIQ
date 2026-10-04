import {z} from 'zod';
import {soccerSchema,roleSchema,observationSchema} from './soccer-schema';
export const boxSchema=z.object({x:z.number().min(0).max(1),y:z.number().min(0).max(1),w:z.number().min(.001).max(1),h:z.number().min(.001).max(1)}).refine(b=>b.x+b.w<=1.001&&b.y+b.h<=1.001,'Box must fit inside video');
const xy=z.object({x:z.number().min(0).max(1),y:z.number().min(0).max(1)});
const memorySchema=z.object({playerId:z.string().max(80),box:z.object({x:z.number().finite(),y:z.number().finite(),w:z.number().positive(),h:z.number().positive()}),appearance:z.array(z.number().finite()).max(64),time:z.number().min(0).max(14400),lastSeen:z.number().min(0).max(14400),edge:z.string().max(8),offscreen:z.boolean()});
const fieldMarkSchema=z.object({id:z.string().max(80),name:z.enum(['far-left','far-right','near-right','near-left','far-touchline','near-touchline','left-goal-line','right-goal-line']),time:z.number().min(0).max(14400),points:z.array(xy).min(1).max(2)}).refine(m=>m.points.length===(m.name.includes('line')?2:1),'A line needs two endpoints; a corner needs one');
export const fieldCameraSchema=z.object({time:z.number().min(0).max(14400),segment:z.string().max(80),scale:z.number().positive().max(100),dx:z.number().finite(),dy:z.number().finite()});
const playerSchema=z.object({id:z.string().min(1).max(80),name:z.string().min(1).max(80),number:z.string().max(5),team:z.enum(['A','B','ball']),starter:z.boolean(),role:roleSchema.optional()});
export const pointSchema=z.object({playerId:z.string().max(80),time:z.number().min(0).max(14400),box:boxSchema,score:z.number().min(0).max(1),source:z.enum(['manual','detector','experimental']),reviewed:z.boolean(),localId:z.string().max(80).optional(),identityConfidence:z.number().min(0).max(1).optional(),pitch:xy.optional(),role:roleSchema.optional(),evidence:z.enum(['detection','predicted','appearance','reidentified']).optional()});
export const trackingSchema=z.object({jerseyFilter:z.object({enabled:z.boolean()}).optional(),soccer:soccerSchema.optional(),observations:z.array(observationSchema).max(4000).optional(),historySegments:z.array(z.object({id:z.string().uuid(),from:z.number(),to:z.number(),generation:z.number().int().min(0)})).max(6000).optional(),historyGeneration:z.number().int().min(0).optional(),historyEdits:z.array(z.object({playerId:z.string().max(80),time:z.number(),generation:z.number().int().min(0)})).max(6000).optional(),version:z.literal(1),teamA:z.string().min(1).max(80),teamB:z.string().min(1).max(80),players:z.array(playerSchema).max(61),points:z.array(pointSchema).max(6000),issues:z.array(z.object({playerId:z.string().max(80),time:z.number().min(0).max(14400),reason:z.string().max(160)})).max(500),substitutions:z.array(z.object({id:z.string().max(80),time:z.number().min(0).max(14400),off:z.string().max(80),on:z.string().max(80)})).max(100),corners:z.array(z.object({x:z.number().min(0).max(1),y:z.number().min(0).max(1)})).max(4),playerMemory:z.array(memorySchema).max(60).optional(),fieldMarks:z.array(fieldMarkSchema).max(200).optional(),fieldCamera:z.array(fieldCameraSchema).max(1800).optional(),checkpoint:z.object({time:z.number().min(0).max(14400),start:z.number().min(0).max(14400)}).optional(),calibrationTime:z.number().min(0).max(14400)}).superRefine((s,ctx)=>{
 const ids=new Set(s.players.map(p=>p.id)); if(ids.size!==s.players.length)ctx.addIssue({code:'custom',message:'Duplicate player ID'});
 if(s.players.filter(p=>p.team==='ball'&&p.id==='ball'&&p.starter).length!==1||s.players.some(p=>p.team==='ball'&&p.id!=='ball'))ctx.addIssue({code:'custom',message:'One active ball with the ball identifier is required'});
 for(const team of ['A','B'])if(s.players.filter(p=>p.team===team&&p.starter).length>11)ctx.addIssue({code:'custom',message:'At most 11 starters per team'});
 const active=new Set(s.players.filter(p=>p.starter).map(p=>p.id));
 for(const sub of [...s.substitutions].sort((a,b)=>a.time-b.time)){
 const off=s.players.find(p=>p.id===sub.off),on=s.players.find(p=>p.id===sub.on);
 if(!off||!on||off.team==='ball'||off.team!==on.team||!active.has(sub.off)||active.has(sub.on)||sub.off===sub.on)ctx.addIssue({code:'custom',message:'Invalid substitution order'});
 active.delete(sub.off);active.add(sub.on);
 }
 if(s.playerMemory?.some(m=>!ids.has(m.playerId)||m.playerId==='ball')||new Set(s.playerMemory?.map(m=>m.playerId)).size!==(s.playerMemory?.length||0))ctx.addIssue({code:'custom',message:'Invalid remembered player'});
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
export function correctPoint(s:TrackingDoc,p:Point):TrackingDoc{const generation=(s.historyGeneration||0)+1;return {...s,historyGeneration:generation,historyEdits:[...(s.historyEdits||[]).filter(e=>e.playerId!==p.playerId||e.time<p.time),{playerId:p.playerId,time:p.time,generation}],observations:s.observations?.filter(o=>o.globalId!==p.playerId||o.time<p.time-.125),soccer:s.soccer?{...s.soccer,identities:s.soccer.identities.filter(g=>g.id!==p.playerId)}:undefined,issues:s.issues.filter(q=>q.playerId!==p.playerId||q.time<p.time-.125),points:[...s.points.filter(q=>q.playerId!==p.playerId||(q.source!=='experimental'?Math.abs(q.time-p.time)>.125:q.time<p.time-.125)),p]};}
