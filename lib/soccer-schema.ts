import {z} from 'zod';

export const roleSchema=z.enum(['PLAYER_TEAM_A','PLAYER_TEAM_B','GOALKEEPER_TEAM_A','GOALKEEPER_TEAM_B','REFEREE','UNKNOWN']);
export type SoccerRole=z.infer<typeof roleSchema>;
const position=z.object({x:z.number().finite(),y:z.number().finite()});
const box=z.object({x:z.number().finite(),y:z.number().finite(),w:z.number().positive(),h:z.number().positive()});
const confidence=z.number().min(0).max(1);
export const identitySchema=z.object({
 id:z.string().max(80),localId:z.string().max(80).optional(),team:z.enum(['A','B','UNKNOWN']),role:roleSchema,
 state:z.enum(['ACTIVE','MISSING','OFF_FIELD','SUBSTITUTED','UNKNOWN']),manualRole:z.boolean(),
 identityConfidence:confidence,teamConfidence:confidence,roleConfidence:confidence,
 gallery:z.array(z.object({embedding:z.array(z.number().int().min(-127).max(127)).length(512),quality:confidence,time:z.number()})).max(5),
 jersey:z.array(z.number().finite()).max(64),kit:z.array(z.number().finite()).max(150),
 lastSeen:z.number(),box,cameraBox:box.optional(),cameraReliable:z.boolean().optional(),velocity:position,pitch:position.optional(),pitchTime:z.number().optional(),
 history:z.array(z.object({time:z.number(),image:position,pitch:position.optional()})).max(40),
 teamVotes:z.array(z.number().nonnegative()).length(2),roleVotes:z.record(z.string(),z.number()),
});
export type GlobalIdentity=z.infer<typeof identitySchema>;
export const reidEventSchema=z.object({time:z.number(),localId:z.string().max(80),globalId:z.string().max(80),accepted:z.boolean(),reason:z.string().max(160),appearance:confidence,spatial:confidence,trajectory:confidence.optional(),teamMatch:z.boolean().optional(),gap:z.number(),confidence});
export type ReIDEvent=z.infer<typeof reidEventSchema>;
export const soccerSchema=z.object({version:z.literal(1),identities:z.array(identitySchema).max(96),events:z.array(reidEventSchema).max(100),nextLocal:z.number().int().nonnegative(),teams:z.array(z.array(z.number().finite()).max(50)).max(2)});
export type SoccerState=z.infer<typeof soccerSchema>;
export const emptySoccerState=():SoccerState=>({version:1,identities:[],events:[],nextLocal:1,teams:[]});
export type SoccerObservation={time:number;globalId?:string;localId:string;box:z.infer<typeof box>;team:GlobalIdentity['team'];role:SoccerRole;identityConfidence:number;teamConfidence:number;roleConfidence:number;pitch?:{x:number;y:number};uncertain:boolean};
export const observationSchema=z.object({time:z.number().min(0).max(14400),globalId:z.string().max(80).optional(),localId:z.string().max(80),box,team:z.enum(['A','B','UNKNOWN']),role:roleSchema,identityConfidence:confidence,teamConfidence:confidence,roleConfidence:confidence,pitch:position.optional(),uncertain:z.boolean()});
