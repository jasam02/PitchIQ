// STUB — contract only. Implemented by the appearance agent.
import type {Descriptor,FieldZone,KitVote,Role,RoleDecision,RoleEvidence,Team,TeamModel} from './types';
export const emptyTeamModel:TeamModel={keepers:[],anchored:false,spread:.2,samples:0};
// Anchored prototypes when team-labelled samples exist; otherwise 2-means with outlier trimming.
// Keeps the A/B mapping stable relative to `previous`.
export function fitTeamModel(samples:{jersey:number[];team?:Team;role?:Role;weight?:number}[],previous?:TeamModel):TeamModel{throw Error('not implemented');}
export function kitVote(model:TeamModel,d:Descriptor):KitVote{throw Error('not implemented');}
export function emptyRoleEvidence():RoleEvidence{return {hits:0,votes:[],nearGoal:0,central:0,boundary:0};}
export function accumulateRole(e:RoleEvidence,vote:KitVote,context:{nearGoal:boolean;zone:FieldZone}):RoleEvidence{throw Error('not implemented');}
// Temporal decision; returns CANDIDATE until the evidence is strong enough.
export function decideRole(e:RoleEvidence,model:TeamModel):RoleDecision{throw Error('not implemented');}
