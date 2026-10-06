// STUB — contract only. Implemented by the identity agent.
import type {Box,Frame,DebugFrame,ReidEvent,RosterSlot,SavedIdentity,SoccerFrameInput,SoccerFrameResult,SoccerOptions,TrackDetection} from './types';
import {defaultFieldFilter} from './pitch';
export const defaultSoccerOptions:SoccerOptions={field:defaultFieldFilter,maxPerTeam:11,minHits:5,boundaryHits:10,reidMin:.72,reidMargin:.15,gallerySize:5};
// Detection -> field filter -> local tracker -> role/team -> appearance re-ID -> global identities.
export class SoccerTracker{
 constructor(options:Partial<SoccerOptions>={},saved:SavedIdentity[]=[]){}
 // Seed appearance for a roster identity from a user-confirmed label on an earlier frame.
 rememberLabel(playerId:string,team:'A'|'B'|'ref',frame:Frame,box:Box,time:number):void{throw Error('not implemented');}
 step(input:SoccerFrameInput):SoccerFrameResult{throw Error('not implemented');}
 snapshot():SavedIdentity[]{throw Error('not implemented');}
 // Sanity diagnostics (duplicate-looking identities, too many identities per team, etc.).
 diagnostics(time:number):ReidEvent[]{throw Error('not implemented');}
}
// Stateless explanation of one frame for the debug view (no identities are changed).
export function explainFrame(frame:Frame,detections:TrackDetection[],time:number,roster:RosterSlot[],options:Partial<SoccerOptions>={},tracker?:SoccerTracker):DebugFrame{throw Error('not implemented');}
