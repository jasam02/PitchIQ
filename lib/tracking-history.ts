import type {Point,TrackingDoc} from './tracking';
import type {SoccerObservation} from './soccer-schema';
export type HistorySegment=NonNullable<TrackingDoc['historySegments']>[number];
export function validHistoryPoint(doc:TrackingDoc,segment:HistorySegment,point:Point){
 return !(doc.historyEdits||[]).some(edit=>(edit.playerId==='*'||edit.playerId===point.playerId)&&edit.generation>segment.generation&&point.source==='experimental'&&point.time>=edit.time-.125);
}
export function validObservation(doc:TrackingDoc,segment:HistorySegment,point:SoccerObservation){
 return !(doc.historyEdits||[]).some(edit=>(edit.playerId==='*'||edit.playerId===point.globalId)&&edit.generation>segment.generation&&point.time>=edit.time-.125);
}
export function mergeHistoryPoints(...groups:Point[][]){const unique=new Map<string,Point>();for(const group of groups)for(const p of group){const key=p.playerId+':'+p.time.toFixed(4),prior=unique.get(key);if(!prior||p.source!=='experimental'||prior.source==='experimental')unique.set(key,p);}return [...unique.values()];}
