// Short-term (local) multi-object tracking. Wraps stepTracks from persistent-tracker (Hungarian, appearance
// gate, tie guard, coasting <= 1.4 s). Local ids ('L1','L2',… exposed as numbers) are temporary and never
// reused; who a local track is gets decided by the global identity manager (identity.ts), never here.
import {iou} from '../detection-core';
import {startTrack,stepTracks,type PersistentTrack} from '../persistent-tracker';
import type {Box,CameraMotion,FieldZone,TrackDetection} from './types';

export type LocalDetection=TrackDetection&{zone?:FieldZone};
export type LocalSample<D extends LocalDetection=LocalDetection>={id:number;box:Box;score:number;evidence:'detection'|'predicted';detection?:D;spawned:boolean;recovered:boolean};
export type LocalEnd={id:number;box:Box;lastSeen:number;cut:boolean};
export type LocalStep<D extends LocalDetection=LocalDetection>={samples:LocalSample<D>[];ended:LocalEnd[];spawned:number[]};
export type LocalTrackerOptions={
 spawnScore:number; // only confident detections start a track (weaker ones only continue existing tracks)
 spawnOverlap:number; // no new track over an existing or predicted one (IoU above this)
 spawnZones:FieldZone[]; // zones in which a detection may start a track (zone-less detections always may)
 appearanceRate:number; // EMA rate refreshing a track's shirt feature from isolated detections
 maxTracks:number;
};
export const defaultLocalOptions:LocalTrackerOptions={spawnScore:.35,spawnOverlap:.3,spawnZones:['inside','boundary'],appearanceRate:.2,maxTracks:80};
export const localKey=(id:number)=>'L'+id;
export function localId(key:string){const n=Number(key.slice(1));return key[0]==='L'&&Number.isInteger(n)?n:-1;}

export class LocalTracker<D extends LocalDetection=LocalDetection>{
 tracks:PersistentTrack[]=[];
 readonly options:LocalTrackerOptions;
 private next=1;
 private occludedAtSpawn=new Set<string>(); // shirt feature taken while overlapping someone: replace it once isolated
 constructor(options:Partial<LocalTrackerOptions>={}){this.options={...defaultLocalOptions,...options};}
 // detections: accepted (may start tracks). continuation: only extend existing tracks (e.g. a confirmed player
 // taking a throw-in just beyond the touchline); they never start a track.
 step(detections:D[],time:number,camera:CameraMotion,continuation:D[]=[]):LocalStep<D>{
  const all=[...detections,...continuation].filter(d=>d.kind==='person'),before=this.tracks;
  const r=stepTracks(before,all,time,camera),byBox=new Map<Box,D>(),used=new Set<D>();
  for(const d of all)if(!byBox.has(d.box))byBox.set(d.box,d);
  const samples:LocalSample<D>[]=[];
  for(const s of r.samples){
   const id=localId(s.id);if(id<0)continue;
   const d=s.evidence==='detection'?byBox.get(s.box):undefined;if(d)used.add(d);
   samples.push({id,box:s.box,score:s.score,evidence:s.evidence,detection:d,spawned:false,recovered:s.recovered});
  }
  const alive=new Set(r.tracks.map(t=>t.id)),ended:LocalEnd[]=[];
  for(const t of before)if(!alive.has(t.id)){ended.push({id:localId(t.id),box:t.box,lastSeen:t.lastSeen,cut:camera.cut});this.occludedAtSpawn.delete(t.id);}
  this.tracks=r.tracks;
  // Refresh the gate's shirt feature slowly, only from isolated detections (never while two people overlap).
  const rate=this.options.appearanceRate,byId=new Map(this.tracks.map(t=>[localId(t.id),t]));
  for(const s of samples){
   const d=s.detection,t=byId.get(s.id);if(!d||!t||!d.appearance?.length||rate<=0)continue;
   if(all.some(o=>o!==d&&iou(o.box,d.box)>.02))continue;
   const fresh=this.occludedAtSpawn.delete(t.id);
   t.appearance=!fresh&&t.appearance.length===d.appearance.length?t.appearance.map((v,i)=>v*(1-rate)+d.appearance![i]*rate):d.appearance.slice();
  }
  const spawned:number[]=[];
  for(const d of detections){
   if(used.has(d)||d.kind!=='person'||!(d.score>=this.options.spawnScore))continue;
   if(d.zone&&!this.options.spawnZones.includes(d.zone))continue;
   if(this.tracks.length>=this.options.maxTracks||this.tracks.some(t=>iou(t.box,d.box)>this.options.spawnOverlap))continue;
   const id=this.spawnTrack(d.box,time,d.appearance);used.add(d);spawned.push(id);
   if(all.some(o=>o!==d&&iou(o.box,d.box)>.02))this.occludedAtSpawn.add(localKey(id));
   samples.push({id,box:d.box,score:d.score,evidence:'detection',detection:d,spawned:true,recovered:false});
  }
  return {samples,ended,spawned};
 }
 // Start a track explicitly (user anchor without a matching track). Returns the new local id.
 spawn(box:Box,time:number,appearance:number[]=[]):number{return this.spawnTrack(box,time,appearance);}
 remove(id:number){this.tracks=this.tracks.filter(t=>t.id!==localKey(id));this.occludedAtSpawn.delete(localKey(id));}
 box(id:number):Box|undefined{return this.tracks.find(t=>t.id===localKey(id))?.box;}
 has(id:number){return this.tracks.some(t=>t.id===localKey(id));}
 reset(){const ids=this.tracks.map(t=>localId(t.id));this.tracks=[];this.occludedAtSpawn.clear();return ids;}
 private spawnTrack(box:Box,time:number,appearance:number[]=[]){const id=this.next++;this.tracks.push(startTrack(localKey(id),'person',box,time,appearance.slice()));return id;}
}
