// STUB — contract only. Implemented by the pitch agent.
import type {Box,CameraMotion,FieldFilterConfig,FieldZone,Frame,PitchModel,Pt} from './types';
export const defaultFieldFilter:FieldFilterConfig={enabled:true,boundaryMargin:.35,minMargin:.006,maxMargin:.03};
// Detect the playable region on this frame. `previous` (already from an earlier step) lets boundary
// lines that are briefly not visible be carried forward with camera motion for a short time.
export function analyzePitch(frame:Frame,time:number,previous?:PitchModel,camera?:CameraMotion):PitchModel{throw Error('not implemented');}
// Move a model into the next frame's coordinates using the estimated camera motion.
export function warpPitch(model:PitchModel,camera:CameraMotion,time:number):PitchModel{throw Error('not implemented');}
export function footPoint(box:Box):Pt{return {x:box.x+box.w/2,y:box.y+box.h};}
export function pointInPolygon(p:Pt,poly:Pt[]):boolean{throw Error('not implemented');}
// Signed distance in height units (aspect-corrected): negative inside, positive outside.
export function signedDistance(p:Pt,poly:Pt[],aspect:number):number{throw Error('not implemented');}
export function zoneOf(model:PitchModel,box:Box,config:FieldFilterConfig=defaultFieldFilter):{zone:FieldZone;outsideBy:number}{throw Error('not implemented');}
