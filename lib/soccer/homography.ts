// STUB — contract only. Implemented by the pitch agent.
import type {FieldCamera,FieldMark} from '../tracking';
import type {Box,Pt} from './types';
// 3x3 row-major. Maps normalized image coordinates to normalized pitch coordinates:
// x 0..1 along the touchline (left goal line -> right goal line), y 0..1 far touchline -> near touchline.
export type Homography=number[];
export const PITCH_LENGTH=105,PITCH_WIDTH=68;
export function solveHomography(src:Pt[],dst:Pt[]):Homography|undefined{throw Error('not implemented');}
export function applyHomography(h:Homography,p:Pt):Pt|undefined{throw Error('not implemented');}
export function invertHomography(h:Homography):Homography|undefined{throw Error('not implemented');}
// Build image->pitch homography for `time` from user field marks (corners, and corners implied by
// intersecting named boundary lines), projected with the saved camera chain. Needs 4 non-collinear points.
export function pitchHomography(marks:FieldMark[],time:number,cameras:FieldCamera[]):{h:Homography;points:number;error:number}|undefined{throw Error('not implemented');}
// Foot point of a box in pitch coordinates; undefined when it maps far outside the pitch.
export function toPitch(h:Homography,box:Box):Pt|undefined{throw Error('not implemented');}
