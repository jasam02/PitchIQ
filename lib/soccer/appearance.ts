// STUB — contract only. Implemented by the appearance agent.
import type {Box,Descriptor,Frame,GallerySample,PartDistance} from './types';
export const DESCRIPTOR_LENGTH=0; // length of encodeDescriptor output (<=80)
export const MIN_GALLERY_QUALITY=.45;
// `others` are neighbouring person boxes used to estimate occlusion.
export function describe(frame:Frame,box:Box,others:Box[]=[]):Descriptor{throw Error('not implemented');}
export function encodeDescriptor(d:Descriptor):number[]{throw Error('not implemented');}
export function decodeDescriptor(v:number[]):Descriptor|undefined{throw Error('not implemented');}
export function descriptorDistance(a:Descriptor,b:Descriptor):PartDistance{throw Error('not implemented');}
// Quality- and diversity-aware gallery update. Low-quality samples are ignored.
export function addToGallery(gallery:GallerySample[],d:Descriptor,time:number,max=5):GallerySample[]{throw Error('not implemented');}
// Robust comparison of several probe descriptors against several gallery samples.
export function galleryDistance(gallery:GallerySample[],probes:Descriptor[]):PartDistance|undefined{throw Error('not implemented');}
