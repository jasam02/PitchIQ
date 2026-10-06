// STUB — contract only. Implemented by the pitch agent.
import type {Box,FieldFilterConfig,PitchModel,RejectedDetection,ScoredDetection,SizeModel,TrackDetection} from './types';
// Robust perspective size model from people standing inside the pitch.
export function fitSizeModel(people:{box:Box}[]):SizeModel{throw Error('not implemented');}
// Split person detections into match-relevant (inside/boundary/unknown zone, plausible size) and
// rejected (with reason). Ball detections are returned untouched in `balls`.
export function assessDetections(detections:TrackDetection[],pitch:PitchModel,config:FieldFilterConfig,size?:SizeModel):{accepted:ScoredDetection[];rejected:RejectedDetection[];balls:TrackDetection[];size:SizeModel}{throw Error('not implemented');}
