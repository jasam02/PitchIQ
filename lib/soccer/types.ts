// Shared contract for the soccer-specific tracking layer.
// Every module here is DOM-free so it can run in tests, workers and the browser.
// Image coordinates are normalized to [0,1] of the analysed frame (x right, y down).
import type {Box} from '../tracking';
import type {Frame} from '../patch-tracker';
import type {CameraMotion} from '../track-vision';
import type {TrackDetection} from '../persistent-tracker';
export type {Box,Frame,CameraMotion,TrackDetection};

export type Pt={x:number;y:number};

// ---------- Pitch / field region ----------
// inside: foot point on the playable surface. boundary: within the small touchline tolerance.
// outside: beyond the tolerance (stands, bench, technical area, behind boards).
// unknown: no reliable pitch this frame; nothing new may be promoted while unknown.
export type FieldZone='inside'|'boundary'|'outside'|'unknown';
export type BoundarySide='near'|'far'|'left'|'right';
export type BoundaryLine={a:Pt;b:Pt;side:BoundarySide;support:number;age:number};
export type PitchModel={
 time:number;
 reliable:boolean;
 aspect:number; // frame width / height, used to measure distances in height units
 polygon:Pt[]; // playable region (convex, clipped by detected boundary lines); empty when unreliable
 grassPolygon:Pt[]; // grass hull before line clipping, for debugging
 lines:BoundaryLine[]; // boundary lines currently clipping the region (observed or carried by camera motion)
 coverage:number; // fraction of the frame inside polygon
};
export type FieldFilterConfig={
 enabled:boolean;
 boundaryMargin:number; // tolerance outside the polygon, as a fraction of the person's box height
 minMargin:number; // tolerance floor in height units (normalized y)
 maxMargin:number; // tolerance ceiling in height units
};

// ---------- Relevance (is this detection part of the match?) ----------
export type RejectReason='outside-pitch'|'audience'|'implausible-size'|'low-confidence';
export const rejectText:Record<RejectReason,string>={'outside-pitch':'REJECTED: OUTSIDE PITCH',audience:'REJECTED: AUDIENCE','implausible-size':'REJECTED: SIZE','low-confidence':'REJECTED: LOW PLAYER CONFIDENCE'};
export type ScoredDetection=TrackDetection&{foot:Pt;zone:FieldZone;outsideBy:number;sizeRatio:number};
export type RejectedDetection={box:Box;score:number;reason:RejectReason;detail:string;track?:number};
export type SizeModel={a:number;b:number;reliable:boolean}; // expected box height = a + b * footY

// ---------- Appearance ----------
// Part-based descriptor from inside the box: jersey = upper torso, shorts, socks, plus a coarse
// colour layout. Grass pixels are excluded. quality in [0,1] (size, occlusion, truncation, sharpness).
export type Descriptor={jersey:number[];shorts:number[];socks:number[];layout:number[];quality:number};
export type PartDistance={total:number;jersey:number;shorts:number;socks:number;layout:number}; // 0 identical .. 1 different
export type GallerySample={d:Descriptor;time:number};

// ---------- Teams and roles ----------
export type Team='A'|'B';
export type Role='player'|'goalkeeper'|'referee'|'unknown';
export type RoleLabel='PLAYER_TEAM_A'|'PLAYER_TEAM_B'|'GOALKEEPER_TEAM_A'|'GOALKEEPER_TEAM_B'|'REFEREE'|'UNKNOWN';
export type TeamModel={
 a?:number[]; b?:number[]; // jersey prototypes
 referee?:number[]; // official kit prototype once established from an on-pitch outlier
 keepers:{team?:Team;jersey:number[]}[];
 anchored:boolean; // true when built from user-confirmed team labels
 spread:number; // typical within-team jersey distance
 samples:number;
};
export type KitVote={team?:Team;distA:number;distB:number;distRef:number;outlier:boolean;margin:number};
export type RoleEvidence={hits:number;votes:{team?:Team;outlier:boolean;margin:number;refLike:boolean}[];nearGoal:number;central:number;boundary:number};
export type RoleDecision={label:RoleLabel|'CANDIDATE';role:Role;team?:Team;teamConfidence:number;roleConfidence:number};

// ---------- Identity ----------
export type IdentityStatus='active'|'missing'|'offscreen'|'unknown'|'substituted';
// candidate: collecting evidence. confirmed: bound to a global identity. uncertain: a real participant
// whose identity is not yet decided (never guessed). unknown: a participant we cannot classify.
// rejected: dropped as not part of the match.
export type LocalState='candidate'|'confirmed'|'uncertain'|'unknown'|'rejected';
export type RosterSlot={id:string;team:Team|'ref';number:string;role?:Role;active:boolean};
export type ReidScores={appearance:number;jersey:number;uniform:number;team:boolean;spatial:number;movement:number;temporal:number;final:number;secondBest?:number;missingSeconds?:number};
export type ReidEventKind='reid'|'reid-rejected'|'deferred'|'promotion'|'new-identity'|'role'|'team-change'|'swap-corrected'|'swap-uncertain'|'sanity'|'anchor';
export type ReidEvent={time:number;kind:ReidEventKind;track?:number;playerId?:string;message:string;scores?:ReidScores};
// Persisted per identity in TrackingDoc.identities (see lib/tracking.ts identitySchema).
export type SavedIdentity={playerId:string;role:Role;team?:Team;status:IdentityStatus;gallery:number[][];lastSeen:number;lastBox:Box;lastPitch?:Pt;exitEdge:string;velocity:Pt;identityConfidence:number;teamConfidence:number;roleConfidence:number;anchored:boolean};

// ---------- Pipeline I/O ----------
export type SoccerFrameInput={
 time:number;
 frame:Frame; // RGBA frame used for detection (<=1920 wide)
 detections:TrackDetection[]; // UNFILTERED person + ball detections for this frame
 camera:CameraMotion; // motion since the previous step (estimateCamera)
 anchors:{playerId:string;box:Box}[]; // user-confirmed labels at this time; they override automatic identities
 roster:RosterSlot[]; // active flags already resolved for this time (substitutions)
 homography?:number[]; // image(normalized) -> pitch(normalized) 3x3 row-major, when available
};
export type SoccerObservation={playerId:string;time:number;box:Box;score:number;evidence:'detection'|'predicted'|'reidentified';track:number;conf:number;pitch?:Pt};
export type DebugTrack={track:number;box:Box;playerId?:string;label:string;team?:Team;roleLabel:RoleLabel|'CANDIDATE'|'IDENTITY_UNCERTAIN';state:LocalState;identity:number;zone:FieldZone;reason?:string};
export type DebugFrame={time:number;pitch:PitchModel;tracks:DebugTrack[];rejected:RejectedDetection[]};
export type SoccerFrameResult={observations:SoccerObservation[];debug:DebugFrame;events:ReidEvent[];issues:{playerId:string;time:number;reason:string}[]};
export type SoccerOptions={
 field:FieldFilterConfig;
 maxPerTeam:number; // sanity reference (11), not a hard physical rule
 minHits:number; // detection steps before a candidate may be promoted
 boundaryHits:number; // stronger requirement for candidates first seen in the boundary zone
 reidMin:number; // minimum accumulated identity score to reconnect
 reidMargin:number; // required lead over the second-best identity
 gallerySize:number;
};

// 'A7' -> 'A-07', 'R1' -> 'REF-1'; long form 'TeamA_Player_07' / 'Referee_01'.
export function globalLabel(id:string,long=false){
 const m=/^([AB])(\d{1,2})$/.exec(id);
 if(m)return long?`Team${m[1]}_Player_${m[2].padStart(2,'0')}`:`${m[1]}-${m[2].padStart(2,'0')}`;
 const r=/^R(\d)$/.exec(id);
 if(r)return long?`Referee_${r[1].padStart(2,'0')}`:`REF-${r[1]}`;
 return long?`Player_${id.slice(0,8)}`:id.slice(0,6);
}
export function roleLabel(role:Role,team?:Team):RoleLabel{
 if(role==='referee')return 'REFEREE';
 if(role==='goalkeeper'&&team)return team==='A'?'GOALKEEPER_TEAM_A':'GOALKEEPER_TEAM_B';
 if(role==='player'&&team)return team==='A'?'PLAYER_TEAM_A':'PLAYER_TEAM_B';
 return 'UNKNOWN';
}
