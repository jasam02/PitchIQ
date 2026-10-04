import {env} from 'cloudflare:workers';
// Temporary single workspace until PitchIQ accounts are implemented.
// Keep the previous local owner ID so existing local footage remains accessible.
export async function identity(){return 'local_seedy';}
export const db=()=> (env as unknown as {DB:D1Database}).DB;
export const bucket=()=> (env as unknown as {BUCKET:R2Bucket}).BUCKET;
export function failure(e:unknown){console.error('PitchIQ request failed', e instanceof Error?e.message:'unknown');return Response.json({error:'Unable to save right now. Please retry.'},{status:503});}
