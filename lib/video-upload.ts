// Browser requests stay below the gateway limit. R2 multipart parts must be
// at least 5 MiB (except the last), so assemble THREE 2 MiB chunks per part.
export const CHUNK_SIZE = 2 * 1024 * 1024;
export const MAX_VIDEO_SIZE = 150 * 1024 * 1024;
const CHUNKS_PER_PART = 3;
const SESSION_TTL = 24 * 60 * 60 * 1000;

export class UploadError extends Error {
  constructor(message: string, public status = 400) { super(message); }
}
type Session = {
  id: string; key: string; uploadId: string; filename: string; size: number;
  metadata: Record<string, unknown>; created: number;
};
type Dependencies = { bucket: R2Bucket; db: D1Database };
const sessionKey = (owner: string, id: string) => `${owner}/uploads/${id}/session`;
const chunkKey = (owner: string, id: string, part: number) => `${owner}/uploads/${id}/chunk-${part}`;
function validId(id: unknown): asserts id is string {
  if (typeof id !== 'string' || !/^[0-9a-f-]{36}$/.test(id)) throw new UploadError('Invalid upload. Refresh the page and try again.');
}
async function session(bucket: R2Bucket, owner: string, id: unknown) {
  validId(id);
  const object = await bucket.get(sessionKey(owner, id));
  if (!object) throw new UploadError('Upload session not found. Please upload the video again.', 404);
  const value = await object.json<Session>();
  if (Date.now() - value.created > SESSION_TTL) throw new UploadError('This upload expired. Please upload the video again.', 410);
  return value;
}
async function saved(db: D1Database, owner: string, id: string) {
  return db.prepare('SELECT data FROM records WHERE id=? AND owner=? AND kind=?').bind(id, owner, 'video').first<{data:string}>();
}
async function cleanChunks(bucket: R2Bucket, owner: string, s: Session) {
  await bucket.delete(Array.from({length: Math.ceil(s.size / CHUNK_SIZE)}, (_, i) => chunkKey(owner, s.id, i + 1)));
  await bucket.delete(sessionKey(owner, s.id));
}
export async function beginUpload({bucket}: Dependencies, owner: string, body: any) {
  if (!Number.isInteger(body.size) || body.size < 1 || body.size > MAX_VIDEO_SIZE ||
      typeof body.filename !== 'string' || body.filename.length > 255 || !body.filename.toLowerCase().endsWith('.mp4')) {
    throw new UploadError('Choose a non-empty MP4 up to 150 MB.');
  }
  const metadata = body.metadata ?? {};
  if (!metadata || Array.isArray(metadata) || typeof metadata !== 'object' || JSON.stringify(metadata).length > 20000) throw new UploadError('Invalid match details.');
  const id = crypto.randomUUID(), key = `${owner}/${id}`;
  const upload = await bucket.createMultipartUpload(key, {httpMetadata:{contentType:'video/mp4'}, customMetadata:{uploadSession:id}});
  const s: Session = {id, key, uploadId:upload.uploadId, filename:body.filename, size:body.size, metadata, created:Date.now()};
  try { await bucket.put(sessionKey(owner, id), JSON.stringify(s)); }
  catch (error) { await upload.abort().catch(() => {}); throw error; }
  return {id, partSize:CHUNK_SIZE, protocol:'staged-v2'};
}
export async function putChunk({bucket}: Dependencies, owner: string, id: unknown, part: number, request: Request) {
  const s = await session(bucket, owner, id);
  const count = Math.ceil(s.size / CHUNK_SIZE);
  if (!Number.isInteger(part) || part < 1 || part > count || !request.body) throw new UploadError('Invalid video chunk.');
  const expected = Math.min(CHUNK_SIZE, s.size - (part - 1) * CHUNK_SIZE);
  const declared = request.headers.get('content-length');
  if (declared !== null && Number(declared) !== expected) throw new UploadError('Video chunk has the wrong size.', 413);
  // Bound actual bytes too; never trust Content-Length or buffer a whole video.
  const bytes = new Uint8Array(expected), reader = request.body.getReader();
  let offset = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      if (offset + value.length > expected) { await reader.cancel(); throw new UploadError('Video chunk is too large.', 413); }
      bytes.set(value, offset); offset += value.length;
    }
  } finally { reader.releaseLock(); }
  if (offset !== expected) throw new UploadError('Video chunk was interrupted. Please retry.');
  const result = await bucket.put(chunkKey(owner, s.id, part), bytes);
  return {partNumber:part, etag:result?.etag};
}
export async function finishUpload(deps: Dependencies, owner: string, id: unknown) {
  validId(id);
  const {bucket, db} = deps;
  // Completion can safely be retried after a lost HTTP response.
  const existing = await saved(db, owner, id);
  if (existing) return {id, ...JSON.parse(existing.data)};
  const s = await session(bucket, owner, id);
  let object = await bucket.head(s.key);
  if (!object) {
    const count = Math.ceil(s.size / CHUNK_SIZE), parts: R2UploadedPart[] = [];
    const upload = bucket.resumeMultipartUpload(s.key, s.uploadId);
    for (let first = 1; first <= count; first += CHUNKS_PER_PART) {
      const length = Math.min(CHUNK_SIZE * CHUNKS_PER_PART, s.size - (first - 1) * CHUNK_SIZE);
      const bytes = new Uint8Array(length);
      let offset = 0;
      for (let n = first; n < first + CHUNKS_PER_PART && n <= count; n++) {
        const chunk = await bucket.get(chunkKey(owner, id, n));
        const expected = Math.min(CHUNK_SIZE, s.size - (n - 1) * CHUNK_SIZE);
        if (!chunk || chunk.size !== expected) throw new UploadError(`Video chunk ${n} is missing or incomplete. Please retry the upload.`, 409);
        bytes.set(new Uint8Array(await chunk.arrayBuffer()), offset); offset += expected;
      }
      parts.push(await upload.uploadPart(parts.length + 1, bytes));
    }
    await upload.complete(parts);
    // The production multipart completion response is not an authoritative
    // metadata read. Read the committed object before verifying its size.
    object = await bucket.head(s.key);
  }
  // The storage key comes from this owner's server-created session, never from
  // request input. Do not reject valid objects over optional/case-normalized
  // metadata returned differently by local emulation and the R2 service.
  if (!object || object.size !== s.size) {
    console.error('Upload verification failed', {id, expected:s.size, actual:object?.size});
    throw new UploadError('The saved video has an unexpected size. Please retry the upload.', 409);
  }
  const data = {filename:s.filename, key:s.key, size:s.size, metadata:s.metadata, started:s.created, mode:'manual'};
  // Keep the completed object and session if D1 fails: a retry recovers without
  // retransferring the footage. Ownership is checked above and on every read.
  await db.prepare('INSERT INTO records (id,owner,kind,data,created) VALUES (?,?,?,?,?) ON CONFLICT(id) DO NOTHING')
    .bind(id, owner, 'video', JSON.stringify(data), s.created).run();
  await cleanChunks(bucket, owner, s).catch(error => console.error('Upload temporary cleanup failed', String(error)));
  return {id, ...data};
}
export async function abortUpload(deps: Dependencies, owner: string, id: unknown) {
  validId(id);
  // Never delete a completed video, even when a completion response was lost.
  if (await saved(deps.db, owner, id)) return {aborted:false};
  const s = await session(deps.bucket, owner, id);
  if (await deps.bucket.head(s.key)) return {aborted:false};
  await deps.bucket.resumeMultipartUpload(s.key, s.uploadId).abort();
  await cleanChunks(deps.bucket, owner, s);
  return {aborted:true};
}
