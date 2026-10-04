class RequestError extends Error {
  constructor(message: string, public status: number) { super(message); }
}
export async function uploadRequest(url: string, init: RequestInit, retry = false): Promise<any> {
  for (let attempt = 0; ; attempt++) {
    try {
      const response = await fetch(url, {...init, signal:AbortSignal.timeout(120000)});
      const text = await response.text();
      let data: any;
      try { data = JSON.parse(text); } catch { /* Gateways can return plain text or HTML. */ }
      if (!response.ok) {
        const message = response.status === 413 ? 'The server rejected this video chunk. Refresh the page and try again; no compression is needed.'
          : response.status === 401 ? 'The upload service denied access. Refresh the page and retry.'
          : data?.error || `Upload failed (HTTP ${response.status}). Please retry.`;
        throw new RequestError(message, response.status);
      }
      if (!data) throw new RequestError('The server returned an unexpected response. Please refresh and retry.', 502);
      return data;
    } catch (error) {
      const transient = !(error instanceof RequestError) || error.status === 408 || error.status === 429 || error.status >= 500;
      if (!retry || !transient || attempt >= 2) throw error;
      await new Promise(resolve => setTimeout(resolve, 600 * 2 ** attempt));
    }
  }
}
export type UploadSession = {id:string; partSize:number; protocol:string};
export async function uploadVideo(file: File, metadata: Record<string, unknown>, progress: (percent:number)=>void, resume?: UploadSession, started?: (session:UploadSession)=>void) {
  const json = (body: unknown): RequestInit => ({method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const session: UploadSession = resume ?? await uploadRequest('/api/videos?action=init', json({filename:file.name, size:file.size, metadata}));
  if (session.protocol !== 'staged-v2' || session.partSize !== 2 * 1024 * 1024 || !session.id) throw new Error('The upload service was updated. Please refresh the page and retry.');
  started?.(session);
  for (let offset = 0, part = 1; offset < file.size; offset += session.partSize, part++) {
    const chunk = file.slice(offset, offset + session.partSize);
    await uploadRequest(`/api/videos?action=part&id=${encodeURIComponent(session.id)}&partNumber=${part}`, {method:'PUT', body:chunk}, true);
    progress(Math.min(99, Math.round((offset + chunk.size) / file.size * 99)));
  }
  const result = await uploadRequest('/api/videos?action=complete', json({id:session.id}), true);
  progress(100);
  return result;
}
