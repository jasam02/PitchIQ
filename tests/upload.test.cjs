const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const {createHash} = require('node:crypto');
const wranglerRequire = Module.createRequire(require.resolve('wrangler/package.json'));
const {Miniflare,Request:WorkerRequest} = wranglerRequire('miniflare');
function load(file, overrides = {}) {
  const filename = path.resolve(file), m = new Module(filename, module);
  m.paths = module.paths;
  const original = m.require.bind(m);
  m.require = id => overrides[id] || original(id);
  m._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText, filename);
  return m.exports;
}
const upload = load('lib/video-upload.ts');
const client = load('lib/upload-client.ts');
const digest = bytes => createHash('sha256').update(bytes).digest('hex');

test('video upload: real R2/D1 emulation, browser client, completion, ownership and playback ranges', {timeout:180000}, async t => {
  const mf = new Miniflare({modules:true,script:'export default {fetch(){return new Response("ok")}}',compatibilityDate:'2026-05-15',r2Buckets:['BUCKET'],d1Databases:['DB']});
  const bucket = await mf.getR2Bucket('BUCKET'), db = await mf.getD1Database('DB');
  await db.exec('CREATE TABLE records(id TEXT PRIMARY KEY,owner TEXT,kind TEXT,data TEXT,created INTEGER)');
  let owner = 'test-coach';
  const deps = {bucket, db};
  const api = load('app/api/videos/route.ts', {
    '@/lib/video-upload':upload,
    '@/lib/demo':{stages:[]},
    '@/lib/server':{identity:async()=>{if(!owner)throw Error('Unauthorized');return owner;}, bucket:()=>bucket,
      db:()=>db,
      failure:error=>Response.json({error:error.message},{status:error.message==='Unauthorized'?401:503})}
  });
  const originalFetch = global.fetch;
  const dispatch = (url, init={}) => api[init.method || 'GET'](new WorkerRequest(new URL(url,'https://pitchiq.test'),init));
  global.fetch = dispatch;
  try {
    await t.test('reproduces the shipped 4 MiB R2 multipart failure', async()=>{
      const old = await bucket.createMultipartUpload('test-coach/regression');
      const a = await old.uploadPart(1,new Uint8Array(4*1024*1024));
      const b = await old.uploadPart(2,new Uint8Array(1));
      await assert.rejects(old.complete([a,b]),/minimum|small/i);
      await old.abort();
    });
    let realId;
    const fixture = process.env.PITCHIQ_TEST_VIDEO;
    const cases = [1,2*1024*1024,6*1024*1024,8*1024*1024,16*1024*1024,150*1024*1024];
    if (fixture) cases.push(fixture);
    for(const size of cases) await t.test(`round trip ${typeof size==='string'?'actual soccer MP4':size+' bytes'}`,async()=>{
      const bytes = typeof size === 'string' ? fs.readFileSync(size) : Buffer.alloc(size,123);
      const file = new File([bytes],'match.mp4',{type:'video/mp4'});
      const progress = [];
      const result = await client.uploadVideo(file,{opponent:'Upload QA'},n=>progress.push(n));
      assert.equal(result.size,bytes.length);
      assert.equal(progress.at(-1),100);
      const response = await dispatch('/api/videos?id='+result.id);
      assert.equal(response.status,200,response.ok?'':await response.text());
      assert.equal(Number(response.headers.get('Content-Length')),bytes.length);
      assert.equal(digest(Buffer.from(await response.arrayBuffer())),digest(bytes));
      const range = await dispatch('/api/videos?id='+result.id,{headers:{Range:'bytes=0-'+Math.min(1023,bytes.length-1)}});
      assert.equal(range.status,206);
      assert.deepEqual(Buffer.from(await range.arrayBuffer()),bytes.subarray(0,Math.min(1024,bytes.length)));
      const repeated = await upload.finishUpload(deps,owner,result.id);
      assert.equal(repeated.id,result.id);
      assert.equal((await bucket.list({prefix:`${owner}/uploads/${result.id}/`})).objects.length,0);
      if(typeof size==='string')realId=result.id;
    });
    await t.test('rejects bad sizes and isolates upload sessions and saved videos by owner',async()=>{
      for(const size of [0,-1,150*1024*1024+1,1.5]) await assert.rejects(upload.beginUpload(deps,owner,{filename:'bad.mp4',size}),/MP4/);
      const s = await upload.beginUpload(deps,owner,{filename:'test.mp4',size:8*1024*1024});
      const request = ()=>new Request('https://test',{method:'PUT',body:new Uint8Array(2*1024*1024)});
      await assert.rejects(upload.putChunk(deps,'other-coach',s.id,1,request()),/not found/);
      await assert.rejects(upload.finishUpload(deps,'other-coach',s.id),/not found/);
      await assert.rejects(upload.putChunk(deps,owner,s.id,5,request()),/Invalid video chunk/);
      await assert.rejects(upload.putChunk(deps,owner,s.id,1,new Request('https://test',{method:'PUT',body:new Uint8Array(1)})),/interrupted/);
      await assert.rejects(upload.putChunk(deps,owner,s.id,1,new Request('https://test',{method:'PUT',body:new Uint8Array(2*1024*1024+1)})),/too large/);
      await assert.rejects(upload.finishUpload(deps,owner,s.id),/missing/);
      assert.equal((await upload.abortUpload(deps,owner,s.id)).aborted,true);
      assert.equal((await bucket.list({prefix:`${owner}/uploads/${s.id}/`})).objects.length,0);
      owner='other-coach';
      if(realId)assert.equal((await dispatch('/api/videos?id='+realId)).status,404);
      owner='';
      assert.equal((await dispatch('/api/videos?action=init',{method:'POST',body:'{}'})).status,401);
      owner='test-coach';
    });
    await t.test('recovers completion after database failure without losing or duplicating video',async()=>{
      const s = await upload.beginUpload(deps,owner,{filename:'retry.mp4',size:7});
      await upload.putChunk(deps,owner,s.id,1,new Request('https://test',{method:'PUT',body:'example'}));
      const flaky = {...deps,db:{prepare(sql){if(sql.startsWith('INSERT'))throw Error('database offline');return db.prepare(sql);}}};
      await assert.rejects(upload.finishUpload(flaky,owner,s.id),/database offline/);
      assert.equal((await bucket.head(`${owner}/${s.id}`)).size,7);
      assert.equal((await upload.abortUpload(deps,owner,s.id)).aborted,false);
      const recovered = await upload.finishUpload(deps,owner,s.id);
      assert.equal(recovered.size,7);
      assert.equal((await upload.finishUpload(deps,owner,s.id)).id,s.id);
    });
    await t.test('completion accepts authoritative size without optional multipart metadata',async()=>{
      const s=await upload.beginUpload(deps,owner,{filename:'metadata.mp4',size:7});
      await upload.putChunk(deps,owner,s.id,1,new Request('https://test',{method:'PUT',body:'example'}));
      const serviceBucket=new Proxy(bucket,{get(target,key){
        if(key==='head')return async name=>{const value=await target.head(name);return value?{...value,customMetadata:undefined}:null;};
        if(key==='resumeMultipartUpload')return (...args)=>{const original=target.resumeMultipartUpload(...args);return {...original,uploadPart:original.uploadPart.bind(original),complete:async parts=>{await original.complete(parts);return {};}};};
        const value=target[key];return typeof value==='function'?value.bind(target):value;
      }});
      const result=await upload.finishUpload({...deps,bucket:serviceBucket},owner,s.id);
      assert.equal(result.size,7);
    });
    await t.test('automatically retries a failed chunk and lost completion response',async()=>{
      let chunkFailures=0, completeFailures=0, initCount=0;
      global.fetch=async(url,init)=>{
        if(url.includes('action=init'))initCount++;
        if(url.includes('action=part')&&chunkFailures++===0)return new Response('Temporary failure',{status:503});
        const result=await dispatch(url,init);
        if(url.includes('action=complete')&&completeFailures++===0)throw new TypeError('Simulated lost response');
        return result;
      };
      const result=await client.uploadVideo(new File(['example'],'retry.mp4'),{},()=>{});
      assert.ok(result.id);assert.equal(initCount,1);assert.equal(completeFailures,2);
      global.fetch=dispatch;
    });
    await t.test('plain-text gateway errors are actionable, not JSON parser errors',async()=>{
      global.fetch=async()=>new Response('Payload Too Large',{status:413});
      await assert.rejects(client.uploadRequest('/test',{},true),/server rejected this video chunk/i);
      global.fetch=dispatch;
    });
  } finally {global.fetch=originalFetch;await mf.dispose();}
});
