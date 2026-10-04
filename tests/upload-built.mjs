// Run after pnpm build. Tests the actual Worker bundle over local HTTP, never
// a live account. Example: node tests/upload-built.mjs /path/to/match.mp4
import {createRequire} from 'node:module';
import {readFileSync,readdirSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const require = createRequire(import.meta.url);
const {Miniflare} = createRequire(require.resolve('wrangler/package.json'))('miniflare');
const filePath = process.argv[2];
if(!filePath)throw Error('Supply a local MP4 path.');
const bytes = readFileSync(filePath);
const mf = new Miniflare({
  modules:[{type:'ESModule',path:resolve('dist/server/index.js')},...readdirSync('dist/server',{recursive:true}).filter(name=>name!=='index.js'&&/\.(m?js)$/.test(name)).map(name=>({type:'ESModule',path:resolve('dist/server',name)}))],
  compatibilityDate:'2026-05-15',compatibilityFlags:['nodejs_compat'],
  r2Buckets:['BUCKET'],d1Databases:['DB'],assets:{directory:resolve('dist/client'),binding:'ASSETS',routerConfig:{has_user_worker:true}},
});
let id;
try {
  const db = await mf.getD1Database('DB');
  for(const statement of readFileSync('drizzle/0000_sudden_beast.sql','utf8').split('--> statement-breakpoint')) {
    if(statement.trim())await db.prepare(statement.trim()).run();
  }
  const origin = await mf.ready;
  // The temporary shared workspace must work without auth headers or cookies.
  const call = (path,init={})=>fetch(new URL(path,origin),{...init,signal:AbortSignal.timeout(120000)});
  const json = body=>({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const decoded = async response=>{const text=await response.text();assert.equal(response.ok,true,`${response.status}: ${text.slice(0,500)}`);return JSON.parse(text);};
  const home = await call('/');assert.equal(home.status,200);
  const html = await home.text();assert.match(html,/PitchIQ/);
  const css = html.match(/href="([^" ]+\.css)"/);
  assert.ok(css,'Page references a stylesheet');
  assert.equal((await call(css[1])).status,200);
  for(const name of ['worker.js','engine.py',...JSON.parse(readFileSync('scripts/norfair-assets.json','utf8')).map(a=>'runtime/'+a.name),'runtime/norfair-2.3.0-py3-none-any.whl','runtime/filterpy-1.4.5-py3-none-any.whl']){
    const response=await call('/norfair/'+name);assert.equal(response.status,200,name);
    const actual=Buffer.from(await response.arrayBuffer()),expected=readFileSync('public/norfair/'+name);
    assert.equal(createHash('sha256').update(actual).digest('hex'),createHash('sha256').update(expected).digest('hex'),name+' must serve the engine asset, not an HTML fallback');
  }
  const session = await decoded(await call('/api/videos?action=init',json({filename:'upload-qa.mp4',size:bytes.length,metadata:{opponent:'Upload QA',consent:true}})));
  id=session.id;
  let chunks=0;
  for(let offset=0,n=1;offset<bytes.length;offset+=session.partSize,n++) {
    await decoded(await call(`/api/videos?action=part&id=${id}&partNumber=${n}`,{method:'PUT',body:bytes.subarray(offset,offset+session.partSize)}));chunks++;
  }
  const result = await decoded(await call('/api/videos?action=complete',json({id})));
  assert.equal(result.id,id);
  const download = await call('/api/videos?id='+id);
  assert.equal(download.status,200);
  const received = Buffer.from(await download.arrayBuffer());
  const hash = value=>createHash('sha256').update(value).digest('hex');
  assert.equal(hash(received),hash(bytes));
  for(const range of ['bytes=0-1023','bytes=4096-8191','bytes=-1024']) {
    const response = await call('/api/videos?id='+id,{headers:{Range:range}});
    assert.equal(response.status,206);
    const expected=range==='bytes=-1024'?bytes.subarray(-1024):range==='bytes=0-1023'?bytes.subarray(0,1024):bytes.subarray(4096,8192);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()),expected);
  }
  assert.equal((await call('/api/videos?id='+id,{headers:{Range:`bytes=${bytes.length}-`}})).status,416);
  const tracking = await decoded(await call('/api/tracking?videoId='+id));
  assert.ok(tracking.document.players.length>=22);
  const page = await call('/tracking/'+id);assert.equal(page.status,200);
  const trackingHtml=await page.text();
  assert.match(trackingHtml,/tracking-shell/);
  assert.ok(trackingHtml.includes(id),'Tracking page receives the uploaded video ID');
  for(const match of trackingHtml.matchAll(/(?:src|href)="([^" ]+\.js)"/g))assert.equal((await call(match[1])).status,200);
  const repeated = await decoded(await call('/api/videos?action=complete',json({id})));
  assert.equal(repeated.id,id);
  console.log(JSON.stringify({passed:true,bytes:bytes.length,chunks,sha256:hash(bytes),checks:['built Worker HTTP upload','completion','identical download','range seeking','out-of-range 416','upload without sign-in','tracking API roster','tracking page','homepage CSS','idempotent completion','all Norfair browser assets byte-identical']}));
} finally {await mf.dispose();}
