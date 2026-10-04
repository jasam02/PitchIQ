import {identity,db,bucket,failure} from '@/lib/server';
import {stages} from '@/lib/demo';
import {beginUpload,putChunk,finishUpload,abortUpload,UploadError} from '@/lib/video-upload';
function uploadFailure(error:unknown){if(error instanceof UploadError){console.error('Upload rejected',error.status,error.message);return Response.json({error:error.message},{status:error.status});}return failure(error);}
export async function POST(request:Request){try{const owner=await identity();const action=new URL(request.url).searchParams.get('action');
 if(action){const deps={bucket:bucket(),db:db()};const body=await request.json() as any;try{if(action==='init')return Response.json(await beginUpload(deps,owner,body));if(action==='complete')return Response.json(await finishUpload(deps,owner,body.id));if(action==='abort')return Response.json(await abortUpload(deps,owner,body.id));return Response.json({error:'Invalid upload action.'},{status:400});}catch(error){return uploadFailure(error);}}
 const size=Number(request.headers.get('content-length'));if(size>150*1024*1024)return Response.json({error:'MVP uploads are limited to 150 MB. Use a shorter MP4 excerpt.'},{status:413});const form=await request.formData();const f=form.get('file') as File; if(!f||!f.name?.toLowerCase().endsWith('.mp4')||f.size>150*1024*1024)return Response.json({error:'Choose an MP4 under 150 MB.'},{status:400});const id=crypto.randomUUID();const key=owner+'/'+id;const data={filename:f.name,key,metadata:JSON.parse(String(form.get('metadata')||'{}')),started:Date.now(),mode:'manual'};await bucket().put(key,f.stream(),{httpMetadata:{contentType:'video/mp4'}});try{await db().prepare('INSERT INTO records (id,owner,kind,data,created) VALUES (?,?,?,?,?)').bind(id,owner,'video',JSON.stringify(data),Date.now()).run();}catch(e){await bucket().delete(key);throw e;}return Response.json({id,...data});}catch(e){return failure(e);}}
export async function GET(request:Request){
 try{
  const owner=await identity(),url=new URL(request.url),id=url.searchParams.get('id');
  const row=await db().prepare('SELECT data FROM records WHERE id=? AND owner=? AND kind=?').bind(id,owner,'video').first<{data:string}>();
  if(!row)return new Response('Not found',{status:404});
  const data=JSON.parse(row.data);
  if(url.searchParams.has('status')){const step=Math.min(7,Math.floor((Date.now()-data.started)/2500));return Response.json({step,stage:stages[step],mode:'manual'});}
  const headers=new Headers({'Content-Type':'video/mp4','Accept-Ranges':'bytes','Cache-Control':'private, no-store'});
  const range=request.headers.get('range');
  let requested:{offset:number;length:number}|undefined;
  if(range){
   const info=await bucket().head(data.key);
   if(!info)return new Response('Not found',{status:404});
   const match=/^bytes=(\d*)-(\d*)$/.exec(range);
   const start=match?.[1]?Number(match[1]):Math.max(0,info.size-Number(match?.[2]));
   const end=match?.[1]&&match?.[2]?Math.min(Number(match[2]),info.size-1):info.size-1;
   if(!match||(!match[1]&&!match[2])||!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<0||start>=info.size||end<start){
    headers.set('Content-Range',`bytes */${info.size}`);return new Response(null,{status:416,headers});
   }
   requested={offset:start,length:end-start+1};
   headers.set('Content-Range',`bytes ${start}-${end}/${info.size}`);
  }
  const obj=await bucket().get(data.key,requested?{range:requested}:undefined);
  if(!obj)return new Response('Not found',{status:404});
  headers.set('Content-Length',String(requested?.length??obj.size));
  return new Response(obj.body,{status:requested?206:200,headers});
 }catch(e){return failure(e);}
}
export async function DELETE(request:Request){try{const owner=await identity();const {id}=await request.json() as any;const row=await db().prepare('SELECT data FROM records WHERE id=? AND owner=? AND kind=?').bind(id,owner,'video').first<{data:string}>();if(row){await bucket().delete(JSON.parse(row.data).key);let cursor:string|undefined;do{const page=await bucket().list({prefix:owner+'/tracking/'+id+'/',cursor});if(page.objects.length)await bucket().delete(page.objects.map(o=>o.key));cursor=page.truncated?page.cursor:undefined;}while(cursor);}await db().batch([db().prepare('DELETE FROM records WHERE id=? AND owner=? AND kind=?').bind(id,owner,'video'),db().prepare('DELETE FROM records WHERE id=? AND owner=? AND kind=?').bind(owner+':tracking:'+id,owner,'tracking')]);return Response.json({deleted:true});}catch(e){return failure(e);}}
export async function PUT(request:Request){try{const owner=await identity();const url=new URL(request.url);if(url.searchParams.get('action')!=='part')return Response.json({error:'Invalid upload action.'},{status:400});return Response.json(await putChunk({bucket:bucket(),db:db()},owner,url.searchParams.get('id'),Number(url.searchParams.get('partNumber')),request));}catch(e){return uploadFailure(e);}}
