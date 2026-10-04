import {identity,db,bucket,failure} from '@/lib/server';
// Immutable history objects are readable only when referenced by the owner's
// committed tracking manifest; a guessed R2 key never grants access.
export async function GET(request:Request){try{
 const owner=await identity(),url=new URL(request.url),videoId=url.searchParams.get('videoId'),id=url.searchParams.get('id');
 if(!videoId||!id)return Response.json({error:'Missing history reference.'},{status:400});
 const row=await db().prepare('SELECT data FROM records WHERE id=? AND owner=? AND kind=?').bind(owner+':tracking:'+videoId,owner,'tracking').first<{data:string}>();
 if(!row||!JSON.parse(row.data).document.historySegments?.some((s:{id:string})=>s.id===id))return new Response('Not found',{status:404});
 const object=await bucket().get(`${owner}/tracking/${videoId}/${id}.json`);
 return object?new Response(object.body,{headers:{'Content-Type':'application/json','Cache-Control':'private, no-store'}}):new Response('Not found',{status:404});
}catch(error){return failure(error);}}
