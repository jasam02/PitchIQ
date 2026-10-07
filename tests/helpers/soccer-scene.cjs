// Synthetic broadcast scene shared by the soccer integration test and scripts/benchmark-soccer.cjs.
// Stands with painted fans at the top, advertising boards, far touchline at y=.2, striped pitch with a halfway
// line, near touchline at y=.85, a strip of run-off grass, then a non-grass bench/track area from y=.9.
const {makeFrame}=require('./ts-require.cjs');
const FAR=.2,NEAR=.85,BENCH=.9;
const KITS={
 red:{shirt:[200,30,35],shorts:[240,240,240],socks:[200,30,35]},
 blue:{shirt:[30,60,190],shorts:[20,20,30],socks:[240,240,240]},
 ref:{shirt:[235,215,40],shorts:[15,15,15],socks:[15,15,15]},
 fan:{shirt:[200,30,35],shorts:[60,60,60],socks:[60,60,60]},
 coat:{shirt:[60,60,70],shorts:[60,60,70],socks:[30,30,30]},
};
const SKIN=[205,160,130],BOOT=[25,25,25];
function makeScene(W=640,H=360,seed=7){
 const rnd=()=>{seed=(seed*1103515245+12345)%2147483648;return seed/2147483648;};
 const base=makeFrame(W,H,(x,y)=>{
  const Y=y/H,X=x/W,n=(rnd()-.5)*12;
  const c=Y<.15?(((x*7+y*13)%23<3)?[200,200,200]:[95,95,105]):Y<.18?(Math.floor(X*12)%2?[235,235,235]:[40,60,200]):
   Y>=BENCH?(Math.floor(X*20)%2?[90,70,50]:[150,150,155]):
   (Math.abs(Y-FAR)<.004||Math.abs(Y-NEAR)<.004||(Y>FAR&&Y<NEAR&&Math.abs(X-.5)<.002))?[230,235,230]:
   (Math.floor(X*16)%2?[40,140,50]:[52,162,62]);
  return c.map(v=>Math.max(0,Math.min(255,v+n)));
 });
 // A person is {key, x, y (foot point), kit, h?}; the box follows a simple perspective size model.
 const boxOf=p=>{const h=p.h??(.14+.08*p.y),w=h*.32*H/W;return {x:p.x-w/2,y:p.y-h,w,h};};
 // People lower in the frame stand in front (painted last).
 function paint(people){
  const data=new Uint8ClampedArray(base.data);
  for(const p of [...people].sort((a,b)=>a.y-b.y)){
   const b=boxOf(p),k=KITS[p.kit],x0=Math.round(b.x*W),y0=Math.round(b.y*H),w=Math.round(b.w*W),h=Math.round(b.h*H);
   for(let j=0;j<h;j++)for(let i=0;i<w;i++){
    const x=x0+i,y=y0+j;if(x<0||y<0||x>=W||y>=H)continue;const u=i/w,v=j/h;let c=null;
    if(v<.15){if(u>.35&&u<.65)c=SKIN;}else if(v<.5){if(u>.15&&u<.85)c=k.shirt;}else if(v<.7){if(u>.2&&u<.8)c=k.shorts;}
    else if(v<.94){if((u>.25&&u<.45)||(u>.55&&u<.75))c=k.socks;}else if((u>.22&&u<.46)||(u>.54&&u<.78))c=BOOT;
    if(c){const q=(y*W+x)*4;data[q]=c[0];data[q+1]=c[1];data[q+2]=c[2];}
   }
  }
  return {width:W,height:H,data};
 }
 // Detector-like output: boxes clipped to the frame, as the browser detector returns them.
 const detect=(p,i)=>{const b=boxOf(p),x=Math.max(0,b.x),y=Math.max(0,b.y);return {id:'d'+i,kind:'person',score:p.score??.85,box:{x,y,w:Math.min(1,b.x+b.w)-x,h:Math.min(1,b.y+b.h)-y}};};
 return {W,H,base,boxOf,paint,detect};
}
module.exports={makeScene,KITS,FAR,NEAR,BENCH};
