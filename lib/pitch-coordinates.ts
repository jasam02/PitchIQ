import {projectFieldMark} from './field-reference';
import type {Box,FieldMark,FieldCamera} from './tracking';
type XY={x:number;y:number};
// Four genuine pitch corners, not corners of a grass mask. Solve the projective
// transform with pivoting; reject degenerate/concave calibration geometry.
export function homography(corners:XY[]):number[]|undefined {
 if(corners.length!==4)return;
 let sign=0,area=0;
 for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4],c=corners[(i+2)%4],cross=(b.x-a.x)*(c.y-b.y)-(b.y-a.y)*(c.x-b.x);if(Math.abs(cross)<1e-5||sign&&Math.sign(cross)!==sign)return;sign=Math.sign(cross);area+=a.x*b.y-b.x*a.y;}
 if(Math.abs(area)<.005)return;
 const target=[{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}],matrix:number[][]=[];
 corners.forEach(({x,y},i)=>{const u=target[i].x,v=target[i].y;matrix.push([x,y,1,0,0,0,-u*x,-u*y,u],[0,0,0,x,y,1,-v*x,-v*y,v]);});
 for(let col=0;col<8;col++){let pivot=col;for(let r=col+1;r<8;r++)if(Math.abs(matrix[r][col])>Math.abs(matrix[pivot][col]))pivot=r;if(Math.abs(matrix[pivot][col])<1e-9)return;[matrix[col],matrix[pivot]]=[matrix[pivot],matrix[col]];const d=matrix[col][col];for(let j=col;j<=8;j++)matrix[col][j]/=d;for(let r=0;r<8;r++)if(r!==col){const scale=matrix[r][col];for(let j=col;j<=8;j++)matrix[r][j]-=scale*matrix[col][j];}}
 return [...matrix.map(row=>row[8]),1];
}
export function projectPosition(matrix:number[]|undefined,p:XY):XY|undefined {
 if(!matrix)return;const denominator=matrix[6]*p.x+matrix[7]*p.y+1;if(Math.abs(denominator)<1e-6)return;
 const x=(matrix[0]*p.x+matrix[1]*p.y+matrix[2])/denominator,y=(matrix[3]*p.x+matrix[4]*p.y+matrix[5])/denominator;
 return Number.isFinite(x)&&Number.isFinite(y)&&x>=-.08&&x<=1.08&&y>=-.08&&y<=1.08?{x,y}:undefined;
}
export function pitchMapper(marks:FieldMark[],cameras:FieldCamera[],time:number){
 const names=['far-left','far-right','near-right','near-left'];
 const corners=names.map(name=>marks.filter(m=>m.name===name&&m.time<=time+.01).sort((a,b)=>b.time-a.time).map(m=>projectFieldMark(m,time,cameras)).find(Boolean)?.points[0]);
 const matrix=corners.every(Boolean)?homography(corners as XY[]):undefined;
 return (box:Box)=>projectPosition(matrix,{x:box.x+box.w/2,y:box.y+box.h});
}
