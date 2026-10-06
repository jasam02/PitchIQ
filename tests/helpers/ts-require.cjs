// Lets node:test files require TypeScript modules (and their relative .ts imports) directly.
const fs=require('node:fs');
const ts=require('typescript');
if(!require.extensions['.ts'])require.extensions['.ts']=(m,filename)=>m._compile(ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,esModuleInterop:true}}).outputText,filename);
// Synthetic RGBA frame: paint(x,y) returns [r,g,b] for each pixel.
function makeFrame(width,height,paint){const data=new Uint8ClampedArray(width*height*4);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const [r,g,b]=paint(x,y),i=(y*width+x)*4;data[i]=r;data[i+1]=g;data[i+2]=b;data[i+3]=255;}return {width,height,data};}
module.exports={makeFrame};
