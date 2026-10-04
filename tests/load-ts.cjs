const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),ts=require('typescript');
const cache=new Map();
module.exports=function load(file){
 const filename=path.resolve(file);if(cache.has(filename))return cache.get(filename).exports;
 const m=new Module(filename,module);m.paths=module.paths;cache.set(filename,m);const original=m.require.bind(m);
 m.require=id=>{const local=id.startsWith('.')?path.resolve(path.dirname(filename),id+'.ts'):id.startsWith('@/')?path.resolve(id.slice(2)+'.ts'):null;return local&&fs.existsSync(local)?load(local):original(id);};
 m._compile(ts.transpileModule(fs.readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,filename);return m.exports;
};
