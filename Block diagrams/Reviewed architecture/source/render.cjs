const sharp = require('C:/Users/riche/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const path = require('path');
const fs = require('fs');
const folder=path.resolve(__dirname,'../previews');
(async()=>{
 if(process.argv.includes('--svg-render')) for(let n=1;n<=8;n++){
  const s=String(n).padStart(2,'0');
  await sharp(path.join(folder,s+'.svg')).png().toFile(path.join(folder,s+'.png'));
 }
 const thumbs=await Promise.all(Array.from({length:8},async(_,i)=>({input:await sharp(path.join(folder,String(i+1).padStart(2,'0')+'.png')).resize(880,680,{fit:'contain',background:'#e2e8f0'}).png().toBuffer(),left:(i%2)*900+10,top:Math.floor(i/2)*700+10})));
 await sharp({create:{width:1800,height:2800,channels:3,background:'#e2e8f0'}}).composite(thumbs).png().toFile(path.resolve(folder,'../All pages.png'));
 console.log('Rendered contact sheet from the eight PNG exports.');
})();
