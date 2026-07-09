#!/usr/bin/env node
// Auto-test the demo's zh/en code-switch segmentation FSM (the boundary logic
// in ime.js feedKey). Runs keystroke strings through the same state machine and
// checks where the zh/en boundaries land -- so bugs like "English run swallows
// the following zhuyin" are caught without hand-typing in the browser.
//
//   node space-static/test-codeswitch.mjs
//
// NOTE: this MIRRORS ime.js feedKey/splitTrailingSyllable -- keep in sync.
import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';
const DIR=path.dirname(fileURLToPath(import.meta.url));

const TONES='ˊˇˋ˙';
const strip=s=>[...s].filter(c=>!TONES.includes(c)).join('');
const DACHEN={'1':['ㄅ',0],'q':['ㄆ',0],'a':['ㄇ',0],'z':['ㄈ',0],'2':['ㄉ',0],'w':['ㄊ',0],'s':['ㄋ',0],'x':['ㄌ',0],'e':['ㄍ',0],'d':['ㄎ',0],'c':['ㄏ',0],'r':['ㄐ',0],'f':['ㄑ',0],'v':['ㄒ',0],'5':['ㄓ',0],'t':['ㄔ',0],'g':['ㄕ',0],'b':['ㄖ',0],'y':['ㄗ',0],'h':['ㄘ',0],'n':['ㄙ',0],'u':['ㄧ',1],'j':['ㄨ',1],'m':['ㄩ',1],'8':['ㄚ',2],'i':['ㄛ',2],'k':['ㄜ',2],',':['ㄝ',2],'9':['ㄞ',2],'o':['ㄟ',2],'l':['ㄠ',2],'.':['ㄡ',2],'0':['ㄢ',2],'p':['ㄣ',2],';':['ㄤ',2],'/':['ㄥ',2],'-':['ㄦ',2]};
const TONEK={'6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'};
const PUNCT={'<':'，','>':'。','?':'？','!':'！',':':'：','"':'；','(':'（',')':'）'};

const validBase=new Set();
for(const l of fs.readFileSync(path.join(DIR,'phonetic_table.tsv'),'utf-8').split('\n')){
  const t=l.indexOf('\t'); if(t<0)continue; validBase.add(strip(l.slice(0,t)));
}
function splitTrailingSyllable(raw){
  for(let L=Math.min(3,raw.length);L>=1;L--){
    let bopo='',ok=true;
    for(const c of raw.slice(-L)){ if(!DACHEN[c]){ok=false;break;} bopo+=DACHEN[c][0]; }
    if(ok&&validBase.has(bopo)) return [raw.slice(0,raw.length-L),bopo];
  }
  return null;
}

// --- the FSM under test (mirror of ime.js feedKey) ---
function run(keys){
  let committed=[],cur=['','',''],rawWord='',enRun=false;
  const numSym=()=>(cur[0]?1:0)+(cur[1]?1:0)+(cur[2]?1:0);
  const hasPending=()=>cur[0]||cur[1]||cur[2];
  const pending=()=>cur[0]+cur[1]+cur[2];
  const push=t=>committed.push(t);
  const reset=()=>{cur=['','',''];rawWord='';enRun=false;};
  const commitRun=()=>{ if(enRun&&rawWord)push({t:'en',v:rawWord}); else if(hasPending())push({t:'zh',v:pending()}); else if(rawWord)push({t:'en',v:rawWord}); reset(); };
  for(const k of keys){
    if(k in PUNCT){ if(hasPending()||rawWord)commitRun(); push({t:'punct',v:PUNCT[k]}); continue; }
    if(k===' '){ if(hasPending()||rawWord)commitRun(); continue; }
    if(k in TONEK){
      if(!enRun&&hasPending()){ push({t:'zh',v:pending()+TONEK[k]}); reset(); continue; }
      if(enRun){ const sp=splitTrailingSyllable(rawWord); if(sp){ if(sp[0])push({t:'en',v:sp[0]}); push({t:'zh',v:sp[1]+TONEK[k]}); reset(); continue; } }
      rawWord+=k; enRun=true; continue;
    }
    if(enRun){ rawWord+=k; continue; }
    if(DACHEN[k]){ cur[DACHEN[k][1]]=DACHEN[k][0]; rawWord+=k; if(rawWord.length>numSym())enRun=true; continue; }
    rawWord+=k; enRun=true;
  }
  if(hasPending()||rawWord)commitRun();
  return committed.map(t=>t.t+':'+t.v).join(' | ');
}

// --- cases: [keystrokes, expected segmentation, description] ---
const T=[
  ['python5k4ek7',        'en:python | zh:ㄓㄜˋ | zh:ㄍㄜ˙',        'English then zhuyin, no space (這個)'],
  ['driving5j;4dj;4',     'en:driving | zh:ㄓㄨㄤˋ | zh:ㄎㄨㄤˋ',    'English then zhuyin, no space (狀況)'],
  ['python 5k4',          'en:python | zh:ㄓㄜˋ',                    'English, space, zhuyin'],
  ['5k4python',           'zh:ㄓㄜˋ | en:python',                    'zhuyin then English, no space'],
  ['test123',             'en:test123',                             'English word with digits stays English'],
  ['rm,6',                'zh:ㄐㄩㄝˊ',                              'single zhuyin syllable 覺'],
  ['hello',               'en:hello',                               'pure English'],
  ['5k4 rm,6',            'zh:ㄓㄜˋ | zh:ㄐㄩㄝˊ',                   'two zhuyin, space-separated'],
  ['code5j;4',            'en:code | zh:ㄓㄨㄤˋ',                    'short English then zhuyin (狀)'],
  ['api2u4',              'en:api | zh:ㄉㄧˋ',                       'English then 地/弟 (ㄉㄧˋ)'],
];

let pass=0,fail=0;
for(const [keys,exp,desc] of T){
  const got=run(keys);
  const ok=got===exp;
  console.log(`${ok?'PASS':'FAIL'}  ${desc}`);
  if(!ok){ console.log(`      keys: ${keys}`); console.log(`      got:  ${got}`); console.log(`      want: ${exp}`); }
  ok?pass++:fail++;
}
console.log(`\n${pass}/${pass+fail} passed`);
process.exit(fail?1:0);
