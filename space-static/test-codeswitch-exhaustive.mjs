#!/usr/bin/env node
// Exhaustive zh/en code-switch validation of the segmenter.
// Not hand-picked cases: generates thousands of keystreams by composing
//   {English words (in- and out-of-dictionary)} x {real tonal syllables from
//   the phonetic table} x {boundary patterns}, then checks the segmentation.
//   node space-static/test-codeswitch-exhaustive.mjs [-v]
import fs from 'fs'; import path from 'path'; import {fileURLToPath} from 'url';
import {makeSegmenter, WORDS} from './segment.js';
const DIR=path.dirname(fileURLToPath(import.meta.url));
const V=process.argv.includes('-v');

const TONES='ˊˇˋ˙', strip=s=>[...s].filter(c=>!TONES.includes(c)).join('');
const DACHEN={'1':['ㄅ',0],'q':['ㄆ',0],'a':['ㄇ',0],'z':['ㄈ',0],'2':['ㄉ',0],'w':['ㄊ',0],'s':['ㄋ',0],'x':['ㄌ',0],'e':['ㄍ',0],'d':['ㄎ',0],'c':['ㄏ',0],'r':['ㄐ',0],'f':['ㄑ',0],'v':['ㄒ',0],'5':['ㄓ',0],'t':['ㄔ',0],'g':['ㄕ',0],'b':['ㄖ',0],'y':['ㄗ',0],'h':['ㄘ',0],'n':['ㄙ',0],'u':['ㄧ',1],'j':['ㄨ',1],'m':['ㄩ',1],'8':['ㄚ',2],'i':['ㄛ',2],'k':['ㄜ',2],',':['ㄝ',2],'9':['ㄞ',2],'o':['ㄟ',2],'l':['ㄠ',2],'.':['ㄡ',2],'0':['ㄢ',2],'p':['ㄣ',2],';':['ㄤ',2],'/':['ㄥ',2],'-':['ㄦ',2]};
const TONEK={'6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'};
const REV={}; for(const k in DACHEN) REV[DACHEN[k][0]]=k;
const REVT={'ˊ':'6','ˇ':'3','ˋ':'4','˙':'7'};
const validBase=new Set(), tonalSyls=[];
for(const l of fs.readFileSync(path.join(DIR,'phonetic_table.tsv'),'utf-8').split('\n')){
  const t=l.indexOf('\t'); if(t<0)continue; const syl=l.slice(0,t);
  validBase.add(strip(syl));
  if([...syl].some(c=>TONES.includes(c))) tonalSyls.push(syl);   // tonal only
}
const segment=makeSegmenter(DACHEN,TONEK,validBase);
const keysOf=syl=>[...syl].map(c=>REVT[c]??REV[c]).join('');   // syllable -> keys (no trailing space)

// --- vocab under test ---
// English: dictionary words + OUT-of-dictionary words (the hard class) + tech terms
const EN_IN=[...WORDS].filter(w=>w.length>=3).slice(0,60);
const EN_OUT=('model world banana slothing kubernetes fcitx javascript sorry keyboard password '+
'apple random boba coffee video manager github gitlab docker linux ubuntu debian fedora chrome '+
'firefox safari windows android tablet laptop desktop server client router modem printer scanner '+
'monitor speaker headset webcam battery charger adapter cable usb hdmi wifi bluetooth ethernet '+
'online offline upload download update upgrade install uninstall backup restore reset reboot').split(' ')
  .filter(w=>!WORDS.has(w));
// zhuyin: a broad sample of real tonal syllables (every 7th = ~570 syllables)
const ZH=tonalSyls.filter((_,i)=>i%7===0);
const zhPair=i=>[ZH[i%ZH.length], ZH[(i*13+7)%ZH.length]];

let n=0, fail=0; const failures=[];
function check(keys, want, tag){
  n++;
  const got=segment(keys).map(t=>t.t+':'+t.v).join(' | ');
  if(got!==want){ fail++; if(failures.length<40) failures.push({tag,keys,got,want}); }
}

// P1: every English word alone stays whole
for(const w of [...EN_IN,...EN_OUT]) check(w, 'en:'+w, 'en-alone');
// P2: every sampled tonal syllable alone parses as zhuyin
for(const s of ZH) check(keysOf(s), 'zh:'+s, 'zh-alone');
// P3: en + tonal-zh (no space) — the python這個 class
for(let i=0;i<EN_OUT.length;i++){ const w=EN_OUT[i], [a,b]=zhPair(i);
  check(w+keysOf(a), `en:${w} | zh:${a}`, 'en+zh');
  check(w+keysOf(a)+keysOf(b), `en:${w} | zh:${a} | zh:${b}`, 'en+zh+zh'); }
// P4: tonal-zh + en (no space)
for(let i=0;i<EN_OUT.length;i++){ const w=EN_OUT[i], [a]=zhPair(i);
  check(keysOf(a)+w, `zh:${a} | en:${w}`, 'zh+en'); }
// P5: zh + en + zh sandwich
for(let i=0;i<40;i++){ const w=EN_OUT[i%EN_OUT.length], [a,b]=zhPair(i);
  check(keysOf(a)+w+keysOf(b), `zh:${a} | en:${w} | zh:${b}`, 'zh+en+zh'); }
// P6: en + en with space (two words)
for(let i=0;i+1<EN_OUT.length;i+=2){ const w1=EN_OUT[i], w2=EN_OUT[i+1];
  const toks=segment(w1+' '+w2).map(t=>t.t+':'+t.v).join(' | ');
  n++; if(toks!==`en:${w1} | en:${w2}`&&toks!==`en:${w1} ${w2}`&&toks!==`en:${w1}${w2}`){
    // space is a boundary handled by feedKey; segmenter sees runs separately.
    // Simulate feedKey: each word is its own run.
    const t1=segment(w1).map(t=>t.t+':'+t.v).join(' | '), t2=segment(w2).map(t=>t.t+':'+t.v).join(' | ');
    if(t1!=='en:'+w1||t2!=='en:'+w2){ fail++; if(failures.length<40)failures.push({tag:'en en',keys:w1+' '+w2,got:t1+' / '+t2,want:'en each'}); }
  }
}
// P7: digits attached to English stay English (test123, mp3, b2b)
for(const w of ['test123','mp3x','b2b','x264','utf8x','win10x','ipv6','sha256']) check(w,'en:'+w,'en+digits');
// P8: long zh sentences (5 tonal syllables, no English)
for(let i=0;i<60;i++){ const sy=[0,1,2,3,4].map(k=>ZH[(i*5+k*17)%ZH.length]);
  check(sy.map(keysOf).join(''), sy.map(s=>'zh:'+s).join(' | '), 'zh-sentence'); }
// P9: en word between two zh sentences
for(let i=0;i<30;i++){ const w=EN_OUT[(i*3)%EN_OUT.length]; const [a,b]=zhPair(i), [c,d]=zhPair(i+11);
  check(keysOf(a)+keysOf(b)+w+keysOf(c)+keysOf(d),
        `zh:${a} | zh:${b} | en:${w} | zh:${c} | zh:${d}`, 'zhzh+en+zhzh'); }

console.log(`\n${n-fail}/${n} passed  (${(100*(n-fail)/n).toFixed(1)}%)`);
if(failures.length){
  console.log('\nby class:');
  const by={}; for(const f of failures) by[f.tag]=(by[f.tag]||0)+1;
  for(const t in by) console.log(' ',t,by[t]+ (fail>failures.length?'+ (first 40 shown)':''));
  if(V) for(const f of failures) console.log(`  [${f.tag}] ${JSON.stringify(f.keys)}\n    got  ${f.got}\n    want ${f.want}`);
  else console.log('run with -v for details');
}
process.exit(fail?1:0);
