#!/usr/bin/env node
// Test the continuous re-segmenter (segment.js) — the global replacement for
// the per-case FSM heuristics. Runs raw keystrings through segment() and checks
// the zh/en token boundaries.  node space-static/test-segment.mjs
import fs from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';
import {makeSegmenter} from './segment.js';
const DIR=path.dirname(fileURLToPath(import.meta.url));

const TONES='ˊˇˋ˙';
const strip=s=>[...s].filter(c=>!TONES.includes(c)).join('');
const DACHEN={'1':['ㄅ',0],'q':['ㄆ',0],'a':['ㄇ',0],'z':['ㄈ',0],'2':['ㄉ',0],'w':['ㄊ',0],'s':['ㄋ',0],'x':['ㄌ',0],'e':['ㄍ',0],'d':['ㄎ',0],'c':['ㄏ',0],'r':['ㄐ',0],'f':['ㄑ',0],'v':['ㄒ',0],'5':['ㄓ',0],'t':['ㄔ',0],'g':['ㄕ',0],'b':['ㄖ',0],'y':['ㄗ',0],'h':['ㄘ',0],'n':['ㄙ',0],'u':['ㄧ',1],'j':['ㄨ',1],'m':['ㄩ',1],'8':['ㄚ',2],'i':['ㄛ',2],'k':['ㄜ',2],',':['ㄝ',2],'9':['ㄞ',2],'o':['ㄟ',2],'l':['ㄠ',2],'.':['ㄡ',2],'0':['ㄢ',2],'p':['ㄣ',2],';':['ㄤ',2],'/':['ㄥ',2],'-':['ㄦ',2]};
const TONEK={'6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'};
const validBase=new Set();
for(const l of fs.readFileSync(path.join(DIR,'phonetic_table.tsv'),'utf-8').split('\n')){
  const t=l.indexOf('\t'); if(t<0)continue; validBase.add(strip(l.slice(0,t)));
}
const segment=makeSegmenter(DACHEN,TONEK,validBase);
const show=keys=>segment(keys).map(t=>t.t+':'+t.v).join(' | ');

const T=[
  ['python5k4ek7',   'en:python | zh:ㄓㄜˋ | zh:ㄍㄜ˙',        'English then zhuyin (這個)'],
  ['driving5j;4dj;4','en:driving | zh:ㄓㄨㄤˋ | zh:ㄎㄨㄤˋ',   'English then zhuyin (狀況)'],
  ['happya87',       'en:happy | zh:ㄇㄚ˙',                    'English then neutral-tone 嗎'],
  ['happya8',        'en:happy | zh:ㄇㄚ',                     'English then tone-1 (媽/嗎, no tone key)'],
  ['5k4python',      'zh:ㄓㄜˋ | en:python',                   'zhuyin then English'],
  ['test123',        'en:test123',                            'English+digits stays English (3=tone trap)'],
  ['hello',          'en:hello',                              'pure English'],
  ['is',             'en:is',                                 'short English that is valid-ish zhuyin'],
  ['he',             'en:he',                                 'short English'],
  ['ishe',           'en:ishe',                               '"is he" run stays English'],
  ['5k4',            'zh:ㄓㄜˋ',                               'plain zhuyin syllable'],
  ['rm,6',           'zh:ㄐㄩㄝˊ',                             '覺 (3-symbol + tone)'],
  ['w8 ',            'zh:ㄊㄚ',                                '他 tone-1 (trailing space stripped by caller)'],
  ['api2u4',         'en:api | zh:ㄉㄧˋ',                      'English then 地/弟'],
  ['model',          'en:model',                              'unknown English word not chopped (was mod+ㄍㄠ)'],
  ['world',          'en:world',                              'unknown English word not chopped (was ㄊㄟ+rld)'],
  ['banana',         'en:banana',                             'unknown English word not chopped'],
  ['ek',             'zh:ㄍㄜ',                                'standalone pure-letter syllable 哥 (post-process)'],
  ['vp3',            'zh:ㄒㄣˇ',                               'typo syllable + tone accepted (decoder repairs)'],
  ['ji3vp3',         'zh:ㄨㄛˇ | zh:ㄒㄣˇ',                    'typo syllable in zhuyin context'],
  ['Python',         'en:Python',                              'capital preserved, word whole'],
  ['ji3m/4Python',   'zh:ㄨㄛˇ | zh:ㄩㄥˋ | en:Python',        'capital mid-stream in auto zh/en'],
  ['iPhone',         'en:iPhone',                              'capital not first, still one word'],
  ['do',             'zh:ㄎㄟ',                                 'zhuyin-wins: word that IS a syllable -> zhuyin (Shift for English)'],
  ["let's",          "en:let's",                               'apostrophe rides the English run'],
  ['upgjbj4',        'zh:ㄧㄣ | zh:ㄕㄨ | zh:ㄖㄨˋ',            'toneless zh run stays zhuyin (音輸入)'],
  ['upgj',           'zh:ㄧㄣ | zh:ㄕㄨ',                       'confident multi-syllable run -> zhuyin'],
  ['7-11',           'en:7-11',                                'digit-hyphen-digit literal (not 7兒11)'],
  ['a-b',            'en:a-b',                                 'letter-hyphen-letter literal English'],
  ['0912-345',       'en:0912-345',                            'phone-number hyphen stays literal'],
  ['5k4-',           'zh:ㄓㄜˋ | zh:ㄦ',                       'trailing ㄦ preserved (這兒)'],
  ['sm3-6',          'zh:ㄋㄩˇ | zh:ㄦˊ',                      'ㄦˊ preserved between tone digits (女兒)'],
];
let pass=0,fail=0;
for(const [keys,exp,desc] of T){
  const got=show(keys.replace(/ $/,''));  // caller strips a trailing space boundary
  const ok=got===exp; ok?pass++:fail++;
  console.log(`${ok?'PASS':'FAIL'}  ${desc}`);
  if(!ok){ console.log(`      keys ${JSON.stringify(keys)}`); console.log(`      got  ${got}`); console.log(`      want ${exp}`); }
}
console.log(`\n${pass}/${pass+fail} passed`);
process.exit(fail?1:0);
