// Slothing static web IME — chewing-style UX, fully in-browser.
//  * inline conversion: the preedit shows CHINESE (live-decoded) as you type;
//    bopomofo is visible only for the syllable still being composed
//  * Enter commits the sentence directly (one step, like chewing)
//  * click any character in the preedit to fix it in place (candidate strip)
//  * auto zh/en: valid zhuyin adds one symbol per keystroke, so an overwrite
//    (impossible zhuyin) or a non-zhuyin char flips the run to English
import { AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

const REPO = 'Luigi/slothlm-34m-zhuyin-ONNX';
const TONES = 'ˊˇˋ˙';
const DACHEN = {'1':['ㄅ',0],'q':['ㄆ',0],'a':['ㄇ',0],'z':['ㄈ',0],'2':['ㄉ',0],'w':['ㄊ',0],
  's':['ㄋ',0],'x':['ㄌ',0],'e':['ㄍ',0],'d':['ㄎ',0],'c':['ㄏ',0],'r':['ㄐ',0],'f':['ㄑ',0],
  'v':['ㄒ',0],'5':['ㄓ',0],'t':['ㄔ',0],'g':['ㄕ',0],'b':['ㄖ',0],'y':['ㄗ',0],'h':['ㄘ',0],
  'n':['ㄙ',0],'u':['ㄧ',1],'j':['ㄨ',1],'m':['ㄩ',1],'8':['ㄚ',2],'i':['ㄛ',2],'k':['ㄜ',2],
  ',':['ㄝ',2],'9':['ㄞ',2],'o':['ㄟ',2],'l':['ㄠ',2],'.':['ㄡ',2],'0':['ㄢ',2],'p':['ㄣ',2],
  ';':['ㄤ',2],'/':['ㄥ',2],'-':['ㄦ',2]};
const TONEK = {'6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'};
const ROWS = [['1','2','3','4','5','6','7','8','9','0','-'],['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l',';'],['z','x','c','v','b','n','m',',','.','/']];
const strip = s => [...s].filter(c => !TONES.includes(c)).join('');
const $ = id => document.getElementById(id);

// ---- buffer state ----
// committed: [{t:'zh'|'en', v}] closed tokens; cur/rawWord: the run in progress
let committed = [], cur = ['','',''], rawWord = '', enRun = false;
// live conversion state, aligned with `committed` when pvKey matches
let pvChars = [], pvCands = [], pvKey = null, overrides = [];
let fixIndex = -1;            // preedit char being corrected (-1 = none)
const numSym = () => (cur[0]?1:0)+(cur[1]?1:0)+(cur[2]?1:0);
const hasPending = () => cur[0]||cur[1]||cur[2];
const pending = () => cur[0]+cur[1]+cur[2];
const bufKey = () => committed.map(t=>t.t+':'+t.v).join('|')+($('toneless').checked?'#TL':'');

function resetRun(){ cur=['','','']; rawWord=''; enRun=false; }
function commitRun(){
  if(enRun && rawWord) committed.push({t:'en',v:rawWord});
  else if(hasPending()) committed.push({t:'zh',v:pending()});
  else if(rawWord) committed.push({t:'en',v:rawWord});
  else { resetRun(); return; }
  overrides.push(null);
  resetRun();
}
function clearAll(){ committed=[]; overrides=[]; resetRun(); pvKey=null; fixIndex=-1; }

function feedKey(k){
  if(k===' '){ if(hasPending()||rawWord){commitRun();render();} return true; }
  if(k in TONEK){
    if(!enRun && hasPending()){committed.push({t:'zh',v:pending()+TONEK[k]});overrides.push(null);resetRun();render();return true;}
    rawWord+=k; enRun=true; render(); return true;
  }
  if(enRun){ rawWord+=k; render(); return true; }
  if(DACHEN[k]){ cur[DACHEN[k][1]]=DACHEN[k][0]; rawWord+=k;
    if(rawWord.length>numSym()) enRun=true;
    render(); return true; }
  rawWord+=k; enRun=true; render(); return true;
}
function backspace(){
  fixIndex=-1;
  if(rawWord){ rawWord=rawWord.slice(0,-1);
    if(!enRun){ cur=['','','']; for(const c of rawWord) if(DACHEN[c]) cur[DACHEN[c][1]]=DACHEN[c][0]; }
    else if(rawWord.length<=numSym()) enRun=false;
    if(!rawWord) resetRun();
  } else if(committed.length){ committed.pop(); overrides.pop(); }
  render();
}

// display text of committed token i (override > live conversion > bopomofo)
function displayFor(i){
  if(committed[i].t==='en') return committed[i].v;
  if(overrides[i]) return overrides[i];
  if(pvKey===bufKey() && pvChars[i]!=null) return pvChars[i];
  return committed[i].v;   // bopomofo until the decode lands
}
function sentenceText(){
  let out='';
  committed.forEach((tok,i)=>{const v=displayFor(i);
    if(tok.t==='en') out+=(out&&!out.endsWith(' ')?' ':'')+v+' '; else out+=v;});
  return out.trim();
}

// ---- rendering: one preedit line, chewing-style ----
function render(){
  const pre=$('pre'); pre.innerHTML='';
  committed.forEach((tok,i)=>{
    const span=document.createElement('span');
    span.className='pchar'+(fixIndex===i?' fix':'')+(tok.t==='en'?' en':'');
    span.textContent=(tok.t==='en'?' '+displayFor(i)+' ':displayFor(i));
    if(tok.t==='zh'){ span.onclick=()=>{ fixIndex=(fixIndex===i?-1:i); render(); }; }
    pre.appendChild(span);
  });
  const run=enRun?rawWord:pending();
  if(run){ const tail=document.createElement('span'); tail.className='ptail';
    tail.textContent=(enRun?' ':'')+run; pre.appendChild(tail); }
  pre.classList.toggle('empty', !committed.length && !run);
  if(!committed.length && !run) pre.textContent='​';

  // candidate strip for the char being fixed
  const strip2=$('cands'); strip2.innerHTML='';
  if(fixIndex>=0 && pvKey===bufKey() && pvCands[fixIndex]){
    pvCands[fixIndex].forEach(c=>{
      const b=document.createElement('button');
      b.className='cand'+(displayFor(fixIndex)===c?' sel':'');
      b.textContent=c;
      b.onclick=()=>{ overrides[fixIndex]=c; fixIndex=-1; render(); };
      strip2.appendChild(b);
    });
  }
  if(ready) $('hint').textContent = committed.length||run
    ? '⏎ 上字　點字改字　⌫ 刪除' : '直接打注音或英文，自動辨識';
  schedulePreview();
}

// ---- live conversion (debounced, serialized, stale-dropped) ----
let previewTimer=null, previewBusy=false, previewGen=0;
function schedulePreview(){
  if(!ready) return;
  clearTimeout(previewTimer);
  if(!committed.some(t=>t.t==='zh')){ pvKey=bufKey(); pvChars=committed.map(t=>t.v); pvCands=committed.map(t=>[t.v]); return; }
  if(pvKey===bufKey()) return;                 // already current
  previewTimer=setTimeout(()=>{ runPreview(); },200);
}
async function runPreview(){
  if(previewBusy) return;
  previewBusy=true;
  const key=bufKey(), gen=++previewGen;
  try{
    const toneless=$('toneless').checked;
    const zh=committed.filter(t=>t.t==='zh').map(t=>toneless?strip(t.v):t.v);
    const r=zh.length?await decodeZh(zh):{chars:[],cands:[]};
    if(gen===previewGen && key===bufKey()){
      pvChars=[]; pvCands=[]; let zc=0;
      for(const tok of committed){
        if(tok.t==='zh'){ pvChars.push(r.chars[zc]); pvCands.push(r.cands[zc]); zc++; }
        else { pvChars.push(tok.v); pvCands.push([tok.v]); }
      }
      pvKey=key; render();
    }
  }catch(e){ console.error(e); }
  previewBusy=false;
  if(pvKey!==bufKey()) schedulePreview();
}
async function ensureConverted(){
  if(hasPending()||rawWord) commitRun();
  if(!committed.length) return false;
  if(pvKey!==bufKey()){
    $('hint').textContent='解碼中…';
    clearTimeout(previewTimer);
    while(previewBusy) await new Promise(r=>setTimeout(r,50));
    if(pvKey!==bufKey()) await runPreview();
    while(previewBusy) await new Promise(r=>setTimeout(r,50));
  }
  return pvKey===bufKey();
}
async function commitSentence(){
  if(!ready) return;
  if(!(await ensureConverted())) return;
  $('out').textContent+=sentenceText();
  clearAll(); render();
}

// ---- model + decode ----
let tokenizer, model, ready=false;
const tonal={};
function tid(ch){const ids=tokenizer.encode(ch,{add_special_tokens:false});return ids.length===1?ids[0]:null;}
function candChars(syl){let chars=tonal[syl];
  if(!chars){const base=strip(syl);chars=[];for(const k in tonal)if(strip(k)===base)for(const c of tonal[k])if(!chars.includes(c))chars.push(c);}
  return chars.length?chars:[syl];}
function candSet(syl){const out=[],ids=[];for(const c of candChars(syl)){const i=tid(c);if(i!=null&&!ids.includes(i)){ids.push(i);out.push(c);}}return{chars:out,ids};}
class Masker extends LogitsProcessor{constructor(p,s){super();this.p=p;this.s=s;}
  _call(ii,l){const st=ii[0].length-this.s;if(st>=0&&st<this.p.length){const a=new Set(this.p[st]);const d=l.data;for(let i=0;i<d.length;i++)if(!a.has(i))d[i]=-Infinity;}return l;}}
async function decodeZh(syls){
  const sets=syls.map(candSet), posIds=sets.map(s=>s.ids);
  const prompt='<|im_start|>system\n注音轉繁體中文。<|im_end|>\n<|im_start|>user\n'+syls.join(' ')+'<|im_end|>\n<|im_start|>assistant\n';
  const enc=tokenizer(prompt);const start=enc.input_ids.dims[1];
  const out=await model.generate({...enc,max_new_tokens:posIds.length+1,do_sample:false,
    logits_processor:new LogitsProcessorList([new Masker(posIds,start)])});
  return{chars:[...tokenizer.decode(out.tolist()[0].slice(start),{skip_special_tokens:true})],cands:sets.map(s=>s.chars)};
}

// ---- keyboard UI ----
const kb=$('kb');
ROWS.forEach(row=>{const r=document.createElement('div');r.className='krow';
  row.forEach(key=>{const b=document.createElement('button');b.className='key';
    const sym=DACHEN[key]?DACHEN[key][0]:(key in TONEK?TONEK[key]:'');
    b.innerHTML='<span class="k">'+key+'</span><span class="s">'+sym+'</span>';
    b.onclick=()=>feedKey(key);r.appendChild(b);});kb.appendChild(r);});
const r=document.createElement('div');r.className='krow';
const sp=document.createElement('button');sp.className='key wide';sp.textContent='空白（一聲）';
sp.onclick=()=>{ if(hasPending()||rawWord) feedKey(' '); };
const ent=document.createElement('button');ent.className='key wide2';ent.textContent='⏎ 上字';
ent.onclick=()=>commitSentence();
const bs=document.createElement('button');bs.className='key';bs.style.width='64px';
bs.innerHTML='<span class="s">⌫</span>';
bs.onclick=()=>{ if(committed.length||hasPending()||rawWord) backspace(); };
r.appendChild(sp);r.appendChild(ent);r.appendChild(bs);kb.appendChild(r);

document.addEventListener('keydown',e=>{
  if(e.ctrlKey||e.altKey||e.metaKey)return;const k=e.key;
  if(k==='Enter'){ if(committed.length||hasPending()||rawWord){commitSentence();e.preventDefault();} }
  else if(k===' '){ if(hasPending()||rawWord){feedKey(' ');e.preventDefault();}
                    else if(committed.length){e.preventDefault();} }
  else if(k==='Backspace'){ if(committed.length||hasPending()||rawWord){backspace();e.preventDefault();} }
  else if(k==='Escape'){ clearAll(); render(); }
  else if(k.length===1&&(DACHEN[k]||k in TONEK||/[A-Za-z0-9]/.test(k))){ feedKey(k); e.preventDefault(); }
});
$('commit').onclick=()=>commitSentence();
$('clear').onclick=()=>{ clearAll(); render(); };
$('toneless').onchange=()=>{ pvKey=null; render(); };

(async function init(){
  const txt=await (await fetch('./phonetic_table.tsv')).text();
  for(const line of txt.split('\n')){const t=line.indexOf('\t');if(t<0)continue;tonal[line.slice(0,t)]=[...line.slice(t+1)];}
  tokenizer=await AutoTokenizer.from_pretrained(REPO);
  model=await AutoModelForCausalLM.from_pretrained(REPO,{dtype:'q8'});
  ready=true; render();
})().catch(e=>{console.error(e);$('hint').textContent='模型載入失敗：'+e.message;});
render();
