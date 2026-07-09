// Slothing static web IME — chewing-style UX, fully in-browser.
//  * inline conversion: preedit shows CHINESE as you type (live decode);
//    bopomofo visible only for the syllable being composed
//  * Enter commits; Space = tone 1
//  * preedit cursor: ←/→ move, type/delete mid-sentence
//  * fix a char: click it or press ↓ at the cursor; paged candidates,
//    number keys 1-9 select (chewing-style)
//  * punctuation: Shift+, Shift+. Shift+/ Shift+1 Shift+; → ，。？！：
//  * session learning: your picks become the default for that syllable
//  * auto zh/en: impossible-zhuyin keystrokes flip the run to English
import { AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, Tensor, env }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

// iOS Safari has no SharedArrayBuffer here (HF static Spaces send no COOP/COEP),
// so threaded WASM fails to instantiate ("Internal error"). Force single-thread,
// no worker-proxy -> the plain WASM path that loads reliably on mobile Safari.
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.proxy = false;

const REPO = 'Luigi/slothlm-34m-zhuyin-ONNX';
const TONES = 'ˊˇˋ˙';
const DACHEN = {'1':['ㄅ',0],'q':['ㄆ',0],'a':['ㄇ',0],'z':['ㄈ',0],'2':['ㄉ',0],'w':['ㄊ',0],
  's':['ㄋ',0],'x':['ㄌ',0],'e':['ㄍ',0],'d':['ㄎ',0],'c':['ㄏ',0],'r':['ㄐ',0],'f':['ㄑ',0],
  'v':['ㄒ',0],'5':['ㄓ',0],'t':['ㄔ',0],'g':['ㄕ',0],'b':['ㄖ',0],'y':['ㄗ',0],'h':['ㄘ',0],
  'n':['ㄙ',0],'u':['ㄧ',1],'j':['ㄨ',1],'m':['ㄩ',1],'8':['ㄚ',2],'i':['ㄛ',2],'k':['ㄜ',2],
  ',':['ㄝ',2],'9':['ㄞ',2],'o':['ㄟ',2],'l':['ㄠ',2],'.':['ㄡ',2],'0':['ㄢ',2],'p':['ㄣ',2],
  ';':['ㄤ',2],'/':['ㄥ',2],'-':['ㄦ',2]};
const TONEK = {'6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'};
const PUNCT = {'<':'，','>':'。','?':'？','!':'！',':':'：','"':'；','(':'（',')':'）'};
const ROWS = [['1','2','3','4','5','6','7','8','9','0','-'],['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l',';'],['z','x','c','v','b','n','m',',','.','/']];
const PAGE = 9;
const strip = s => [...s].filter(c => !TONES.includes(c)).join('');
const $ = id => document.getElementById(id);

// ---- buffer state ----
// committed: [{t:'zh'|'en'|'punct', v}]; cursor = insertion index into it.
// cur/rawWord: the run being typed (lives at the cursor).
let committed = [], overrides = [], cursor = 0;
let cur = ['','',''], rawWord = '', enRun = false;
let pvChars = [], pvCands = [], pvKey = null;
let fix = -1, fixPage = 0;              // char being corrected
const numSym = () => (cur[0]?1:0)+(cur[1]?1:0)+(cur[2]?1:0);
const hasPending = () => cur[0]||cur[1]||cur[2];
const pending = () => cur[0]+cur[1]+cur[2];
const hasRun = () => hasPending()||rawWord;
const bufKey = () => committed.map(t=>t.t+':'+t.v).join('|')+($('toneless').checked?'#TL':'');

// session learning (persisted): syllable -> last picked char
let learn = {};
try{ learn = JSON.parse(localStorage.getItem('slothing-learn')||'{}'); }catch(e){}
const saveLearn = ()=>{ try{localStorage.setItem('slothing-learn',JSON.stringify(learn));}catch(e){} };

const validBase=new Set();   // toneless syllable bases, filled from the table
// Longest trailing run of keys (<=3) that forms a valid zhuyin syllable, so an
// English run followed by zhuyin (no space) can be split: 'python5k' + tone ->
// ['python', 'ㄓㄜ'].  Returns [englishPrefix, syllableBopomofo] or null.
function splitTrailingSyllable(raw){
  for(let L=Math.min(3,raw.length);L>=1;L--){
    let bopo='',ok=true;
    for(const c of raw.slice(-L)){ if(!DACHEN[c]){ok=false;break;} bopo+=DACHEN[c][0]; }
    if(ok && validBase.has(bopo)) return [raw.slice(0,raw.length-L), bopo];
  }
  return null;
}
function resetRun(){ cur=['','','']; rawWord=''; enRun=false; }
function insertTok(tok){ committed.splice(cursor,0,tok); overrides.splice(cursor,0,null); cursor++; }
function commitRun(){
  if(enRun && rawWord) insertTok({t:'en',v:rawWord});
  else if(hasPending()) insertTok({t:'zh',v:pending()});
  else if(rawWord) insertTok({t:'en',v:rawWord});
  resetRun();
}
function clearAll(){ committed=[];overrides=[];cursor=0;resetRun();pvKey=null;fix=-1; }

function feedKey(k){
  fix=-1;
  if(k in PUNCT){ if(hasRun())commitRun(); insertTok({t:'punct',v:PUNCT[k]}); render(); return true; }
  if(k===' '){ if(hasRun()){commitRun();render();} return true; }
  if(k in TONEK){
    if(!enRun && hasPending()){ insertTok({t:'zh',v:pending()+TONEK[k]}); resetRun(); render(); return true; }
    if(enRun){
      // English run then a tone => the trailing keys were zhuyin (tones only
      // exist in zhuyin). Split so 'python這個' works with no space between.
      const sp=splitTrailingSyllable(rawWord);
      if(sp){
        if(sp[0]) insertTok({t:'en',v:sp[0]});
        insertTok({t:'zh',v:sp[1]+TONEK[k]}); resetRun(); render(); return true;
      }
    }
    rawWord+=k; enRun=true; render(); return true;
  }
  if(enRun){ rawWord+=k; render(); return true; }
  if(DACHEN[k]){ cur[DACHEN[k][1]]=DACHEN[k][0]; rawWord+=k;
    if(rawWord.length>numSym()) enRun=true; render(); return true; }
  rawWord+=k; enRun=true; render(); return true;
}
function backspace(){
  fix=-1;
  if(rawWord){ rawWord=rawWord.slice(0,-1);
    if(!enRun){ cur=['','','']; for(const c of rawWord) if(DACHEN[c]) cur[DACHEN[c][1]]=DACHEN[c][0]; }
    else if(rawWord.length<=numSym()) enRun=false;
    if(!rawWord) resetRun();
  } else if(cursor>0){ committed.splice(cursor-1,1); overrides.splice(cursor-1,1); cursor--; }
  render();
}
function moveCursor(d){
  if(hasRun()) commitRun();          // close the run before moving
  fix=-1;
  cursor=Math.max(0,Math.min(committed.length,cursor+d));
  render();
}
function openFix(i){
  if(i<0||i>=committed.length||committed[i].t!=='zh') return;
  if(pvKey!==bufKey()) return;       // wait for decode
  fix=i; fixPage=0; phrase=null; phraseBusy=false;
  const sel=displayFor(i), cands=pvCands[i]||[];
  const at=cands.indexOf(sel); if(at>=0) fixPage=Math.floor(at/PAGE);
  render();
  buildPhrases(i);                   // async: 2-char phrase alternatives
}

// Phrase-level candidates (LLM-only, ranked by the model's joint probability,
// noise tail thresholded out): score every 2-char phrase for the focused
// syllable + the next by P(first)·P(second|first) from direct forward passes,
// keep only phrases clearly above the tail -> real words like chewing, no
// lexicon. Context-free on the syllable pair (like a phrase dictionary).
let phrase=null, phraseBusy=false, phraseFor=-1;
async function forwardLogits(ids){
  const n=ids.length;
  const input=new Tensor('int64',ids.map(BigInt),[1,n]);
  const attn=new Tensor('int64',new Array(n).fill(1n),[1,n]);
  const pos=new Tensor('int64',ids.map((_,i)=>BigInt(i)),[1,n]);
  const out=await model({input_ids:input,attention_mask:attn,position_ids:pos});
  return out.logits;
}
function softmaxOver(data,base,ids){let mx=-Infinity;for(const id of ids)if(data[base+id]>mx)mx=data[base+id];
  let z=0;const e={};for(const id of ids){e[id]=Math.exp(data[base+id]-mx);z+=e[id];}return{e,z};}
async function buildPhrases(i){
  const next=i+1;
  if(next>=committed.length||committed[next].t!=='zh'){ return; }
  const tl=$('toneless').checked;
  const s0=tl?strip(committed[i].v):committed[i].v, s1=tl?strip(committed[next].v):committed[next].v;
  const c0=candSet(s0), c1=candSet(s1);
  if(!c0.ids.length||!c1.ids.length) return;
  phraseBusy=true; phraseFor=i; if(fix===i) render();
  try{
    const prompt='<|im_start|>system\n注音轉繁體中文。<|im_end|>\n<|im_start|>user\n'+s0+' '+s1+'<|im_end|>\n<|im_start|>assistant\n';
    const pids=[...tokenizer(prompt).input_ids.data].map(Number);
    const L0=await forwardLogits(pids); if(phraseFor!==i) return;
    const V=L0.dims[2], q0=(L0.dims[1]-1)*V;
    const {e:e0,z:z0}=softmaxOver(L0.data,q0,c0.ids);
    const firsts=c0.ids.map((id,k)=>({id,c:c0.chars[k],p:e0[id]/z0})).sort((a,b)=>b.p-a.p).slice(0,5);
    const scored=[];
    for(const f of firsts){
      if(phraseFor!==i) return;
      const L1=await forwardLogits([...pids,f.id]); const q1=(L1.dims[1]-1)*V;
      const {e:e1,z:z1}=softmaxOver(L1.data,q1,c1.ids);
      let bp=-1,bc=c1.chars[0];
      c1.ids.forEach((id,k)=>{const pp=e1[id]/z1; if(pp>bp){bp=pp;bc=c1.chars[k];}});
      scored.push({ph:f.c+bc, j:f.p*bp});
    }
    scored.sort((a,b)=>b.j-a.j);
    const top=scored[0]?scored[0].j:0;
    const kept=scored.filter(x=>x.j>=Math.max(0.06, 0.15*top)).map(x=>x.ph);
    if(phraseFor===i){ phrase=[...new Set(kept)]; if(fix===i) render(); }
  }catch(e){ console.error(e); }
  finally{ if(phraseFor===i) phraseBusy=false; }
}
function pickCand(j){                 // j is index within current page
  const cands=pvCands[fix]||[];
  const idx=fixPage*PAGE+j;
  if(idx>=cands.length) return;
  overrides[fix]=cands[idx];
  const syl=$('toneless').checked?strip(committed[fix].v):committed[fix].v;
  learn[syl]=cands[idx]; saveLearn();
  fix=-1; phrase=null; render();
}
function pickPhrase(p){               // p = 2-char phrase for [fix, fix+1]
  const chars=[...p];
  overrides[fix]=chars[0];
  // next zh token
  let n=fix+1; while(n<committed.length&&committed[n].t!=='zh') n++;
  if(n<committed.length&&chars[1]) overrides[n]=chars[1];
  const tl=$('toneless').checked;
  learn[tl?strip(committed[fix].v):committed[fix].v]=chars[0];
  if(n<committed.length&&chars[1]) learn[tl?strip(committed[n].v):committed[n].v]=chars[1];
  saveLearn();
  fix=-1; phrase=null; render();
}

function displayFor(i){
  const tok=committed[i];
  if(tok.t!=='zh') return tok.v;
  if(overrides[i]) return overrides[i];
  if(pvKey===bufKey() && pvChars[i]!=null) return pvChars[i];
  return tok.v;                        // bopomofo until decode lands
}
function sentenceText(){
  let out='';
  committed.forEach((tok,i)=>{const v=displayFor(i);
    if(tok.t==='en') out+=(out&&!/[\s（]$/.test(out)?' ':'')+v+' ';
    else out+=v;});
  return out.replace(/\s+([，。？！：；）])/g,'$1').trim();
}

// ---- rendering ----
function render(){
  const pre=$('pre'); pre.innerHTML='';
  const caret=()=>{const c=document.createElement('span');c.className='caret';return c;};
  committed.forEach((tok,i)=>{
    if(i===cursor && !hasRun()) pre.appendChild(caret());
    const span=document.createElement('span');
    span.className='pchar'+(fix===i?' fixsel':'')+(tok.t!=='zh'?' en':'');
    span.textContent=(tok.t==='en'?' '+displayFor(i)+' ':displayFor(i));
    if(tok.t==='zh') span.onclick=()=>{ fix===i?(fix=-1,render()):openFix(i); };
    pre.appendChild(span);
    if(i===committed.length-1 && cursor===committed.length && !hasRun()) pre.appendChild(caret());
  });
  if(!committed.length && !hasRun()) pre.appendChild(caret());
  if(hasRun()){
    // the run renders at the cursor position
    const nodes=[...pre.childNodes];
    const tail=document.createElement('span'); tail.className='ptail';
    tail.textContent=(enRun?' ':'')+(enRun?rawWord:pending());
    // insert after cursor-1 tokens
    let anchor=null, count=0;
    for(const n of nodes){ if(n.classList&&n.classList.contains('pchar')){count++; if(count===cursor){anchor=n;break;}} }
    if(cursor===0) pre.insertBefore(tail,pre.firstChild);
    else if(anchor&&anchor.nextSibling) pre.insertBefore(tail,anchor.nextSibling);
    else pre.appendChild(tail);
  }
  pre.classList.toggle('empty',!committed.length&&!hasRun());

  // phrase candidates (2-char), when available
  const phEl=$('phrases'); phEl.innerHTML='';
  if(fix>=0 && phrase && phrase.length){
    const lbl=document.createElement('span'); lbl.className='pg'; lbl.textContent='詞';
    phEl.appendChild(lbl);
    const seen=new Set();
    phrase.forEach(p=>{ if(seen.has(p))return; seen.add(p);
      const b=document.createElement('button'); b.className='cand ph';
      b.textContent=p; b.onclick=()=>pickPhrase(p); phEl.appendChild(b); });
  } else if(fix>=0 && phraseBusy){
    const s=document.createElement('span'); s.className='pg'; s.textContent='組詞中…'; phEl.appendChild(s);
  }

  // paged candidate strip
  const stripEl=$('cands'); stripEl.innerHTML='';
  if(fix>=0 && pvKey===bufKey() && pvCands[fix]){
    const cands=pvCands[fix], pages=Math.ceil(cands.length/PAGE);
    if(pages>1){ const bp=document.createElement('button'); bp.className='cand nav'; bp.textContent='‹';
      bp.onclick=()=>{fixPage=(fixPage-1+pages)%pages;render();}; stripEl.appendChild(bp); }
    cands.slice(fixPage*PAGE,fixPage*PAGE+PAGE).forEach((c,j)=>{
      const b=document.createElement('button');
      b.className='cand'+(displayFor(fix)===c?' sel':'');
      b.innerHTML='<span class="n">'+(j+1)+'</span>'+c;
      b.onclick=()=>pickCand(j);
      stripEl.appendChild(b);
    });
    if(pages>1){ const bn=document.createElement('button'); bn.className='cand nav'; bn.textContent='›';
      bn.onclick=()=>{fixPage=(fixPage+1)%pages;render();}; stripEl.appendChild(bn);
      const pg=document.createElement('span'); pg.className='pg'; pg.textContent=(fixPage+1)+'/'+pages;
      stripEl.appendChild(pg); }
  }
  if(ready) $('hint').textContent = fix>=0 ? '1-9 選字　↑↓ 翻頁　Esc 取消'
    : (committed.length||hasRun() ? '⏎ 上字　←→ 游標　↓ 選字　點字改字' : '直接打注音或英文，自動辨識；Shift+，。？！ 輸入標點');
  schedulePreview();
}

// ---- live conversion ----
let previewTimer=null, previewBusy=false, previewGen=0;
function schedulePreview(){
  if(!ready) return;
  clearTimeout(previewTimer);
  if(!committed.some(t=>t.t==='zh')){ pvKey=bufKey(); pvChars=committed.map(t=>t.v); pvCands=committed.map(t=>[t.v]); return; }
  if(pvKey===bufKey()) return;
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
      pvChars=[];pvCands=[];let zc=0;
      for(const tok of committed){
        if(tok.t==='zh'){
          const syl=toneless?strip(tok.v):tok.v;
          const cands=r.cands[zc];
          let ch=r.chars[zc]; if(ch==null||!cands.includes(ch)) ch=cands[0];  // never fall back to bopomofo
          if(learn[syl]&&cands.includes(learn[syl])) ch=learn[syl];  // learned pick wins
          pvChars.push(ch); pvCands.push(cands); zc++;
        } else { pvChars.push(tok.v); pvCands.push([tok.v]); }
      }
      pvKey=key; render();
    }
  }catch(e){ console.error(e); }
  previewBusy=false;
  if(pvKey!==bufKey()) schedulePreview();
}
async function ensureConverted(){
  if(hasRun()) commitRun();
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
  // insert into the editable output at its cursor (the output is a real
  // document: click into it to move the caret / fix committed text).
  const ta=$('out'), t=sentenceText();
  const a=(ta.selectionStart!=null)?ta.selectionStart:ta.value.length;
  const b=(ta.selectionEnd!=null)?ta.selectionEnd:a;
  ta.value=ta.value.slice(0,a)+t+ta.value.slice(b);
  const c=a+t.length; ta.selectionStart=ta.selectionEnd=c;
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
async function decodeZh(syls, forced={}){
  const sets=syls.map(candSet), posIds=sets.map(s=>s.ids);
  for(const [i,ch] of Object.entries(forced)){
    const t=tid(ch); if(t!=null) posIds[i]=[t];   // pin this position
  }
  const prompt='<|im_start|>system\n注音轉繁體中文。<|im_end|>\n<|im_start|>user\n'+syls.join(' ')+'<|im_end|>\n<|im_start|>assistant\n';
  const enc=tokenizer(prompt);const start=enc.input_ids.dims[1];
  const out=await model.generate({...enc,max_new_tokens:posIds.length+1,do_sample:false,
    logits_processor:new LogitsProcessorList([new Masker(posIds,start)])});
  const dec=tokenizer.decode(out.tolist()[0].slice(start),{skip_special_tokens:true}).replace(/\s/g,'');
  return{chars:[...dec],cands:sets.map(s=>s.chars)};
}

// ---- keyboard UI ----
const kb=$('kb');
ROWS.forEach(row=>{const r=document.createElement('div');r.className='krow';
  row.forEach(key=>{const b=document.createElement('button');b.className='key';
    const sym=DACHEN[key]?DACHEN[key][0]:(key in TONEK?TONEK[key]:'');
    b.innerHTML='<span class="k">'+key+'</span><span class="s">'+sym+'</span>';
    b.onclick=()=>feedKey(key);r.appendChild(b);});kb.appendChild(r);});
const rowP=document.createElement('div');rowP.className='krow';
[['，','<'],['。','>'],['？','?'],['！','!'],['：',':']].forEach(([label,k])=>{
  const b=document.createElement('button');b.className='key';b.style.width='40px';
  b.innerHTML='<span class="s">'+label+'</span>';
  b.onclick=()=>feedKey(k);rowP.appendChild(b);});
const bl=document.createElement('button');bl.className='key';bl.style.width='40px';
bl.innerHTML='<span class="s">←</span>';bl.onclick=()=>moveCursor(-1);rowP.appendChild(bl);
const br=document.createElement('button');br.className='key';br.style.width='40px';
br.innerHTML='<span class="s">→</span>';br.onclick=()=>moveCursor(1);rowP.appendChild(br);
kb.appendChild(rowP);
const r=document.createElement('div');r.className='krow';
const sp=document.createElement('button');sp.className='key wide';sp.textContent='空白（一聲）';
sp.onclick=()=>{ if(hasRun()) feedKey(' '); };
const ent=document.createElement('button');ent.className='key wide2';ent.textContent='⏎ 上字';
ent.onclick=()=>commitSentence();
const bs=document.createElement('button');bs.className='key';bs.style.width='64px';
bs.innerHTML='<span class="s">⌫</span>';
bs.onclick=()=>{ if(committed.length||hasRun()) backspace(); };
r.appendChild(sp);r.appendChild(ent);r.appendChild(bs);kb.appendChild(r);

document.addEventListener('keydown',e=>{
  if(e.ctrlKey||e.altKey||e.metaKey)return;const k=e.key;
  // candidate-fix mode: numbers pick, arrows page, Esc closes
  if(fix>=0){
    if(k>='1'&&k<='9'){ pickCand(k.charCodeAt(0)-49); e.preventDefault(); return; }
    if(k==='ArrowDown'||k==='ArrowUp'){
      const pages=Math.ceil((pvCands[fix]||[]).length/PAGE), d=k==='ArrowDown'?1:-1;
      if(pages>1){fixPage=(fixPage+d+pages)%pages;render();} e.preventDefault(); return; }
    if(k==='Escape'){ fix=-1; render(); e.preventDefault(); return; }
    if(k==='Enter'){ fix=-1; render(); e.preventDefault(); return; }
  }
  if(k==='Enter'){ if(committed.length||hasRun()){commitSentence();e.preventDefault();} }
  else if(k===' '){ if(hasRun()){feedKey(' ');e.preventDefault();} else if(committed.length) e.preventDefault(); }
  else if(k==='Backspace'){ if(committed.length||hasRun()){backspace();e.preventDefault();} }
  else if(k==='ArrowLeft'){ if(committed.length||hasRun()){moveCursor(-1);e.preventDefault();} }
  else if(k==='ArrowRight'){ if(committed.length||hasRun()){moveCursor(1);e.preventDefault();} }
  else if(k==='ArrowDown'){ if(cursor>0&&committed[cursor-1]&&committed[cursor-1].t==='zh'){openFix(cursor-1);e.preventDefault();} }
  else if(k==='Escape'){ clearAll(); render(); }
  else if(k in PUNCT){ feedKey(k); e.preventDefault(); }
  else if(k.length===1&&(DACHEN[k]||k in TONEK||/[A-Za-z0-9]/.test(k))){ feedKey(k); e.preventDefault(); }
});
$('commit').onclick=()=>commitSentence();
$('clear').onclick=()=>{ clearAll(); render(); };
$('toneless').onchange=()=>{ pvKey=null; render(); };

(async function init(){
  const txt=await (await fetch('./phonetic_table.tsv')).text();
  for(const line of txt.split('\n')){const t=line.indexOf('\t');if(t<0)continue;tonal[line.slice(0,t)]=[...line.slice(t+1)];}
  for(const k in tonal) validBase.add(strip(k));   // for splitTrailingSyllable
  tokenizer=await AutoTokenizer.from_pretrained(REPO);
  model=await AutoModelForCausalLM.from_pretrained(REPO,{dtype:'q8'});
  ready=true; render();
})().catch(e=>{console.error(e);$('hint').textContent='模型載入失敗：'+(e&&e.name?e.name+' / ':'')+(e&&e.message||e);});
render();
