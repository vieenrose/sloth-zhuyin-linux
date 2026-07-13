// Slothing static web IME — chewing-style UX, fully in-browser.
//  * inline conversion: preedit shows CHINESE as you type (live decode);
//    bopomofo visible only for the syllable being composed
//  * Enter commits; Space = tone 1
//  * preedit cursor: ←/→ move, type/delete mid-sentence
//  * fix a char: click it or press ↓ at the cursor; paged candidates,
//    number keys 1-9 select (chewing-style)
//  * punctuation: Shift+, Shift+. Shift+/ Shift+1 Shift+; → ，。？！：
//  * session learning: your picks become the default for that syllable
//  * auto zh/en: continuous DP re-segmentation of the keystream (segment.js)
// SlothLM-E-T (25M ternary bidirectional encoder) runs via libslothe — a ggml
// forward pass compiled to WebAssembly (enc/slothe.{js,wasm} + the GGUF). One
// non-autoregressive pass, per-position argmax over each syllable's legal
// char-ids. No transformers.js, no autoregression, no onnxruntime.
import { makeSegmenter } from './segment.js?v=20260710zw';
import { makeAssoc } from './assoc.js?v=20260711a';

const ENC = './enc/';   // slothe.{js,wasm} + slothe-t-25m.gguf + syl_vocab.json + char2id.json
const TONES = 'ˊˇˋ˙';
const DACHEN = {'1':['ㄅ',0],'q':['ㄆ',0],'a':['ㄇ',0],'z':['ㄈ',0],'2':['ㄉ',0],'w':['ㄊ',0],
  's':['ㄋ',0],'x':['ㄌ',0],'e':['ㄍ',0],'d':['ㄎ',0],'c':['ㄏ',0],'r':['ㄐ',0],'f':['ㄑ',0],
  'v':['ㄒ',0],'5':['ㄓ',0],'t':['ㄔ',0],'g':['ㄕ',0],'b':['ㄖ',0],'y':['ㄗ',0],'h':['ㄘ',0],
  'n':['ㄙ',0],'u':['ㄧ',1],'j':['ㄨ',1],'m':['ㄩ',1],'8':['ㄚ',2],'i':['ㄛ',2],'k':['ㄜ',2],
  ',':['ㄝ',2],'9':['ㄞ',2],'o':['ㄟ',2],'l':['ㄠ',2],'.':['ㄡ',2],'0':['ㄢ',2],'p':['ㄣ',2],
  ';':['ㄤ',2],'/':['ㄥ',2],'-':['ㄦ',2]};
const TONEK = {'6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'};
const PUNCT = {'<':'，','>':'。','?':'？','!':'！',':':'：','"':'；','(':'（',')':'）','\\':'、'};
// Halfwidth ASCII form of each mark, for "en punctuation inside en text":
// a punctuation key in a pure-English clause stays halfwidth (Expected: 你好),
// \ stays literal (paths/regex). See docs/ZH-EN-MIXING.md.
const PUNCT_HALF = {'<':',','>':'.','?':'?','!':'!',':':':','"':';','(':'(',')':')','\\':'\\'};
// Does a punctuation key belong to a pure-English clause (-> halfwidth)? True
// iff the last committed token is English/ASCII and no Chinese precedes it in
// the current clause (scan back to the last sentence terminator). Keeps a comma
// in Chinese prose that abuts English fullwidth (我推薦 Python，因為…).
function punctInEnglishClause(){
  const n=committed.length, enish=t=>t.t!=='zh' && /^[\x00-\x7f]*$/.test(t.v);
  if(!n || !enish(committed[n-1])) return false;
  for(let i=n-1;i>=0;i--){
    if(committed[i].t==='zh') return false;
    if(/^[.!?。！？\n]$/.test(committed[i].v)) break;
  }
  return true;
}
// 微軟新注音 / 自然輸入法-style ` symbol menu (categories from libchewing symbols.dat)
const SYMBOLS = {
  '常用':'，、。．？！；：…—～‧「」『』（）',
  '括號':'（）「」『』〔〕【】《》〈〉｛｝︵︶﹁﹂',
  '數學':'＋－×÷＝≠≒±√＜＞≦≧∞∩∪∫∵∴',
  '單位':'℃℉°′″＄％＠＃＆＊§￥',
  '箭號':'↑↓←→↖↗↙↘',
  '圖形':'○●△▲☆★◇◆□■▽▼◎⊙※',
};
const SYMCATS = Object.keys(SYMBOLS);
let symbolMode = false, symCat = 0, fullWidth = false;
const FW = {}; for(let i=33;i<127;i++){ FW[String.fromCharCode(i)]=String.fromCharCode(i+0xFEE0); } FW[' ']='　';
const ROWS = [['1','2','3','4','5','6','7','8','9','0','-'],['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l',';'],['z','x','c','v','b','n','m',',','.','/']];
const PAGE = 9;
const strip = s => [...s].filter(c => !TONES.includes(c)).join('');
const $ = id => document.getElementById(id);

// ---- buffer state ----
// committed: [{t:'zh'|'en'|'punct', v}]; cursor = insertion index into it.
// rawKeys: the raw keys of the current composing run, segmented live by the
// continuous re-segmenter (segment.js) into zh/en tokens.
let committed = [], overrides = [], cursor = 0;
let rawKeys = '', enMode = false;
let pvChars = [], pvCands = [], pvMargins = [], pvKey = null;
let fix = -1, fixPage = 0;              // char being corrected
const validBase = new Set();           // legal syllable bases, filled from the table
const segment = makeSegmenter(DACHEN, TONEK, validBase);
const runToks = () => enMode ? (rawKeys ? [{t:'en',v:rawKeys}] : []) : segment(rawKeys);
const hasRun = () => rawKeys.length>0;
const bufKey = () => committed.map(t=>t.t+':'+t.v).join('|');

// ?fresh — start from a factory state (demos, testing): drop the persisted
// learn store + personal 聯想 bigrams before either module reads them
if (new URLSearchParams(location.search).has('fresh')) {
  try{ localStorage.removeItem('slothing-learn'); localStorage.removeItem('slothing-assoc'); }catch(e){}
}

// session learning (persisted): syllable -> last picked char
let learn = {};
try{ learn = JSON.parse(localStorage.getItem('slothing-learn')||'{}'); }catch(e){}
const saveLearn = ()=>{ try{localStorage.setItem('slothing-learn',JSON.stringify(learn));}catch(e){} };

// 聯想 (dictionary + personal bigrams; lock-step with engine/common/assoc.h)
const assoc = makeAssoc();
fetch('assoc_tc.tsv').then(r=>r.ok?r.text():'').then(t=>assoc.load(t)).catch(()=>{});
let predictChain = 0;

function resetRun(){ rawKeys=''; }
function insertTok(tok){ committed.splice(cursor,0,tok); overrides.splice(cursor,0,null); cursor++; }
function commitRun(){ for(const t of runToks()) insertTok(t); resetRun(); }
function clearAll(){ committed=[];overrides=[];cursor=0;resetRun();pvKey=null;fix=-1; }

// insert a literal symbol/punctuation directly (、 / and English symbols),
// bypassing zhuyin parsing — for the punctuation buttons.
function directPunct(ch){ fix=-1; if(hasRun())commitRun(); insertTok({t:'punct',v:ch}); render(); }
function feedKey(k){
  fix=-1;
  // Forced-English mode: literal chars, no zhuyin parsing (short words like
  // "is he", symbols, clean typo editing). Space ends the word.
  if(enMode){ // passthrough (微軟/chewing English mode): straight to output
    const ch=(fullWidth&&FW[k])?FW[k]:k;
    const ta=$('out');
    const a=(ta.selectionStart!=null)?ta.selectionStart:ta.value.length;
    const b=(ta.selectionEnd!=null)?ta.selectionEnd:a;
    ta.value=ta.value.slice(0,a)+ch+ta.value.slice(b);
    ta.selectionStart=ta.selectionEnd=a+ch.length;
    render(); return true;
  }
  if(k in PUNCT){ if(hasRun())commitRun();
    insertTok({t:'punct',v:punctInEnglishClause()?PUNCT_HALF[k]:PUNCT[k]}); render(); return true; }
  if(k===' '){ if(hasRun()){ commitRun();
    // preserve a user-typed space separating two English words (faithful; no
    // CJK-Latin autospace). A space after zhuyin was tone-1, not a literal.
    const last=committed[cursor-1];
    if(last&&last.t==='en'&&last.v!==' ') insertTok({t:'en',v:' '});
    render(); } return true; }
  if(k in TONEK){ rawKeys+=k; commitRun(); render(); return true; } // a tone completes the last syllable -> finalize run
  // any other key just extends the run; the segmenter re-decides zh/en live
  rawKeys+=k; render(); return true;
}
function backspace(){
  fix=-1;
  if(rawKeys){ rawKeys=rawKeys.slice(0,-1); }
  else if(cursor>0){ committed.splice(cursor-1,1); overrides.splice(cursor-1,1); cursor--; }
  render();
}
function moveCursor(d){
  if(hasRun()) return;               // chewing: arrows ignored while composing a syllable
  fix=-1;
  cursor=Math.max(0,Math.min(committed.length,cursor+d));
  render();
}
let preFixCursor=-1, fixHl=0;      // fixHl: highlighted candidate (absolute)
function openFix(i){
  if(i<0||i>=committed.length||committed[i].t!=='zh') return;
  if(pvKey!==bufKey()) return;       // wait for decode
  preFixCursor=cursor; cursor=i;     // chewing: cursor moves onto the char
  fix=i; fixPage=0; phrase=null; phraseBusy=false;
  fixHl=Math.max(0,(pvCands[i]||[]).indexOf(displayFor(i)));
  const sel=displayFor(i), cands=pvCands[i]||[];
  const at=cands.indexOf(sel); if(at>=0) fixPage=Math.floor(at/PAGE);
  render();
  buildPhrases(i);                   // async: 2-char phrase alternatives
}

// Phrase-level candidates from the encoder: one forward pass over the syllable
// pair gives per-position marginals; rank 2-char phrases by P(c0|pos0)·P(c1|pos1)
// and keep those clearly above the noise tail -> real words, no lexicon.
let phrase=null, phraseBusy=false, phraseFor=-1;
function softmaxOver(data,base,ids){let mx=-Infinity;for(const id of ids)if(data[base+id]>mx)mx=data[base+id];
  let z=0;const e={};for(const id of ids){e[id]=Math.exp(data[base+id]-mx);z+=e[id];}return{e,z};}
async function pairScores(a,b){   // 2-char phrases for zh tokens a,b -> [{ph,j}]
  const s0=committed[a].v, s1=committed[b].v;
  const c0=candSet(s0), c1=candSet(s1);
  if(!c0.ids.length||!c1.ids.length) return [];
  const {data,V}=await encForward([s0,s1]);
  const {e:e0,z:z0}=softmaxOver(data,0,c0.ids);
  const {e:e1,z:z1}=softmaxOver(data,V,c1.ids);
  const p0=c0.ids.map((id,k)=>({c:c0.chars[k],p:e0[id]/z0})).sort((x,y)=>y.p-x.p).slice(0,5);
  const p1=c1.ids.map((id,k)=>({c:c1.chars[k],p:e1[id]/z1})).sort((x,y)=>y.p-x.p).slice(0,5);
  const scored=[];
  for(const x of p0) for(const y of p1) scored.push({ph:x.c+y.c, j:x.p*y.p});
  scored.sort((x,y)=>y.j-x.j);
  const top=scored[0]?scored[0].j:0;
  return scored.filter(x=>x.j>=Math.max(0.06,0.15*top)).slice(0,6);
}
async function buildPhrases(i){
  // words COVERING the focused char (chewing/新注音): both the (i-1,i) and
  // (i,i+1) windows, merged by joint probability.
  phraseBusy=true; phraseFor=i; if(fix===i) render();
  try{
    const wins=[];
    if(i-1>=0&&committed[i-1].t==='zh') wins.push(i-1);
    if(i+1<committed.length&&committed[i+1].t==='zh') wins.push(i);
    const merged=[];
    for(const w of wins){
      for(const s of await pairScores(w,w+1)) merged.push({ph:s.ph, at:w, j:s.j});
      if(phraseFor!==i) return;
    }
    merged.sort((a,b)=>b.j-a.j);
    const seen=new Set(), kept=[];
    for(const m of merged){ const key=m.at+':'+m.ph; if(seen.has(key))continue; seen.add(key); kept.push(m); if(kept.length>=8)break; }
    if(phraseFor===i){ phrase=kept; if(fix===i) render(); }
  }catch(e){ console.error(e); }
  finally{ if(phraseFor===i) phraseBusy=false; }
}
function pickCand(j){                 // j is index within current page
  const cands=pvCands[fix]||[];
  const idx=fixPage*PAGE+j;
  if(idx>=cands.length) return;
  overrides[fix]=cands[idx];
  const syl=committed[fix].v;
  learn[syl]=cands[idx]; saveLearn();
  if(preFixCursor>=0){ cursor=preFixCursor; preFixCursor=-1; } // chewing: restore
  fix=-1; phrase=null;
  pvKey=null; schedulePreview();      // re-score the sentence around the pick
  render();
}
function pickPhrase(p){               // p = {ph, at}: 2-char phrase at [at, at+1]
  const at=p.at!=null?p.at:fix, chars=[...(p.ph||p)];
  overrides[at]=chars[0];
  let n=at+1; while(n<committed.length&&committed[n].t!=='zh') n++;
  if(n<committed.length&&chars[1]) overrides[n]=chars[1];
  learn[committed[at].v]=chars[0];
  if(n<committed.length&&chars[1]) learn[committed[n].v]=chars[1];
  saveLearn();
  pvKey=null; schedulePreview();      // re-score the sentence around the pick
  fix=-1; phrase=null; render();
}

let staleToks=[], staleChars=[];      // last successful conversion snapshot
function staleFor(i){
  // stable display while a decode is in flight: reuse the previous decoded
  // char for tokens unchanged at the same position from the start (prefix)
  // or from the end (suffix — covers mid-sentence edits). New/changed
  // tokens show bopomofo until their first decode, like chewing's cursor.
  const n=committed.length, m=staleToks.length;
  if(i<m && staleToks[i].t===committed[i].t && staleToks[i].v===committed[i].v){
    let ok=true;                       // prefix must match up to i
    for(let k=0;k<=i;k++){ if(k>=m||staleToks[k].t!==committed[k].t||staleToks[k].v!==committed[k].v){ok=false;break;} }
    if(ok) return staleChars[i];
  }
  const j=m-(n-i);                     // suffix alignment
  if(j>=0 && j<m && staleToks[j].t===committed[i].t && staleToks[j].v===committed[i].v){
    let ok=true;
    for(let k=i;k<n;k++){ const q=m-(n-k); if(q<0||staleToks[q].t!==committed[k].t||staleToks[q].v!==committed[k].v){ok=false;break;} }
    if(ok) return staleChars[j];
  }
  return null;
}
function displayFor(i){
  const tok=committed[i];
  if(tok.t!=='zh') return tok.v;
  if(overrides[i]) return overrides[i];
  if(pvKey===bufKey() && pvChars[i]!=null) return pvChars[i];
  const st=staleFor(i); if(st!=null) return st;
  return tok.v;                        // bopomofo until the first decode
}
function sentenceText(){
  // faithful: no injected CJK-Latin spaces (zh-TW convention; the inter-script
  // gap is typographic, supplied by the renderer). See docs/ZH-EN-MIXING.md.
  let out='';
  committed.forEach((tok,i)=>{ out+=displayFor(i); });
  return out;
}

// ---- rendering ----
// ` symbol menu: categories in #phrases, symbols in #cands (reuse candidate UI)
function renderSymbols(){
  const phEl=$('phrases'); phEl.innerHTML='';
  SYMCATS.forEach((cat,i)=>{ const b=document.createElement('button');
    b.className='cand'+(i===symCat?' ph':''); b.textContent=cat;
    b.onclick=()=>{ symCat=i; render(); }; phEl.appendChild(b); });
  const el=$('cands'); el.innerHTML='';
  [...SYMBOLS[SYMCATS[symCat]]].forEach((s,j)=>{ const b=document.createElement('button');
    b.className='cand'; b.innerHTML=(j<9?'<span class="n">'+(j+1)+'</span>':'')+s;
    b.onclick=()=>insSym(s); el.appendChild(b); });
  $('hint').textContent='符號選單　1-9 選　←→ 換類別　Esc 關閉';
}
function insSym(s){ symbolMode=false; directPunct(s); }
function toggleSymbols(){ symbolMode=!symbolMode; if(symbolMode){ if(hasRun())commitRun(); fix=-1; } render(); }
function toggleEnMode(){ if(hasRun())commitRun(); enMode=!enMode; symbolMode=false; paintEn&&paintEn(); typeof paintKeys==='function'&&paintKeys(); render(); }
function render(){
  const pre=$('pre'); pre.innerHTML='';
  const caret=()=>{const c=document.createElement('span');c.className='caret';return c;};
  committed.forEach((tok,i)=>{
    if(i===cursor && !hasRun()) pre.appendChild(caret());
    const span=document.createElement('span');
    span.className='pchar'+(fix===i?' fixsel':'')+(tok.t!=='zh'?' en':'');
    span.textContent=displayFor(i);   // faithful: no injected CJK-Latin spaces
    if(tok.t==='zh') span.onclick=()=>{ fix===i?(fix=-1,render()):openFix(i); };
    pre.appendChild(span);
    if(i===committed.length-1 && cursor===committed.length && !hasRun()) pre.appendChild(caret());
  });
  if(!committed.length && !hasRun()) pre.appendChild(caret());
  if(hasRun()){
    // the run renders at the cursor position, live-segmented into zh/en tokens
    const nodes=[...pre.childNodes];
    const frag=document.createDocumentFragment();
    for(const t of runToks()){
      const s=document.createElement('span'); s.className='ptail'+(t.t==='en'?' en':'');
      s.textContent=t.v; frag.appendChild(s);   // faithful: no injected CJK-Latin spaces
    }
    let anchor=null, count=0;
    for(const n of nodes){ if(n.classList&&n.classList.contains('pchar')){count++; if(count===cursor){anchor=n;break;}} }
    if(cursor===0) pre.insertBefore(frag,pre.firstChild);
    else if(anchor&&anchor.nextSibling) pre.insertBefore(frag,anchor.nextSibling);
    else pre.appendChild(frag);
  }
  pre.classList.toggle('empty',!committed.length&&!hasRun());

  if(symbolMode){ renderSymbols(); return; }

  // phrase candidates (2-char), when available
  const phEl=$('phrases'); phEl.innerHTML='';
  if(fix>=0 && phrase && phrase.length){
    const lbl=document.createElement('span'); lbl.className='pg'; lbl.textContent='詞';
    phEl.appendChild(lbl);
    const seen=new Set();
    phrase.forEach((p,pi)=>{ const key=p.at+':'+p.ph; if(seen.has(key))return; seen.add(key);
      const b=document.createElement('button'); b.className='cand ph'+((fixHl<0&&(-1-fixHl)===pi)?' hlp':'');
      b.innerHTML='<span class="n">⇧'+(pi+1)+'</span>'+p.ph;
      b.onclick=()=>pickPhrase(p); phEl.appendChild(b); });
  } else if(fix>=0 && phraseBusy){
    const s=document.createElement('span'); s.className='pg'; s.textContent='組詞中…'; phEl.appendChild(s);
  } else if(fix<0 && !committed.length && !hasRun() && assoc.hasTail()){
    // 聯想 (mobile/微軟 convention): after a commit the row shows next-word
    // predictions; tap or ⇧1-9 inserts + chains; typing dismisses (buffer
    // non-empty stops this branch rendering).
    const preds=assoc.predictions();
    if(preds.length){
      const lbl=document.createElement('span'); lbl.className='pg pg-assoc'; lbl.textContent='聯';
      phEl.appendChild(lbl);
      preds.forEach((w,i)=>{
        const b=document.createElement('button'); b.className='cand ph';
        b.innerHTML='<span class="n">⇧'+(i+1)+'</span>'+w;
        b.onclick=()=>pickPredict(w); phEl.appendChild(b);
      });
    }
  }

  // paged candidate strip
  const stripEl=$('cands'); stripEl.innerHTML='';
  // touch-mode composing strip (mobile convention, mirrors Android): 字
  // candidates for the LAST word + 句 sentence alternates, auto-shown
  if(touchDev && fix<0 && committed.length && pvKey===bufKey()){
    let li=-1; for(let i=committed.length-1;i>=0;i--) if(committed[i].t==='zh'){ li=i; break; }
    if(li>=0 && (pvCands[li]||[]).length>1){
      const lbl=document.createElement('span'); lbl.className='pg'; lbl.textContent='字';
      stripEl.appendChild(lbl);
      const cur=displayFor(li);
      pvCands[li].slice(0,6).forEach(c=>{
        const b=document.createElement('button'); b.className='cand'+(c===cur?' sel':'');
        b.textContent=c;
        b.onclick=()=>{ overrides[li]=c;
          const syl=committed[li].v;
          learn[syl]=c; saveLearn();
          pvKey=null; schedulePreview(); render(); };
        stripEl.appendChild(b);
      });
    }
    // Whole-sentence 句 alternates removed: single-flip n-best only differs by
    // one char and repeats the confident prefix. Correction is per-position —
    // 字 above (last word) and click-a-char (點字改字) / ↓ for earlier ones,
    // always local to the cursor. Matches Android + desktop.
  }
  if(fix>=0 && pvKey===bufKey() && pvCands[fix]){
    const cands=pvCands[fix], pages=Math.ceil(cands.length/PAGE);
    if(pages>1){ const bp=document.createElement('button'); bp.className='cand nav'; bp.textContent='‹';
      bp.onclick=()=>{fixPage=(fixPage-1+pages)%pages;render();}; stripEl.appendChild(bp); }
    cands.slice(fixPage*PAGE,fixPage*PAGE+PAGE).forEach((c,j)=>{
      const b=document.createElement('button');
      b.className='cand'+((fixPage*PAGE+j)===fixHl?' sel':'');
      b.innerHTML='<span class="n">'+(j+1)+'</span>'+c;
      b.onclick=()=>pickCand(j);
      stripEl.appendChild(b);
    });
    if(pages>1){ const bn=document.createElement('button'); bn.className='cand nav'; bn.textContent='›';
      bn.onclick=()=>{fixPage=(fixPage+1)%pages;render();}; stripEl.appendChild(bn);
      const pg=document.createElement('span'); pg.className='pg'; pg.textContent=(fixPage+1)+'/'+pages;
      stripEl.appendChild(pg); }
  }
  if(ready) $('hint').textContent = fix>=0 ? '1-9 選字　←→ 移動　⏎ 確認　↑↓ 翻頁　Esc 取消'
    : (committed.length||hasRun() ? '⏎ 上字　←→ 游標　↓ 選字　點字改字' : '直接打注音或英文，自動辨識；Shift+，。？！ 輸入標點');
  schedulePreview();
}

// ---- live conversion ----
let previewTimer=null, previewBusy=false, previewGen=0;
function schedulePreview(){
  if(!ready) return;
  if(!committed.some(t=>t.t==='zh')){ pvKey=bufKey(); pvChars=committed.map(t=>t.v); pvCands=committed.map(t=>[t.v]); return; }
  if(pvKey===bufKey()) return;
  // do NOT reset a pending timer on every keystroke — that starves the
  // decode during continuous typing (everything stayed bopomofo until a
  // pause). One short timer; the completion handler reschedules if stale.
  if(previewTimer||previewBusy) return;
  previewTimer=setTimeout(()=>{ previewTimer=null; runPreview(); },60);
}
async function runPreview(){
  if(previewBusy) return;
  previewBusy=true;
  const key=bufKey(), gen=++previewGen;
  const toksSnap=committed.map(t=>({t:t.t,v:t.v}));  // snapshot for stale seed
  try{
    // Decode each punctuation-delimited segment independently: punctuation
    // (，。、！？…) marks a semantic boundary, so the model needn't see across
    // it. Shorter context => faster (decode time ~ segment length, not whole
    // buffer) and no long-sequence truncation.
    const zhChars=[], zhCands=[], zhMargins=[]; let seg=[], segIdx=[]; let ci=0;
    let ctxAcc=$('out').value.slice(-CTX_MAX);   // document context (committed)
    const flushSeg=async()=>{
      const zh=seg.filter(t=>t.t==='zh').map(t=>t.v);
      // user picks (overrides) condition the whole segment's re-decode
      const hints={}; let rel=0;
      for(let q=0;q<seg.length;q++) if(seg[q].t==='zh'){
        const oi=segIdx[q]; if(overrides[oi]) hints[rel]=overrides[oi]; rel++; }
      const r=zh.length?await decodeZh(zh,{},hints,ctxAcc):{chars:[],cands:[],margins:[]};
      if(r.chars.length) ctxAcc=(ctxAcc+r.chars.join('')).slice(-CTX_MAX);
      let zc=0;
      for(const tok of seg) if(tok.t==='zh'){ zhCands.push(r.cands[zc]); zhChars.push(r.chars[zc]); zhMargins.push((r.margins||[])[zc] ?? Infinity); zc++; }
      seg=[];
    };
    for(const tok of committed){ seg.push(tok); segIdx.push(ci++); if(tok.t==='punct'){ await flushSeg(); segIdx=[]; } }
    await flushSeg();
    if(gen===previewGen){
      const chars=[], cands=[], margins=[]; let zc=0;
      for(const tok of toksSnap){
        if(tok.t==='zh'){
          const cs=zhCands[zc];
          let ch=zhChars[zc]; if(ch==null||!cs.includes(ch)) ch=cs[0];  // never fall back to bopomofo
          chars.push(ch); cands.push(cs); margins.push(zhMargins[zc] ?? Infinity); zc++;
        } else { chars.push(tok.v); cands.push([tok.v]); margins.push(Infinity); }
      }
      // even a STALE decode (user typed meanwhile) seeds the snapshot so the
      // unchanged prefix/suffix paints converted mid-burst (chewing-like)
      staleToks=toksSnap; staleChars=chars;
      if(key===bufKey()){ pvChars=chars; pvCands=cands; pvMargins=margins; pvKey=key; }
      render();
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
  assoc.record(t); predictChain=0;   // 聯想: strip flips to predictions
  clearAll(); render();
}

// tap/⇧n on a 聯想 chip: insert at the output caret, learn, chain (cap 5)
function pickPredict(w){
  const ta=$('out');
  const a=(ta.selectionStart!=null)?ta.selectionStart:ta.value.length;
  const b=(ta.selectionEnd!=null)?ta.selectionEnd:a;
  ta.value=ta.value.slice(0,a)+w+ta.value.slice(b);
  ta.selectionStart=ta.selectionEnd=a+w.length;
  assoc.record(w);
  if(++predictChain>=5) assoc.clearTail();
  render();
}

// ---- model + decode (SlothLM-E-T ternary encoder, libslothe ggml/WASM) ----
let sylVocab, char2id, ready=false;
let slotheMod=null, slotheNChar=0;
const tonal={};
// Damerau-Levenshtein distance capped at 1 (typo tolerance: one wrong /
// missing / extra / swapped bopomofo symbol).
function dl1(a,b){ if(a===b)return 0; const la=a.length,lb=b.length;
  if(Math.abs(la-lb)>1)return 2;
  if(la===lb){ let d=0,sw=-1;
    for(let i=0;i<la;i++)if(a[i]!==b[i]){ if(d===0)sw=i; d++; }
    if(d===1)return 1;
    if(d===2&&sw>=0&&a[sw]===b[sw+1]&&a[sw+1]===b[sw])return 1;   // transpose
    return 2; }
  const s=la<lb?a:b, l=la<lb?b:a;                                  // 1 ins/del
  for(let i=0;i<=s.length;i++){ if(s.slice(0,i)+l[i]+s.slice(i)===l)return 1; }
  return 2; }
function candChars(syl){const chars=tonal[syl]||[];  // tone-optional removed: unmarked = tone-1 (its bare row), no all-tone union
  return chars.length?chars:[syl];}
// Typo tolerance: for an impossible syllable, candidate corrections within
// edit distance 1 that keep the typed tone: [{syl, chars}].
function typoFixes(syl){const tone=[...syl].filter(c=>TONES.includes(c)).join('');
  const base=strip(syl), out=[];
  for(const k in tonal){ const kb=strip(k), kt=[...k].filter(c=>TONES.includes(c)).join('');
    if(kt===tone&&dl1(kb,base)<=1) out.push({syl:k,chars:tonal[k]}); }
  return out;}
// candidate chars + their output token-ids (from char2id) for a syllable
function candSet(syl){const out=[],ids=[];for(const c of candChars(syl)){const i=char2id[c];if(i!=null&&!ids.includes(i)){ids.push(i);out.push(c);}}return{chars:out,ids};}
// one bidirectional forward pass over the syllable sequence -> per-position logits
let modelHasHints=false;   // ternary encoder has no hint/context channel
const CTX_MAX=12;
// ---- libslothe (WASM) forward path ----
// The ternary ggml encoder is syl-only: no context/hint channel, so L=0 and
// T = syllable count. Output logits columns are char2id indices (V = n_char),
// identical alignment to the ONNX head — decodeZh/candSet are unchanged.
function slotheLogits(ids){
  const T=ids.length, V=slotheNChar;
  const sp=slotheMod._malloc(T*4);
  slotheMod.HEAP32.set(Int32Array.from(ids.map(x=>Number(x))), sp>>2);
  const op=slotheMod._malloc(T*V*4);
  slotheMod.ccall('slothe_wasm_logits', null, ['number','number','number'], [sp, T, op]);
  const out=slotheMod.HEAPF32.slice(op>>2, (op>>2)+T*V);
  slotheMod._free(sp); slotheMod._free(op);
  return out;   // Float32Array [T*V]
}
function slotheForward(syls){
  const ids=syls.map(s=>sylVocab[s]??1);
  return {data:slotheLogits(ids), V:slotheNChar, T:ids.length, L:0};
}
function slotheForwardBatch(rows){
  const B=rows.length, T=rows[0].length, V=slotheNChar;
  const data=new Float32Array(B*T*V);
  for(let b=0;b<B;b++) data.set(slotheLogits(rows[b]), b*T*V);
  return {data, V, T};
}
async function encForward(syls){ return slotheForward(syls); }
// batched forward for typo scoring: rows = variants of the same sentence
async function encForwardBatch(rows){ return slotheForwardBatch(rows); }
const LEARN_BONUS=2.0;   // recalibrated 2026-07-11: 6.0 over-personalized (59% vs 74% on the 免選字 set with a used store); 2.0 keeps near-tie flips (ㄧㄣ→音) without overriding strong context
async function decodeZh(sylsIn, forced={}, hints={}, ctx=''){
  // typo tolerance: an impossible syllable (no legal chars) is replaced by
  // the edit-distance-1 correction the model itself scores highest. Legal
  // syllables never enter this path.
  const syls=[...sylsIn];
  for(let i=0;i<syls.length;i++){
    if(candSet(syls[i]).ids.length) continue;
    const fixes=typoFixes(syls[i]).filter(f=>sylVocab[f.syl]!=null&&f.chars.some(c=>char2id[c]!=null));
    if(!fixes.length) continue;
    const baseIds=syls.map(s=>sylVocab[s]??1);
    const rows=fixes.map(f=>{const r=[...baseIds];r[i]=sylVocab[f.syl];return r;});
    const {data:bd,V:bV}=await encForwardBatch(rows);
    let bj=0,bv=-Infinity;
    const typedLen=strip(syls[i]).length;
    fixes.forEach((f,j)=>{ let best=-Infinity;
      for(const c of f.chars){ const id=char2id[c];
        if(id!=null){ const v=bd[(j*syls.length+i)*bV+id]; if(v>best)best=v; } }
      // prior: a MISSED key (longer corrected base) is the common slip
      best += 1.5*(strip(f.syl).length - typedLen);
      if(best>bv){bv=best;bj=j;} });
    syls[i]=fixes[bj].syl;
  }
  const sets=syls.map(candSet);
  let hintIds=null;
  if(modelHasHints && Object.keys(hints).length){
    hintIds=new Array(syls.length).fill(0);
    for(const k in hints){ const id=char2id[hints[k]]; if(id!=null) hintIds[+k]=id+1; }
  }
  const {data,V,L}=await encForward(syls, hintIds, ctx);
  const chars=[], ranked=[], margins=[];
  for(let i0=0;i0<syls.length;i0++){
    const {chars:cc,ids}=sets[i0];
    // model-score candidate order (chewing/新注音 rank by score, not table)
    const scored=cc.map((c,k)=>({c,v:data[(L+i0)*V+ids[k]] + (learn[syls[i0]]===c?LEARN_BONUS:0)}))
                   .sort((a,b)=>b.v-a.v);
    ranked.push(scored.map(x=>x.c));
    margins.push(scored.length>1?scored[0].v-scored[1].v:Infinity);
    if(forced[i0]!=null){ chars.push(forced[i0]); continue; }
    chars.push(scored.length?scored[0].c:syls[i0]);
  }
  return{chars,cands:ranked,margins};
}

// ---- keyboard UI ----
// On phones, render the iOS zhuyin arrangement: the SAME Dàqiān columns
// (Apple uses this exact grid), bopomofo-only caps, ⌫/、/⏎ as a right-hand
// function column — instantly familiar to iPhone 注音 users.
const kb=$('kb');
const keyBtns=[];
const touchDev = matchMedia('(pointer:coarse)').matches || navigator.maxTouchPoints > 0 ||
  ('ontouchstart' in window) || new URLSearchParams(location.search).has('kb');
const iosKb = matchMedia('(pointer:coarse) and (max-width:600px)').matches
              || new URLSearchParams(location.search).get('kb')==='ios';
if(iosKb) document.body.classList.add('ioskb');
if(touchDev) document.body.classList.add('touchdev');
ROWS.forEach((row,ri)=>{const r=document.createElement('div');r.className='krow';
  row.forEach(key=>{const b=document.createElement('button');b.className='key';
    const sym=DACHEN[key]?DACHEN[key][0]:(key in TONEK?TONEK[key]:'');
    b.onclick=()=>feedKey(key);r.appendChild(b);keyBtns.push({b,key,sym});});
  if(iosKb){ // right-hand function column, iOS-style
    const fns=[
      {t:'⌫', f:()=>{ if(committed.length||hasRun()) backspace(); }},
      {t:'、', f:()=>directPunct('、')},
      {t:'？', f:()=>directPunct('？')},
      {t:'⏎', f:()=>commitSentence()},
    ][ri];
    if(fns){ const b=document.createElement('button'); b.className='key fn';
      b.innerHTML='<span class="s">'+fns.t+'</span>'; b.onclick=fns.f; r.appendChild(b); }
  }
  kb.appendChild(r);});
// In English mode show the QWERTY letter big (zhuyin small); Chinese mode: zhuyin big.
function paintKeys(){ for(const {b,key,sym} of keyBtns){
  b.innerHTML=enMode ? '<span class="s">'+key.toUpperCase()+'</span>'
                     : '<span class="k">'+key+'</span><span class="s">'+sym+'</span>'; } }
paintKeys();
const rowP=document.createElement('div');rowP.className='krow';
// direct-insert punctuation incl. 、(頓號) and / (slash), which have no free
// zhuyin key; the mapped ones route through feedKey/PUNCT.
[['，','<'],['。','>'],['、',''],['？','?'],['！','!'],['：',':'],['/','']].forEach(([label,k])=>{
  const b=document.createElement('button');b.className='key';b.style.width='36px';
  b.innerHTML='<span class="s">'+label+'</span>';
  b.onclick=()=>{ (k in PUNCT)?feedKey(k):directPunct(label); };rowP.appendChild(b);});
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
const bs=document.createElement('button');bs.className='key';bs.style.width='56px';
bs.innerHTML='<span class="s">⌫</span>';
bs.onclick=()=>{ if(committed.length||hasRun()) backspace(); };
// English-mode toggle: escape hatch for what auto-detection can't infer
// (short English words that look like zhuyin, symbols, clean typo editing).
const en=document.createElement('button');en.className='key';en.style.width='52px';
const paintEn=()=>{ en.innerHTML='<span class="s">'+(enMode?'英':'中')+'</span>'; en.style.background=enMode?'var(--brown)':''; en.style.color=enMode?'var(--hi)':''; };
en.onclick=()=>toggleEnMode();
paintEn();
// ` symbol menu button (微軟/自然-style)
const symb=document.createElement('button');symb.className='key';symb.style.width='52px';
symb.innerHTML='<span class="s">符</span>';symb.onclick=()=>toggleSymbols();
r.appendChild(sp);r.appendChild(ent);r.appendChild(bs);r.appendChild(en);r.appendChild(symb);kb.appendChild(r);

// Physical-keyboard feedback: flash the matching virtual key on keydown.
const keyByChar={}; keyBtns.forEach(({b,key})=>keyByChar[key]=b);
function flashKey(btn){ if(!btn)return; btn.classList.add('pressed');
  clearTimeout(btn._pt); btn._pt=setTimeout(()=>btn.classList.remove('pressed'),130); }
let shiftAlone=false;
document.addEventListener('keydown',e=>{
  if(e.key==='Shift'){ shiftAlone=true; return; }        // track lone-Shift (微軟 English toggle)
  if(e.key!=='Shift') shiftAlone=false;
  if(e.ctrlKey||e.altKey||e.metaKey)return;const k=e.key;
  flashKey(k===' '?sp : k==='Enter'?ent : k==='Backspace'?bs
           : k==='ArrowLeft'?bl : k==='ArrowRight'?br : keyByChar[k.toLowerCase()]);
  // Shift+Space toggles 全形/半形 (微軟/自然 convention)
  if(k===' '&&e.shiftKey){ fullWidth=!fullWidth; if(ready)$('hint').textContent=fullWidth?'全形':'半形'; e.preventDefault(); return; }
  // ` symbol menu: number keys pick, ←→ switch category, Esc/` close
  if(symbolMode){
    if(k>='1'&&k<='9'){ const arr=[...SYMBOLS[SYMCATS[symCat]]]; if(arr[k.charCodeAt(0)-49])insSym(arr[k.charCodeAt(0)-49]); e.preventDefault(); return; }
    if(k==='ArrowLeft'){ symCat=(symCat-1+SYMCATS.length)%SYMCATS.length; render(); e.preventDefault(); return; }
    if(k==='ArrowRight'){ symCat=(symCat+1)%SYMCATS.length; render(); e.preventDefault(); return; }
    if(k==='Escape'||k==='`'){ symbolMode=false; render(); e.preventDefault(); return; }
    return;
  }
  // candidate-fix mode: numbers pick, arrows page, Esc closes
  if(fix>=0){
    // chewing: the candidate window is MODAL — selection keys act, ←→ move
    // the disambiguation target (window follows), everything else is ignored.
    if(e.shiftKey && e.code && e.code.startsWith('Digit')){   // ⇧1-9 = 詞
      const idx=e.code.charCodeAt(5)-49;
      if(phrase && phrase[idx]){ pickPhrase(phrase[idx]); e.preventDefault(); return; }
    }
    if(k>='1'&&k<='9'){ pickCand(k.charCodeAt(0)-49); e.preventDefault(); return; }
    if(k==='ArrowDown'||k==='ArrowUp'){
      const pages=Math.ceil((pvCands[fix]||[]).length/PAGE), d=k==='ArrowDown'?1:-1;
      if(pages>1){fixPage=(fixPage+d+pages)%pages; fixHl=fixPage*PAGE; render();}
      e.preventDefault(); return; }
    if(k==='ArrowLeft'||k==='ArrowRight'){ // ←→ walk 詞 chips + chars as one loop
      const cands=pvCands[fix]||[], nph=(phrase||[]).length, d=k==='ArrowLeft'?-1:1;
      const total=nph+cands.length;
      if(total){
        // map to combined index: phrases occupy 0..nph-1, chars follow
        let ci=(fixHl<0? -1-fixHl : nph+fixHl);
        ci=(ci+d+total)%total;
        if(ci<nph){ fixHl=-1-ci; } else { fixHl=ci-nph; fixPage=Math.floor(fixHl/PAGE); }
        render(); }
      e.preventDefault(); return; }
    if(k===' '){                            // chewing: space pages
      const pages=Math.ceil((pvCands[fix]||[]).length/PAGE);
      if(pages>1){fixPage=(fixPage+1)%pages; fixHl=fixPage*PAGE; render();}
      e.preventDefault(); return; }
    if(k==='Enter'){                        // Enter confirms the highlight
      if(fixHl<0){ const ph=(phrase||[])[-1-fixHl]; if(ph) pickPhrase(ph); }
      else pickCand(fixHl-fixPage*PAGE);
      e.preventDefault(); return; }
    if(k==='j'||k==='k'){                     // chewing: j/k move the target
      const d=k==='j'?-1:1; let t=fix+d;
      while(t>=0&&t<committed.length&&committed[t].t!=='zh') t+=d;
      if(t>=0&&t<committed.length){ fix=t; cursor=t; fixPage=0; phrase=null; render(); buildPhrases(t); }
      e.preventDefault(); return; }
    if(k==='Escape'){
      if(preFixCursor>=0){ cursor=preFixCursor; preFixCursor=-1; }
      fix=-1; render(); e.preventDefault(); return; }
    e.preventDefault(); return;   // Enter, zhuyin keys, etc: ignored (modal)
  }
  if(fix<0 && !committed.length && !hasRun() && e.shiftKey && e.code && e.code.startsWith('Digit')){
    const preds=assoc.predictions(); const idx=+e.code.slice(5)-1;
    if(idx>=0 && preds[idx]){ pickPredict(preds[idx]); e.preventDefault(); return; }
  }
  if(k==='Enter'){ if(committed.length||hasRun()){commitSentence();e.preventDefault();} }
  else if(k===' '){ if(enMode){feedKey(' ');e.preventDefault();} else if(hasRun()){feedKey(' ');e.preventDefault();} else if(committed.length) e.preventDefault(); }
  else if(k==='Backspace'){ if(committed.length||hasRun()){backspace();e.preventDefault();} }
  else if(k==='Home'){ if(committed.length||hasRun()){ if(!hasRun()){cursor=0; fix=-1; render();} e.preventDefault(); } }
  else if(k==='End'){ if(committed.length||hasRun()){ if(!hasRun()){cursor=committed.length; fix=-1; render();} e.preventDefault(); } }
  else if(k==='ArrowLeft'){ if(committed.length||hasRun()){moveCursor(-1);e.preventDefault();} }
  else if(k==='ArrowRight'){ if(committed.length||hasRun()){moveCursor(1);e.preventDefault();} }
  else if(k==='ArrowDown'){ // chewing: char AT the cursor; at end, the last char
    const t=(cursor<committed.length)?cursor:cursor-1;
    if(t>=0&&committed[t]&&committed[t].t==='zh'){openFix(t);e.preventDefault();} }
  else if(k==='Escape'){ if(hasRun()){ resetRun(); } render(); }
  else if(k==='`'){ toggleSymbols(); e.preventDefault(); }               // ` opens symbol menu (微軟/自然)
  else if(enMode && k.length===1){ feedKey(k); e.preventDefault(); }  // English mode: literal (feedKey applies width)
  else if(k in PUNCT){ feedKey(k); e.preventDefault(); }
  else if(k.length===1&&(DACHEN[k]||k in TONEK||/[A-Za-z0-9']/.test(k))){ feedKey(k); e.preventDefault(); }
});
document.addEventListener('keyup',e=>{ if(e.key==='Shift'&&shiftAlone){ shiftAlone=false; toggleEnMode(); } });
$('commit').onclick=()=>commitSentence();
$('clear').onclick=()=>{ clearAll(); render(); };

// test hook for the differential UI-parity suite (eval/ui-parity): the
// observable UI state, same schema as chewing_trace.c (structure only).
window.__ui = () => ({
  zh: committed.length,
  bopo: hasRun() ? 1 : 0,
  cand: (fix >= 0 || symbolMode) ? 1 : 0,
  cursor: cursor,
  out: $('out').value.length,
  settled: !hasRun() || true,
  fresh: pvKey === bufKey() || !committed.some(t => t.t === 'zh'),
});

// Load the encoder: libslothe (ggml/WASM) is the only backend. Any failure
// (missing gguf/module, load error) surfaces as a load-error message.
async function initEncoder(){
  try{
    const resp=await fetch(ENC+'slothe-t-25m.gguf');
    if(!resp.ok) throw new Error('gguf http '+resp.status);
    // Multi-threaded build needs SharedArrayBuffer (crossOriginIsolated, set by
    // the coi-serviceworker's COOP/COEP). Fall back to single-thread otherwise.
    const mt = (typeof crossOriginIsolated!=='undefined') && crossOriginIsolated;
    const {default:createSlotheModule}=await import(ENC+(mt?'slothe-mt.js':'slothe.js'));
    slotheMod=await createSlotheModule();
    console.log('encoder: '+(mt?'multi-thread':'single-thread')+' WASM');
    const bytes=new Uint8Array(await resp.arrayBuffer());
    const p=slotheMod._malloc(bytes.length);
    slotheMod.HEAPU8.set(bytes, p);
    const ok=slotheMod.ccall('slothe_wasm_load','number',['number','number'],[p, bytes.length]);
    slotheMod._free(p);
    if(!ok) throw new Error('gguf load failed');
    slotheNChar=slotheMod.ccall('slothe_wasm_n_char','number',[],[]);
    modelHasHints=false;   // ternary encoder has no hint/context channel
    console.log('encoder: libslothe (ggml/WASM), n_char='+slotheNChar);
    return 'slothe';
  }catch(e){
    console.error('libslothe failed to load:', (e&&e.message)||e);
    throw e;
  }
}

(async function init(){
  const txt=await (await fetch('./phonetic_table.tsv')).text();
  for(const line of txt.split('\n')){const t=line.indexOf('\t');if(t<0)continue;tonal[line.slice(0,t)]=[...line.slice(t+1)];}
  for(const k in tonal) validBase.add(strip(k));   // for the segmenter
  [sylVocab, char2id] = await Promise.all([
    fetch(ENC+'syl_vocab.json').then(r=>r.json()),
    fetch(ENC+'char2id.json').then(r=>r.json()),
  ]);
  await initEncoder();
  ready=true; render();
})().catch(e=>{console.error(e);$('hint').textContent='模型載入失敗：'+(e&&e.name?e.name+' / ':'')+(e&&e.message||e);});
render();
