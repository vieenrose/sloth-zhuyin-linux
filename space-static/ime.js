// Slothing static web IME — runs entirely in the browser (Transformers.js).
// Auto-detects Chinese vs English with NO mode toggle: valid zhuyin adds one
// bopomofo symbol per keystroke, so the moment the raw keystrokes exceed the
// assembled symbols (an impossible zhuyin structure), the run is English.
// The model decodes the zhuyin runs (per-position legal-char mask); English
// tokens pass through verbatim.
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
const TONEK = {'6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'};   // tone keys (space handled apart)
const ROWS = [['1','2','3','4','5','6','7','8','9','0','-'],['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l',';'],['z','x','c','v','b','n','m',',','.','/']];
const strip = s => [...s].filter(c => !TONES.includes(c)).join('');
const $ = id => document.getElementById(id);
const isEn = tok => /[A-Za-z]/.test(tok.v);

// committed: array of {t:'zh'|'en', v}. cur/rawWord = the run being typed.
let committed = [], cur = ['','',''], rawWord = '', enRun = false;
let positions = [], segSel = [], segFocus = 0, choosing = false;
const numSym = () => (cur[0]?1:0)+(cur[1]?1:0)+(cur[2]?1:0);
const hasPending = () => cur[0]||cur[1]||cur[2];
const pending = () => cur[0]+cur[1]+cur[2];
function resetRun(){ cur=['','','']; rawWord=''; enRun=false; }
function commitRun(){
  if(enRun && rawWord) committed.push({t:'en',v:rawWord});
  else if(hasPending()) committed.push({t:'zh',v:pending()});   // tone-1
  else if(rawWord) committed.push({t:'en',v:rawWord});
  resetRun();
}
function runText(){ return enRun ? rawWord : pending(); }

function preText(){
  let p='';
  for(const tok of committed){
    if(tok.t==='en') p+=(p&&!p.endsWith(' ')?' ':'')+tok.v+' ';
    else p+=tok.v;
  }
  const r=runText();
  if(r) p+=(enRun&&p&&!p.endsWith(' ')?' ':'')+r;
  return p.trim();
}
function feedKey(k){
  if(choosing) return false;
  if(k===' '){ if(committed.length||hasPending()||rawWord){commitRun();renderComposing();} return true; }
  if(k in TONEK){                    // definite zhuyin tone
    if(!enRun && hasPending()){committed.push({t:'zh',v:pending()+TONEK[k]});resetRun();renderComposing();return true;}
    rawWord+=k; enRun=true; renderComposing(); return true;   // digit inside English word
  }
  if(enRun){ rawWord+=k; renderComposing(); return true; }
  if(DACHEN[k]){ cur[DACHEN[k][1]]=DACHEN[k][0]; rawWord+=k;
    if(rawWord.length>numSym()) enRun=true;     // overwrite => impossible zhuyin => English
    renderComposing(); return true; }
  rawWord+=k; enRun=true; renderComposing(); return true;     // non-zhuyin char => English
}
function backspace(){
  if(rawWord){ rawWord=rawWord.slice(0,-1);
    if(!enRun){ // rebuild cur from remaining raw keys
      cur=['','','']; for(const c of rawWord) if(DACHEN[c]) cur[DACHEN[c][1]]=DACHEN[c][0]; }
    else if(rawWord.length<=numSym()) enRun=false;
    if(!rawWord) resetRun();
  } else if(committed.length) committed.pop();
  renderComposing();
}

function renderComposing(){
  choosing=false;$('segs').innerHTML='';$('cands').innerHTML='';
  const p=preText();$('pre').textContent=p||'​';$('pre').classList.toggle('empty',!p);
  if(ready) $('hint').textContent=p?'空白鍵 / 轉換 → 解碼（中英自動切換）':'直接打注音或英文，會自動辨識';
}
function composed(){
  return positions.map((c,i)=>{const v=c[segSel[i]];return v;}).join('');
}
function renderSegments(){
  choosing=true;$('pre').textContent='​';
  $('hint').textContent='點字改詞　←→ 換詞　↑↓ 換字　⏎ 確認　Esc 取消';
  const wrap=$('segs');wrap.innerHTML='';
  positions.forEach((cands,i)=>{const seg=document.createElement('div');
    seg.className='seg'+(i===segFocus?' focus':'');seg.textContent=cands[segSel[i]];
    seg.onclick=()=>{segFocus=i;renderSegments();};wrap.appendChild(seg);});
  const strip2=$('cands');strip2.innerHTML='';
  (positions[segFocus]||[]).forEach((c,j)=>{const b=document.createElement('button');
    b.className='cand'+(j===segSel[segFocus]?' sel':'');b.textContent=c;
    b.onclick=()=>{segSel[segFocus]=j;const nx=positions.findIndex((p,k)=>k>segFocus&&p.length>1);if(nx>=0)segFocus=nx;renderSegments();};
    strip2.appendChild(b);});
}
function commit(){
  // join with spaces around English segments
  let out='';
  positions.forEach((c,i)=>{const v=c[segSel[i]];
    if(/[A-Za-z]/.test(v)) out+=(out&&!out.endsWith(' ')?' ':'')+v+' '; else out+=v;});
  $('out').textContent+=out.trim();
  committed=[];resetRun();positions=[];renderComposing();
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

async function convert(){
  if(!ready) return;
  if(hasPending()||rawWord) commitRun();
  const tokens=committed.map(t=>({t:t.t,v:t.t==='en'?t.v:($('toneless').checked?strip(t.v):t.v)}));
  if(!tokens.length) return;
  $('hint').textContent='解碼中…';
  try{
    const zh=tokens.filter(t=>t.t==='zh').map(t=>t.v);
    const r=zh.length?await decodeZh(zh):{chars:[],cands:[]};
    positions=[];const best=[];let zc=0;
    for(const tok of tokens){
      if(tok.t==='zh'){ positions.push(r.cands[zc]); best.push(r.chars[zc]); zc++; }
      else { positions.push([tok.v]); best.push(tok.v); }
    }
    segSel=positions.map((c,k)=>{const j=c.indexOf(best[k]);return j>=0?j:0;});
    segFocus=segSel.findIndex((_,k)=>positions[k].length>1);if(segFocus<0)segFocus=0;
    renderSegments();
  }catch(e){console.error(e);$('hint').textContent='解碼失敗：'+e.message;}
}

// ---- keyboard UI ----
const kb=$('kb');
ROWS.forEach(row=>{const r=document.createElement('div');r.className='krow';
  row.forEach(key=>{const b=document.createElement('button');b.className='key';
    const sym=DACHEN[key]?DACHEN[key][0]:(key in TONEK?TONEK[key]:'');
    b.innerHTML='<span class="k">'+key+'</span><span class="s">'+sym+'</span>';
    b.onclick=()=>feedKey(key);r.appendChild(b);});kb.appendChild(r);});
const r=document.createElement('div');r.className='krow';
const sp=document.createElement('button');sp.className='key wide';sp.textContent='空白 / 轉換';
sp.onclick=()=>{if(!choosing&&(committed.length||hasPending()||rawWord))convert();};
r.appendChild(sp);kb.appendChild(r);

document.addEventListener('keydown',e=>{
  if(e.ctrlKey||e.altKey||e.metaKey)return;const k=e.key;
  if(choosing){
    if(k==='Enter'){commit();e.preventDefault();}
    else if(k==='Escape'){renderComposing();e.preventDefault();}
    else if(k==='ArrowRight'||k==='ArrowLeft'){const d=k==='ArrowRight'?1:-1;
      for(let i=segFocus+d;i>=0&&i<positions.length;i+=d)if(positions[i].length>1){segFocus=i;break;}renderSegments();e.preventDefault();}
    else if(k==='ArrowDown'||k==='ArrowUp'){const n=positions[segFocus].length,d=k==='ArrowDown'?1:-1;
      segSel[segFocus]=(segSel[segFocus]+d+n)%n;renderSegments();e.preventDefault();}
    return;
  }
  if(k==='Enter'){if(committed.length||hasPending()||rawWord){convert();e.preventDefault();}}
  else if(k==='Backspace'){if(committed.length||hasPending()||rawWord){backspace();e.preventDefault();}}
  else if(k==='Escape'){committed=[];resetRun();renderComposing();}
  else if(k.length===1&&(DACHEN[k]||k in TONEK||/[A-Za-z0-9]/.test(k))){feedKey(k);e.preventDefault();}
});
$('convert').onclick=convert;
$('commit').onclick=commit;
$('clear').onclick=()=>{committed=[];resetRun();positions=[];renderComposing();};

(async function init(){
  const txt=await (await fetch('./phonetic_table.tsv')).text();
  for(const line of txt.split('\n')){const t=line.indexOf('\t');if(t<0)continue;tonal[line.slice(0,t)]=[...line.slice(t+1)];}
  tokenizer=await AutoTokenizer.from_pretrained(REPO);
  model=await AutoModelForCausalLM.from_pretrained(REPO,{dtype:'q8'});
  ready=true;$('convert').disabled=false;renderComposing();
})().catch(e=>{console.error(e);$('hint').textContent='模型載入失敗：'+e.message;});
renderComposing();
