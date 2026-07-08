// Slothing static web IME. Runs entirely in the browser: the ZhuyinBuffer FSM
// parses keystrokes, and Transformers.js decodes the syllables with SlothLM
// (ONNX q8) under a per-position legal-char mask (the phonetic-legality
// constraint, reproduced client-side because our tokenizer is one-token-per-
// char). No server -- validated identical to the slothingd decode.
import { AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3';

const REPO = 'Luigi/slothlm-34m-zhuyin-ONNX';
const TONES = 'ˊˇˋ˙';

// ---- zhuyin keyboard FSM (Dàqiān) ----
const DACHEN = {'1':['ㄅ',0],'q':['ㄆ',0],'a':['ㄇ',0],'z':['ㄈ',0],'2':['ㄉ',0],'w':['ㄊ',0],
  's':['ㄋ',0],'x':['ㄌ',0],'e':['ㄍ',0],'d':['ㄎ',0],'c':['ㄏ',0],'r':['ㄐ',0],'f':['ㄑ',0],
  'v':['ㄒ',0],'5':['ㄓ',0],'t':['ㄔ',0],'g':['ㄕ',0],'b':['ㄖ',0],'y':['ㄗ',0],'h':['ㄘ',0],
  'n':['ㄙ',0],'u':['ㄧ',1],'j':['ㄨ',1],'m':['ㄩ',1],'8':['ㄚ',2],'i':['ㄛ',2],'k':['ㄜ',2],
  ',':['ㄝ',2],'9':['ㄞ',2],'o':['ㄟ',2],'l':['ㄠ',2],'.':['ㄡ',2],'0':['ㄢ',2],'p':['ㄣ',2],
  ';':['ㄤ',2],'/':['ㄥ',2],'-':['ㄦ',2]};
const TONE = {' ':'','6':'ˊ','3':'ˇ','4':'ˋ','7':'˙'};
const ROWS = [['1','2','3','4','5','6','7','8','9','0','-'],
  ['q','w','e','r','t','y','u','i','o','p'],['a','s','d','f','g','h','j','k','l',';'],
  ['z','x','c','v','b','n','m',',','.','/']];
const strip = s => [...s].filter(c => !TONES.includes(c)).join('');
const $ = id => document.getElementById(id);

let cur = ['','',''], committed = [], positions = [], segSel = [], segFocus = 0, choosing = false;
const hasPending = () => cur[0]||cur[1]||cur[2];
const pending = () => cur[0]+cur[1]+cur[2];
const preedit = () => committed.join('')+pending();

function feedKey(c){
  if(choosing) return false;
  if(DACHEN[c]){cur[DACHEN[c][1]]=DACHEN[c][0];renderComposing();return true;}
  if(c in TONE){if(!hasPending())return c!==' ';committed.push(cur[0]+cur[1]+cur[2]+TONE[c]);cur=['','',''];renderComposing();return true;}
  return false;
}
function backspace(){if(cur[2])cur[2]='';else if(cur[1])cur[1]='';else if(cur[0])cur[0]='';else committed.pop();renderComposing();}
function renderComposing(){
  choosing=false;$('segs').innerHTML='';$('cands').innerHTML='';
  const p=preedit();$('pre').textContent=p||'​';$('pre').classList.toggle('empty',!p);
  if(ready) $('hint').textContent=p?'空白鍵 / 轉換 → 解碼':'在鍵盤上輸入注音（實體鍵盤也可）';
}
function composed(){return positions.map((c,i)=>c[segSel[i]]).join('');}
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
function commit(){$('out').textContent+=composed();cur=['','',''];committed=[];positions=[];renderComposing();}

// ---- model + phonetic table ----
let tokenizer, model, ready=false;
const tonal = {};
function tid(ch){const ids=tokenizer.encode(ch,{add_special_tokens:false});return ids.length===1?ids[0]:null;}
function candChars(syl){
  let chars=tonal[syl];
  if(!chars){const base=strip(syl);chars=[];
    for(const k in tonal) if(strip(k)===base) for(const c of tonal[k]) if(!chars.includes(c)) chars.push(c);}
  return chars.length?chars:[syl];
}
function candSet(syl){
  const chars=candChars(syl), out=[], ids=[];
  for(const c of chars){const i=tid(c); if(i!=null&&!ids.includes(i)){ids.push(i);out.push(c);}}
  return {chars:out, ids};
}
class Masker extends LogitsProcessor{
  constructor(pos,start){super();this.pos=pos;this.start=start;}
  _call(input_ids,logits){
    const step=input_ids[0].length-this.start;
    if(step>=0&&step<this.pos.length){const allow=new Set(this.pos[step]);const d=logits.data;
      for(let i=0;i<d.length;i++) if(!allow.has(i)) d[i]=-Infinity;}
    return logits;
  }
}
async function decode(syllables){
  const sets=syllables.map(candSet);
  const posIds=sets.map(s=>s.ids);
  const prompt='<|im_start|>system\n注音轉繁體中文。<|im_end|>\n<|im_start|>user\n'
    +syllables.join(' ')+'<|im_end|>\n<|im_start|>assistant\n';
  const enc=tokenizer(prompt);
  const start=enc.input_ids.dims[1];
  const out=await model.generate({...enc,max_new_tokens:posIds.length+1,do_sample:false,
    logits_processor:new LogitsProcessorList([new Masker(posIds,start)])});
  const gen=out.tolist()[0].slice(start);
  const sentence=tokenizer.decode(gen,{skip_special_tokens:true});
  // per-position candidate lists (chars) for segment editing
  return {sentence,positions:sets.map(s=>s.chars)};
}

async function convert(){
  if(!ready) return;
  let syls=committed.slice(); if(hasPending())syls.push(pending());
  if(!syls.length) return;
  if($('toneless').checked) syls=syls.map(strip);
  $('hint').textContent='解碼中…';
  try{
    const d=await decode(syls);
    positions=d.positions;
    segSel=positions.map((cands,k)=>{const ch=[...d.sentence][k];const j=cands.indexOf(ch);return j>=0?j:0;});
    segFocus=segSel.findIndex((_,k)=>positions[k].length>1); if(segFocus<0)segFocus=0;
    renderSegments();
  }catch(e){console.error(e);$('hint').textContent='解碼失敗：'+e.message;}
}

// ---- keyboard UI ----
const kb=$('kb');
ROWS.forEach(row=>{const r=document.createElement('div');r.className='krow';
  row.forEach(key=>{const b=document.createElement('button');b.className='key';
    const sym=DACHEN[key]?DACHEN[key][0]:(TONE[key]||'');
    b.innerHTML='<span class="k">'+key+'</span><span class="s">'+sym+'</span>';
    b.onclick=()=>feedKey(key);r.appendChild(b);});kb.appendChild(r);});
const r=document.createElement('div');r.className='krow';
const sp=document.createElement('button');sp.className='key wide';sp.textContent='空白 / 轉換';
sp.onclick=()=>{if(!choosing&&(committed.length||hasPending()))convert();else feedKey(' ');};
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
  if(k==='Enter'||k===' '){if(committed.length||hasPending()){convert();e.preventDefault();}}
  else if(k==='Backspace'){if(committed.length||hasPending()){backspace();e.preventDefault();}}
  else if(k==='Escape'){cur=['','',''];committed=[];renderComposing();}
  else if(k.length===1&&(DACHEN[k]||k in TONE)){feedKey(k);e.preventDefault();}
});
$('convert').onclick=convert;
$('commit').onclick=commit;
$('clear').onclick=()=>{cur=['','',''];committed=[];positions=[];renderComposing();};

// ---- load table + model ----
(async function init(){
  const txt=await (await fetch('./phonetic_table.tsv')).text();
  for(const line of txt.split('\n')){const t=line.indexOf('\t');if(t<0)continue;tonal[line.slice(0,t)]=[...line.slice(t+1)];}
  tokenizer=await AutoTokenizer.from_pretrained(REPO);
  model=await AutoModelForCausalLM.from_pretrained(REPO,{dtype:'q8'});
  ready=true;$('convert').disabled=false;renderComposing();
})().catch(e=>{console.error(e);$('hint').textContent='模型載入失敗：'+e.message;});

renderComposing();
