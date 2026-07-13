// demo_trace: feed a key sequence to the Sloth IME web demo (headless) and
// dump the observable UI state after every keystroke — same JSON schema as
// chewing_trace.c, so compare_traces.py can diff the two structurally.
//
//   node eval/ui-parity/demo_trace.mjs "su3cl3<D>1<E>" [http://localhost:8777]
import { chromium } from 'playwright';

const KEYMAP = { D:'ArrowDown', U:'ArrowUp', L:'ArrowLeft', R:'ArrowRight',
                 E:'Enter', ESC:'Escape', B:'Backspace', T:'Tab',
                 H:'Home', N:'End', S:'Shift' };  // <S> = lone-Shift English toggle

function parseKeys(s){
  const out=[];
  for(let i=0;i<s.length;i++){
    if(s[i]==='<'){ const e=s.indexOf('>',i); out.push({label:s.slice(i+1,e), press:KEYMAP[s.slice(i+1,e)]}); i=e; }
    else out.push({label:s[i], press:s[i]===' '?'Space':s[i]});
  }
  return out;
}

const keys=parseKeys(process.argv[2]||'');
const base=process.argv[3]||'http://localhost:8777';
const b=await chromium.launch();
const p=await b.newPage({viewport:{width:680,height:1000}});
await p.goto(base+'/index.html');
await p.waitForFunction(()=>window.__ui && document.getElementById('hint').textContent.includes('直接打'),{timeout:30000});
await p.evaluate(()=>localStorage.removeItem('sloth-learn'));

let prevOut=0;
for(const k of keys){
  await p.keyboard.press(k.press);
  // deterministic settle: wait for the async decode to catch up
  await p.waitForFunction(()=>window.__ui().fresh,{timeout:5000}).catch(()=>{});
  const st=await p.evaluate(()=>window.__ui());
  const commit=st.out-prevOut; prevOut=st.out;
  console.log(JSON.stringify({key:k.label, zh:st.zh, bopo:st.bopo,
    cand:st.cand, cursor:st.cursor, commit:Math.max(0,commit)}));
}
await b.close();
