// Slothing web IME. Ports the ZhuyinBuffer FSM (engine/.../zhuyin.h) to JS,
// captures keys (physical + on-screen), and calls /api/decode for the
// grammar-constrained SlothLM decode.

// Dàqiān (default) layout: key -> [bopomofo symbol, slot] (0 init,1 med,2 final)
const DACHEN = {
  '1':['ㄅ',0],'q':['ㄆ',0],'a':['ㄇ',0],'z':['ㄈ',0],'2':['ㄉ',0],'w':['ㄊ',0],
  's':['ㄋ',0],'x':['ㄌ',0],'e':['ㄍ',0],'d':['ㄎ',0],'c':['ㄏ',0],'r':['ㄐ',0],
  'f':['ㄑ',0],'v':['ㄒ',0],'5':['ㄓ',0],'t':['ㄔ',0],'g':['ㄕ',0],'b':['ㄖ',0],
  'y':['ㄗ',0],'h':['ㄘ',0],'n':['ㄙ',0],
  'u':['ㄧ',1],'j':['ㄨ',1],'m':['ㄩ',1],
  '8':['ㄚ',2],'i':['ㄛ',2],'k':['ㄜ',2],',':['ㄝ',2],'9':['ㄞ',2],'o':['ㄟ',2],
  'l':['ㄠ',2],'.':['ㄡ',2],'0':['ㄢ',2],'p':['ㄣ',2],';':['ㄤ',2],'/':['ㄥ',2],
  '-':['ㄦ',2],
};
const TONE = {' ':'', '6':'ˊ', '3':'ˇ', '4':'ˋ', '7':'˙'};

class ZhuyinBuffer {
  constructor() { this.cur = ['','','']; this.committed = []; }
  hasPending() { return this.cur[0] || this.cur[1] || this.cur[2]; }
  key(c) {
    if (DACHEN[c]) { this.cur[DACHEN[c][1]] = DACHEN[c][0]; return true; }
    if (c in TONE) {
      if (!this.hasPending()) return c !== ' ';
      this.committed.push(this.cur[0] + this.cur[1] + this.cur[2] + TONE[c]);
      this.cur = ['','',''];
      return true;
    }
    return false;
  }
  backspace() {
    for (let i = 2; i >= 0; i--) if (this.cur[i]) { this.cur[i] = ''; return; }
    this.committed.pop();
  }
  pending() { return this.cur[0] + this.cur[1] + this.cur[2]; }
  preedit() { return this.committed.join('') + this.pending(); }
  empty() { return !this.committed.length && !this.hasPending(); }
  clear() { this.cur = ['','','']; this.committed = []; }
}

const buf = new ZhuyinBuffer();
let positions = [], segSel = [], segFocus = 0, choosing = false;

const $ = id => document.getElementById(id);
const stripTones = s => [...s].filter(c => !'ˊˇˋ˙'.includes(c)).join('');

function renderComposing() {
  choosing = false;
  $('segments').innerHTML = '';
  $('preedit').textContent = buf.preedit() || '​';
  $('preedit').classList.toggle('empty', buf.empty());
  $('hint').textContent = buf.empty()
    ? '在鍵盤上輸入注音，或點擊下方鍵盤'
    : '空白鍵 / 轉換 → 讓 SlothLM 解碼';
}

async function convert() {
  if (buf.empty()) return;
  let syllables = buf.committed.slice();
  if (buf.hasPending()) syllables.push(buf.pending()); // commit pending as tone1
  if ($('toneless').checked) syllables = syllables.map(stripTones);
  $('hint').textContent = '解碼中…';
  const r = await fetch('/api/decode', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({syllables})
  });
  const d = await r.json();
  positions = d.positions;
  segSel = positions.map((cands, i) => {
    const ch = [...d.sentence][i];
    const j = cands.indexOf(ch);
    return j >= 0 ? j : 0;
  });
  segFocus = segSel.findIndex((_, i) => positions[i].length > 1);
  if (segFocus < 0) segFocus = 0;
  renderSegments();
}

function composed() {
  return positions.map((c, i) => c[segSel[i]]).join('');
}

function renderSegments() {
  choosing = true;
  $('preedit').textContent = '​';
  $('hint').textContent = '點選字改詞　⏎ 確認　Esc 取消';
  const wrap = $('segments');
  wrap.innerHTML = '';
  positions.forEach((cands, i) => {
    const seg = document.createElement('div');
    seg.className = 'seg' + (i === segFocus ? ' focus' : '');
    seg.textContent = cands[segSel[i]];
    seg.onclick = () => { segFocus = i; renderSegments(); };
    wrap.appendChild(seg);
  });
  // candidate strip for the focused segment
  const strip = $('cands');
  strip.innerHTML = '';
  if (positions.length) {
    positions[segFocus].forEach((c, j) => {
      const b = document.createElement('button');
      b.className = 'cand' + (j === segSel[segFocus] ? ' sel' : '');
      b.textContent = c;
      b.onclick = () => {
        segSel[segFocus] = j;
        const nxt = positions.findIndex((p, k) => k > segFocus && p.length > 1);
        if (nxt >= 0) segFocus = nxt;
        renderSegments();
      };
      strip.appendChild(b);
    });
  }
}

function commit() {
  $('output').value += composed();
  buf.clear(); positions = []; segSel = [];
  renderComposing();
}

function feed(c) {
  if (choosing) return;
  if (buf.key(c)) renderComposing();
}

document.addEventListener('keydown', e => {
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  const k = e.key;
  if (choosing) {
    if (k === 'Enter') { commit(); e.preventDefault(); }
    else if (k === 'Escape') { renderComposing(); e.preventDefault(); }
    else if (k === 'ArrowRight' || k === 'ArrowLeft') {
      const d = k === 'ArrowRight' ? 1 : -1;
      for (let i = segFocus + d; i >= 0 && i < positions.length; i += d)
        if (positions[i].length > 1) { segFocus = i; break; }
      renderSegments(); e.preventDefault();
    } else if (k === 'ArrowDown' || k === 'ArrowUp') {
      const n = positions[segFocus].length, d = k === 'ArrowDown' ? 1 : -1;
      segSel[segFocus] = (segSel[segFocus] + d + n) % n;
      renderSegments(); e.preventDefault();
    }
    return;
  }
  if (k === 'Enter' || k === ' ') {
    if (!buf.empty()) { convert(); e.preventDefault(); }
  } else if (k === 'Backspace') {
    if (!buf.empty()) { buf.backspace(); renderComposing(); e.preventDefault(); }
  } else if (k === 'Escape') {
    buf.clear(); renderComposing();
  } else if (k.length === 1 && (DACHEN[k] || k in TONE)) {
    feed(k); e.preventDefault();
  }
});

// on-screen keyboard, rows match a physical Dàqiān zhuyin keyboard
const ROWS = [
  ['1','2','3','4','5','6','7','8','9','0','-'],
  ['q','w','e','r','t','y','u','i','o','p'],
  ['a','s','d','f','g','h','j','k','l',';'],
  ['z','x','c','v','b','n','m',',','.','/'],
];
function buildKeyboard() {
  const kb = $('keyboard');
  ROWS.forEach(row => {
    const r = document.createElement('div'); r.className = 'krow';
    row.forEach(key => {
      const b = document.createElement('button');
      b.className = 'key';
      const sym = DACHEN[key] ? DACHEN[key][0] : (TONE[key] || '');
      b.innerHTML = `<span class="k">${key}</span><span class="s">${sym}</span>`;
      b.onclick = () => feed(key);
      r.appendChild(b);
    });
    kb.appendChild(r);
  });
  const r = document.createElement('div'); r.className = 'krow';
  const sp = document.createElement('button');
  sp.className = 'key space'; sp.textContent = '空白 / 轉換';
  sp.onclick = () => (buf.empty() ? feed(' ') : convert());
  r.appendChild(sp);
  kb.appendChild(r);
}

$('convertBtn').onclick = convert;
$('commitBtn').onclick = commit;
$('clearBtn').onclick = () => { buf.clear(); positions = []; renderComposing(); };
buildKeyboard();
renderComposing();

// Poll model readiness; the first Space start downloads + loads the GGUF.
(async function waitReady() {
  try {
    const r = await fetch('/api/ready');
    if ((await r.json()).ready) {
      if (buf.empty() && !choosing) renderComposing();
      return;
    }
  } catch (e) {}
  if (buf.empty() && !choosing)
    $('hint').textContent = '模型載入中…（首次啟動需下載約 20MB）';
  setTimeout(waitReady, 2000);
})();
