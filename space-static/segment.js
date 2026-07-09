// Continuous re-segmentation of a raw keystream into zhuyin syllables + English
// runs — one principled DP pass that REPLACES the sticky-enRun FSM and its
// per-case split heuristics (overwrite→English, tone-split, space-split).
//
// The keystream is fundamentally ambiguous (`is`=ㄛㄋ is valid QWERTY *and*
// zhuyin), so we score every possible segmentation and pick the cheapest:
//   - a zhuyin token must be a phonetically-legal syllable (in validBase);
//     a confident syllable (>=2 symbols, e.g. ㄓㄜ) is cheap, a lone single
//     symbol (could be a stray English letter) is expensive;
//   - an English run costs ~per-character, so the DP prefers to peel a real
//     syllable out of a letter run (python這個) but keeps genuine words whole
//     (is he, test123) and never splits a lone ambiguous symbol.
// Irreducibly ambiguous cases (ni = ㄋㄧ vs "ni") default to the cheaper side;
// the Shift English-mode toggle remains the escape hatch, as in every IME.

// A small set of common English words (+ loanwords used in zh/en code-switch)
// so the DP keeps real words whole even when they contain a valid zhuyin
// substring (code=ㄏㄟ+de, api=ㄇㄣ+i). Not exhaustive — the Shift English mode
// covers the rest.
export const WORDS = new Set(('a about after all also am an and any api app are as at back be '+
'because been best big but buy by call can code come could data day deal do does done down '+
'driving each email end even every file find first for free from get go good google great '+
'group had happy has have he help her here hey hi him his hot hour how i if in info is issue it '+
'its just keyword know last let like line link list live login look mail make man many may me '+
'meeting more most my need new next no not note now number of off ok on one online only open or '+
'order other our out over page part people php play please post python read really right run '+
'same say search see server service she should show sir site so some sorry sound support sure '+
'system take team tech test text than thank thanks that the their them then there these they '+
'thing think this those time to today too tool top try two up us use user very video want was '+
'way we web week well what when where which who why will with word work would year yes you your'
).split(' '));

export function makeSegmenter(DACHEN, TONEK, validBase, words=WORDS){
  const isAlnum = c => c>='0'&&c<='9' || c>='A'&&c<='Z' || c>='a'&&c<='z';
  // all legal zhuyin syllables starting at i: {len, v(bopomofo±tone), syms}
  function zhAt(keys,i){
    const res=[]; let bopo='', lastSlot=-1;
    for(let L=0; L<3 && i+L<keys.length; L++){
      const d=DACHEN[keys[i+L]]; if(!d) break;
      if(d[1]<=lastSlot) break;            // initial < medial < final, each once
      lastSlot=d[1]; bopo+=d[0];
      if(validBase.has(bopo)){
        res.push({len:L+1, v:bopo, syms:L+1});
        const tk=keys[i+L+1];
        if(tk && TONEK[tk]) res.push({len:L+2, v:bopo+TONEK[tk], syms:L+1});
      }
    }
    return res;
  }
  return function segment(keys){
    const n=keys.length;
    const dp=new Array(n+1).fill(null); dp[0]={cost:0,toks:[]};
    const relax=(j,cost,tok,from)=>{ if(!dp[j]||cost<dp[j].cost) dp[j]={cost,toks:[...dp[from].toks,tok]}; };
    for(let i=0;i<n;i++){
      if(!dp[i]) continue;
      for(const s of zhAt(keys,i))                       // zhuyin syllable
        relax(i+s.len, dp[i].cost + (s.syms>=2 ? 1.0 : 2.6), {t:'zh',v:s.v}, i);
      if(isAlnum(keys[i]))                               // English run (each length)
        for(let j=i+1;j<=n && isAlnum(keys[j-1]);j++){
          const seg=keys.slice(i,j);
          // per-char cost; lone letters penalized (a 1-char English token in
          // zhuyin is almost always a mis-segmentation, e.g. api); known words
          // discounted so they stay whole despite a valid zhuyin substring (code).
          const cost=dp[i].cost + 1 + 0.6*(j-i) + ((j-i)===1?1.5:0) - (words.has(seg.toLowerCase())?3:0);
          relax(j, cost, {t:'en',v:seg}, i);
        }
      if(!isAlnum(keys[i]) && !DACHEN[keys[i]] && !TONEK[keys[i]]) // stray symbol
        relax(i+1, dp[i].cost + 1.5, {t:'en',v:keys[i]}, i);
    }
    // merge adjacent English tokens
    const out=[];
    for(const t of (dp[n]||{toks:[]}).toks){
      const p=out[out.length-1];
      if(t.t==='en' && p && p.t==='en') p.v+=t.v; else out.push({...t});
    }
    return out;
  };
}
