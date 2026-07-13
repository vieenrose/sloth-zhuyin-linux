// 聯想 (next-word association) — JS twin of engine/common/assoc.h, kept in
// lock-step (same dict format, same personal-bigram semantics, same ranking:
// personal counts first, then dictionary completions; punctuation/latin
// breaks adjacency and clears the prediction tail).
export function makeAssoc(storageKey = 'sloth-assoc') {
  let dict = {}, user = {}, tail = '';
  try { user = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch (e) {}
  const save = () => { try { localStorage.setItem(storageKey, JSON.stringify(user)); } catch (e) {} };
  const isCjk = c => c >= '一' && c <= '鿿';
  const MAX = 8;
  return {
    // assoc_tc.tsv: head-char \t completions (space-separated, freq-ranked)
    load(tsv) {
      for (const line of tsv.split('\n')) {
        const t = line.indexOf('\t'); if (t < 0) continue;
        const head = line.slice(0, t), comps = line.slice(t + 1).split(' ').filter(Boolean);
        if (comps.length) dict[head] = comps;
      }
    },
    // feed committed text: count CJK bigrams, advance the tail
    record(s) {
      let prev = tail, dirty = false;
      for (const cp of [...s]) {
        if (!isCjk(cp)) { prev = ''; continue; }
        if (prev) { const k = prev + '\t' + cp; user[k] = (user[k] || 0) + 1; dirty = true; }
        prev = cp;
      }
      tail = prev;
      if (dirty) save();
    },
    predictions() {
      if (!tail) return [];
      const out = [];
      const mine = Object.entries(user)
        .filter(([k]) => k.startsWith(tail + '\t'))
        .sort((a, b) => b[1] - a[1])
        .map(([k]) => k.split('\t')[1]);
      for (const m of mine) { if (out.length >= MAX) break; out.push(m); }
      for (const c of (dict[tail] || [])) {
        if (out.length >= MAX) break;
        if (!out.includes(c)) out.push(c);
      }
      return out;
    },
    hasTail() { return !!tail; },
    clearTail() { tail = ''; },
  };
}
