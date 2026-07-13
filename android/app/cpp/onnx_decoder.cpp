// OnnxDecoder — faithful in-process port of engine/slothingd/slothingd_e.py.
// See onnx_decoder.h for the contract. Loads the ternary GGUF encoder (by path)
// + vocab/table strings (from Android assets) and reproduces the daemon's decode
// with the libslothe/ggml forward pass (formerly ONNX Runtime).
#include "onnx_decoder.h"

#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>

#ifdef __ANDROID__
#include <android/log.h>
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "slothing.onnx", __VA_ARGS__)
#else
#include <cstdio>
#define LOGE(...) std::fprintf(stderr, __VA_ARGS__)
#endif

namespace slothing {
namespace {

// UTF-8 codepoint split (each element is one whole UTF-8 char).
std::vector<std::string> cps(const std::string &s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        out.push_back(s.substr(i, n));
        i += n;
    }
    return out;
}
// TONES = "ˊˇˋ˙" (U+02CA U+02C7 U+02CB U+02D9).
bool isTone(const std::string &cp) {
    return cp == "ˊ" || cp == "ˇ" || cp == "ˋ" || cp == "˙";
}
std::string baseOf(const std::string &syl) {
    std::string b;
    for (auto &c : cps(syl))
        if (!isTone(c)) b += c;
    return b;
}
std::string toneOf(const std::string &syl) {
    std::string t;
    for (auto &c : cps(syl))
        if (isTone(c)) t += c;
    return t;
}
int baseLen(const std::string &syl) {
    int n = 0;
    for (auto &c : cps(syl))
        if (!isTone(c)) ++n;
    return n;
}
// Damerau-Levenshtein capped at 1 over codepoint vectors (== _dl1).
int dl1(const std::vector<std::string> &a, const std::vector<std::string> &b) {
    if (a == b) return 0;
    int la = static_cast<int>(a.size()), lb = static_cast<int>(b.size());
    if (std::abs(la - lb) > 1) return 2;
    if (la == lb) {
        std::vector<int> diff;
        for (int i = 0; i < la; ++i)
            if (a[i] != b[i]) diff.push_back(i);
        if (diff.size() == 1) return 1;
        if (diff.size() == 2 && diff[1] == diff[0] + 1 &&
            a[diff[0]] == b[diff[1]] && a[diff[1]] == b[diff[0]])
            return 1; // adjacent transpose
        return 2;
    }
    const auto &s = la < lb ? a : b;
    const auto &l = la < lb ? b : a;
    for (size_t i = 0; i <= s.size(); ++i) { // one insert into s to reach l
        std::vector<std::string> t(s.begin(), s.begin() + i);
        t.push_back(l[i]);
        t.insert(t.end(), s.begin() + i, s.end());
        if (t == l) return 1;
    }
    return 2;
}

std::unordered_map<std::string, int> parseJsonMap(const std::string &blob) {
    std::unordered_map<std::string, int> m;
    if (blob.empty()) return m;
    json j = json::parse(blob, nullptr, /*allow_exceptions=*/false);
    if (!j.is_object()) return m;
    m.reserve(j.size());
    for (auto it = j.begin(); it != j.end(); ++it)
        m[it.key()] = it.value().get<int>();
    return m;
}

} // namespace

OnnxDecoder::OnnxDecoder(const std::string &ggufPath,
                         const std::string &sylVocabJson,
                         const std::string &char2idJson,
                         const std::string &tableTsv, int /*numThreads*/,
                         std::string learnPath) {
    // Load the ternary GGUF encoder. slothe_load exits(1) on a fatal file error
    // (matching the validated port); on a clean load model_ is non-null and
    // ok()/nativeReady report ready. The shipped checkpoint has no hints input,
    // so the char-hint channel stays inert (hasHints_ = false).
    model_ = slothe_load(ggufPath.c_str());
    if (!model_) LOGE("failed to load GGUF: %s", ggufPath.c_str());
    hasHints_ = false;

    sylVocab_ = parseJsonMap(sylVocabJson);
    char2id_ = parseJsonMap(char2idJson);
    // n_char is the model's output width (authoritative for the logits stride).
    if (model_) nChar_ = slothe_n_char(model_);
    loadTable(tableTsv);
    loadLearn(learnPath);

    // warm-up so the first real keystroke is hot (你好).
    if (model_)
        decodeImpl({"ㄋㄧˇ", "ㄏㄠˇ"}, 1, {}, "");
}

OnnxDecoder::~OnnxDecoder() { slothe_free(model_); }

void OnnxDecoder::loadTable(const std::string &tsv) {
    size_t i = 0;
    while (i < tsv.size()) {
        size_t nl = tsv.find('\n', i);
        std::string line =
            tsv.substr(i, nl == std::string::npos ? std::string::npos : nl - i);
        i = (nl == std::string::npos) ? tsv.size() : nl + 1;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string s = line.substr(0, tab), chars = line.substr(tab + 1);
        if (chars.empty()) continue;
        std::vector<std::string> cl = cps(chars);
        tonal_[s] = cl;
        std::string b = baseOf(s);
        auto &dst = toneless_[b];
        for (auto &ch : cl)
            if (std::find(dst.begin(), dst.end(), ch) == dst.end())
                dst.push_back(ch);
    }
}

void OnnxDecoder::loadLearn(const std::string &path) {
    learnFile_ = path;
    if (path.empty()) return;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto t1 = line.find('\t');
        if (t1 == std::string::npos) continue;
        auto t2 = line.find('\t', t1 + 1);
        if (t2 == std::string::npos) continue;
        std::string kind = line.substr(0, t1),
                    key = line.substr(t1 + 1, t2 - t1 - 1),
                    val = line.substr(t2 + 1);
        if (kind == "c") {
            learnChar_[key] = val;
        } else if (kind == "p") {
            auto sp = key.find(' ');
            if (sp != std::string::npos)
                learnPhrase_[{key.substr(0, sp), key.substr(sp + 1)}] = val;
        }
    }
}

void OnnxDecoder::saveLearn() {
    if (learnFile_.empty()) return;
    std::ofstream f(learnFile_);
    for (auto &kv : learnChar_)
        f << "c\t" << kv.first << "\t" << kv.second << "\n";
    for (auto &kv : learnPhrase_)
        f << "p\t" << kv.first.first << " " << kv.first.second << "\t"
          << kv.second << "\n";
}

int OnnxDecoder::sylId(const std::string &s) const {
    auto it = sylVocab_.find(s);
    return it == sylVocab_.end() ? 1 /*<unk>*/ : it->second;
}

std::vector<std::pair<std::string, int>>
OnnxDecoder::cands(const std::string &syl) const {
    // Tone-optional decoding removed: an UNMARKED syllable is tone-1 (its bare
    // row), not a union across all tones — direct lookup (bare -> tone-1;
    // marked -> that tone). toneless_ is kept only for typoFixes (edit-dist-1).
    const std::vector<std::string> *chars = nullptr;
    auto it = tonal_.find(syl);
    if (it != tonal_.end()) chars = &it->second;
    std::vector<std::pair<std::string, int>> out;
    if (chars)
        for (auto &c : *chars) {
            auto ci = char2id_.find(c);
            if (ci != char2id_.end()) out.push_back({c, ci->second});
        }
    return out;
}

std::vector<std::pair<std::string, std::vector<std::string>>>
OnnxDecoder::typoFixes(const std::string &syl) const {
    std::string tone = toneOf(syl);
    auto baseCp = cps(baseOf(syl));
    std::vector<std::pair<std::string, std::vector<std::string>>> out;
    for (auto &kv : toneless_) {
        if (dl1(cps(kv.first), baseCp) > 1) continue;
        const std::string &b = kv.first;
        if (!tone.empty()) {
            auto it = tonal_.find(b + tone);
            if (it != tonal_.end()) out.push_back({b + tone, it->second});
        } else {
            auto it = tonal_.find(b);
            if (it != tonal_.end()) out.push_back({b, it->second});
            else out.push_back({b, kv.second});
        }
    }
    return out;
}

std::map<std::pair<int, std::string>, double>
OnnxDecoder::bonus(const std::vector<std::string> &syl) const {
    std::map<std::pair<int, std::string>, double> b;
    for (int i = 0; i < static_cast<int>(syl.size()); ++i) {
        auto it = learnChar_.find(syl[i]);
        if (it != learnChar_.end()) b[{i, it->second}] += CHAR_BONUS;
    }
    for (int i = 0; i + 1 < static_cast<int>(syl.size()); ++i) {
        auto it = learnPhrase_.find({syl[i], syl[i + 1]});
        if (it != learnPhrase_.end()) {
            auto ph = cps(it->second);
            if (ph.size() >= 2) {
                b[{i, ph[0]}] += PHRASE_BONUS;
                b[{i + 1, ph[1]}] += PHRASE_BONUS;
            }
        }
    }
    return b;
}

std::vector<float>
OnnxDecoder::runForward(const std::vector<int64_t> &syl, int B, int T,
                        const std::vector<int64_t> * /*hints*/) {
    // ggml forward, one sequence at a time (the ternary encoder is bidirectional
    // and unmasked — no amask needed; the shipped GGUF has no hints input, so
    // the hint channel is ignored). slothe_logits writes [T*nChar] row-major, so
    // laying B rows back-to-back reproduces the old ORT [B*T*nChar] layout.
    std::vector<float> out(static_cast<size_t>(B) * T * nChar_);
    if (!model_) return out;
    std::vector<int32_t> row(T);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t)
            row[t] = static_cast<int32_t>(syl[static_cast<size_t>(b) * T + t]);
        slothe_logits(model_, row.data(), T,
                      out.data() + static_cast<size_t>(b) * T * nChar_);
    }
    return out;
}

DecodeResult OnnxDecoder::decode(const std::vector<std::string> &syllables,
                                 int n, const std::string &context) {
    if (!model_) return {};
    return decodeImpl(syllables, n, {}, context);
}

DecodeResult
OnnxDecoder::decodeWithHints(const std::vector<std::string> &syllables,
                             const std::map<int, std::string> &hints) {
    if (!model_) return {};
    return decodeImpl(syllables, 1, hints, "");
}

DecodeResult OnnxDecoder::decodeImpl(const std::vector<std::string> &syl, int n,
                                     const std::map<int, std::string> &hints,
                                     const std::string &context) {
    DecodeResult R;
    if (syl.empty()) return R;

    // context chars ride the hint channel on <pad> prefix positions
    std::vector<std::string> ctx;
    if (!context.empty() && hasHints_)
        for (auto &c : cps(context))
            if (char2id_.count(c)) ctx.push_back(c);
    if (static_cast<int>(ctx.size()) > CTX_MAX)
        ctx.erase(ctx.begin(), ctx.end() - CTX_MAX);
    int L = static_cast<int>(ctx.size());

    std::vector<int64_t> hintIds;
    bool haveHints = false;
    if ((!hints.empty() || !ctx.empty()) && hasHints_) {
        haveHints = true;
        hintIds.assign(L + syl.size(), 0);
        for (int j = 0; j < L; ++j) hintIds[j] = char2id_.at(ctx[j]) + 1;
        for (auto &kv : hints) {
            int i = kv.first;
            if (i >= 0 && i < static_cast<int>(syl.size()) &&
                char2id_.count(kv.second))
                hintIds[L + i] = char2id_.at(kv.second) + 1;
        }
    }

    std::vector<int64_t> ids(L, 0);
    for (auto &s : syl) ids.push_back(sylId(s));
    int T = static_cast<int>(ids.size());
    std::map<int, std::vector<std::pair<std::string, int>>> candOverride;

    // ED1 typo repair with insertion prior (one batched forward per impossible)
    for (int i = 0; i < static_cast<int>(syl.size()); ++i) {
        if (!cands(syl[i]).empty()) continue;
        std::vector<std::pair<std::string, std::vector<std::string>>> fixes;
        for (auto &f : typoFixes(syl[i])) {
            if (!sylVocab_.count(f.first)) continue;
            bool anyc = false;
            for (auto &c : f.second)
                if (char2id_.count(c)) anyc = true;
            if (anyc) fixes.push_back(f);
        }
        if (fixes.empty()) continue;
        int F = static_cast<int>(fixes.size());
        std::vector<int64_t> batch(static_cast<size_t>(F) * T);
        std::vector<int64_t> hb;
        if (haveHints) hb.resize(static_cast<size_t>(F) * T);
        for (int j = 0; j < F; ++j) {
            for (int t = 0; t < T; ++t) {
                batch[static_cast<size_t>(j) * T + t] = ids[t];
                if (haveHints) hb[static_cast<size_t>(j) * T + t] = hintIds[t];
            }
            batch[static_cast<size_t>(j) * T + L + i] =
                sylVocab_.at(fixes[j].first);
        }
        auto lgb = runForward(batch, F, T, haveHints ? &hb : nullptr);
        int typedLen = baseLen(syl[i]);
        int bestJ = 0;
        double bestV = -1e30;
        for (int j = 0; j < F; ++j) {
            double v = -1e30;
            for (auto &c : fixes[j].second) {
                auto it = char2id_.find(c);
                if (it != char2id_.end())
                    v = std::max(v, static_cast<double>(
                                        lg(lgb, j, L + i, T, it->second)));
            }
            v += 1.5 * (baseLen(fixes[j].first) - typedLen); // insertion prior
            if (v > bestV) {
                bestV = v;
                bestJ = j;
            }
        }
        ids[L + i] = sylVocab_.at(fixes[bestJ].first);
        std::vector<std::pair<std::string, int>> ov;
        for (auto &c : fixes[bestJ].second) {
            auto it = char2id_.find(c);
            if (it != char2id_.end()) ov.push_back({c, it->second});
        }
        candOverride[i] = std::move(ov);
    }

    // main forward (drop L context rows via the L+i offset)
    auto flat = runForward(ids, 1, T, haveHints ? &hintIds : nullptr);
    auto B = bonus(syl);
    std::vector<std::string> best;
    std::vector<std::vector<std::string>> ranked;
    std::vector<std::tuple<double, int, std::string>> margins;
    for (int i = 0; i < static_cast<int>(syl.size()); ++i) {
        const auto &cs = candOverride.count(i) ? candOverride[i] : cands(syl[i]);
        if (cs.empty()) return R; // unknown -> no legal decode
        std::vector<std::pair<double, std::string>> scored;
        for (auto &pr : cs) {
            double s = lg(flat, 0, L + i, T, pr.second);
            auto it = B.find({i, pr.first});
            if (it != B.end()) s += it->second;
            scored.push_back({s, pr.first});
        }
        std::sort(scored.begin(), scored.end(), [](auto &a, auto &b) {
            return a.first != b.first ? a.first > b.first : a.second > b.second;
        });
        std::vector<std::string> rk;
        for (auto &p : scored) rk.push_back(p.second);
        ranked.push_back(rk);
        best.push_back(scored[0].second);
        if (scored.size() > 1)
            margins.push_back(
                {scored[0].first - scored[1].first, i, scored[1].second});
    }
    std::string s0;
    for (auto &c : best) s0 += c;
    R.sentences.push_back(s0);
    std::sort(margins.begin(), margins.end()); // lowest margin first
    for (int k = 0; k < std::max(0, n - 1) && k < static_cast<int>(margins.size());
         ++k) {
        auto alt = best;
        alt[std::get<1>(margins[k])] = std::get<2>(margins[k]);
        std::string s;
        for (auto &c : alt) s += c;
        R.sentences.push_back(s);
    }
    R.candidates = std::move(ranked);
    return R;
}

std::vector<std::pair<double, std::string>>
OnnxDecoder::phrasesScored(const std::vector<std::string> &syllables, int at,
                           int n) {
    if (!model_) return {};
    return phrases(syllables, at, n);
}

std::vector<std::pair<double, std::string>>
OnnxDecoder::phrases(const std::vector<std::string> &syl, int i, int n) {
    std::vector<std::pair<double, std::string>> out;
    if (!(i >= 0 && i < static_cast<int>(syl.size()) - 1)) return out;
    auto c0 = cands(syl[i]), c1 = cands(syl[i + 1]);
    if (c0.empty() || c1.empty()) return out;
    std::vector<int64_t> ids;
    for (auto &s : syl) ids.push_back(sylId(s));
    int T = static_cast<int>(ids.size());
    auto flat = runForward(ids, 1, T, nullptr); // context-free, no hints
    auto top = [&](const std::vector<std::pair<std::string, int>> &cs, int pos,
                   int k) {
        std::vector<double> v;
        for (auto &c : cs) v.push_back(lg(flat, 0, pos, T, c.second));
        double mx = *std::max_element(v.begin(), v.end());
        double Z = 0;
        for (auto &x : v) {
            x = std::exp(x - mx);
            Z += x;
        }
        for (auto &x : v) x /= Z;
        std::vector<int> idx(cs.size());
        for (int j = 0; j < static_cast<int>(idx.size()); ++j) idx[j] = j;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] > v[b]; });
        std::vector<std::pair<std::string, double>> r;
        for (int j = 0; j < static_cast<int>(idx.size()) && j < k; ++j)
            r.push_back({cs[idx[j]].first, v[idx[j]]});
        return r;
    };
    auto t0 = top(c0, i, 5), t1 = top(c1, i + 1, 5);
    std::vector<std::pair<double, std::string>> scored; // (jointProb, phrase)
    for (auto &a : t0)
        for (auto &b : t1)
            scored.push_back({a.second * b.second, a.first + b.first});
    std::sort(scored.begin(), scored.end(),
              [](auto &x, auto &y) { return x.first > y.first; });
    double cut = scored.empty() ? 0 : std::max(0.06, 0.15 * scored[0].first);
    for (auto &pr : scored) {
        if (pr.first < cut || static_cast<int>(out.size()) >= n) break;
        bool dup = false;
        for (auto &o : out)
            if (o.second == pr.second) dup = true;
        if (!dup) out.push_back(pr);
    }
    return out; // best-first (prob, phrase), matches queryPhrasesScored
}

void OnnxDecoder::learn(const json &payload) {
    std::lock_guard<std::mutex> lk(learnMu_);
    if (payload.contains("chars"))
        for (auto &e : payload["chars"])
            if (e.is_array() && e.size() == 2)
                learnChar_[e[0].get<std::string>()] = e[1].get<std::string>();
    if (payload.contains("phrases"))
        for (auto &e : payload["phrases"])
            if (e.is_array() && e.size() == 2) {
                std::string k = e[0].get<std::string>();
                auto sp = k.find(' ');
                if (sp != std::string::npos)
                    learnPhrase_[{k.substr(0, sp), k.substr(sp + 1)}] =
                        e[1].get<std::string>();
            }
    saveLearn();
}

} // namespace slothing
