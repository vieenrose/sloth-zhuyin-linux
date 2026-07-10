// Continuous re-segmentation of a raw keystream into zhuyin syllables +
// English runs — a faithful C++ port of the web demo's segment.js (the DP
// that scores every possible segmentation and picks the cheapest). Keeps the
// two implementations in lock-step: same costs, same word list, same
// post-processing. See space-static/segment.js for the design rationale.
//
// Header-only and fcitx5-free so it can be unit-tested offline
// (segment_test.cpp mirrors space-static/test-segment.mjs).
#ifndef _FCITX5_CHEWING_SEGMENT_H_
#define _FCITX5_CHEWING_SEGMENT_H_

#include "zhuyin.h"
#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <vector>

namespace slothing {

struct SegTok {
    bool zh;        // zhuyin syllable (v = bopomofo, tone mark included) …
    std::string v;  // … or literal English/number run
    bool operator==(const SegTok &o) const { return zh == o.zh && v == o.v; }
};

// Common English words (+ loanwords used in zh/en code-switch) so the DP
// keeps real words whole even when they contain a valid zhuyin substring.
// MUST match segment.js's WORDS.
inline const std::set<std::string> &segmentWords() {
    static const std::set<std::string> w = [] {
        const char *list =
            "a about after all also am an and any api app are as at back be "
            "because been best big but buy by call can code come could data "
            "day deal do does done down driving each email end even every "
            "file find first for free from get go good google great group "
            "had happy has have he help her here hey hi him his hot hour how "
            "i if in info is issue it its just keyword know last let like "
            "line link list live login look mail make man many may me "
            "meeting more most my need new next no not note now number of "
            "off ok on one online only open or order other our out over page "
            "part people php play please post python read really right run "
            "same say search see server service she should show sir site so "
            "some sorry sound support sure system take team tech test text "
            "than thank thanks that the their them then there these they "
            "thing think this those time to today too tool top try two up us "
            "use user very video want was way we web week well what when "
            "where which who why will with word work would year yes you your";
        std::set<std::string> out;
        std::string cur;
        for (const char *p = list;; p++) {
            if (*p == ' ' || *p == '\0') {
                if (!cur.empty()) out.insert(cur);
                cur.clear();
                if (*p == '\0') break;
            } else {
                cur += *p;
            }
        }
        return out;
    }();
    return w;
}

// Tone keys the segmenter recognises mid-stream (space is a run finalizer,
// not a tone signal — same as segment.js's TONEK).
inline const char *segToneMark(char c) {
    switch (c) {
    case '6': return "ˊ";
    case '3': return "ˇ";
    case '4': return "ˋ";
    case '7': return "˙";
    default: return nullptr;
    }
}

class Segmenter {
public:
    // validBase: legal TONELESS syllable bases (tone marks stripped).
    explicit Segmenter(std::set<std::string> validBase)
        : validBase_(std::move(validBase)) {}

    std::vector<SegTok> segment(const std::string &keys) const {
        const int n = static_cast<int>(keys.size());
        struct Node {
            double cost = 0;
            std::vector<SegTok> toks;
            bool set = false;
        };
        std::vector<Node> dp(n + 1);
        dp[0].set = true;
        auto relax = [&](int j, double cost, SegTok tok, int from) {
            if (!dp[j].set || cost < dp[j].cost) {
                dp[j].set = true;
                dp[j].cost = cost;
                dp[j].toks = dp[from].toks;
                dp[j].toks.push_back(std::move(tok));
            }
        };
        for (int i = 0; i < n; i++) {
            if (!dp[i].set) continue;
            for (const auto &s : zhAt(keys, i)) {
                double c = s.hard ? (s.syms >= 2 ? 1.0 : 2.6)
                                  : (s.syms >= 2 ? 3.0 : 4.2);
                if (s.typo) c += 1.5;
                relax(i + s.len, dp[i].cost + c, {true, s.v}, i);
            }
            if (isAlnum(keys[i])) {
                for (int j = i + 1; j <= n && isAlnum(keys[j - 1]); j++) {
                    std::string seg = keys.substr(i, j - i);
                    const int L = j - i;
                    std::string lower = seg;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char ch) { return std::tolower(ch); });
                    const double disc =
                        (L >= 3 && segmentWords().count(lower)) ? 3.0 : 0.0;
                    const double cost =
                        dp[i].cost +
                        std::max(0.9, 1 + 0.6 * L + (L == 1 ? 1.5 : 0.0) - disc);
                    relax(j, cost, {false, seg}, i);
                }
            }
            if (!isAlnum(keys[i]) && !dachenMap().count(keys[i]) &&
                !segToneMark(keys[i])) {
                relax(i + 1, dp[i].cost + 1.5, {false, std::string(1, keys[i])},
                      i);
            }
        }
        // merge adjacent English tokens
        std::vector<SegTok> out;
        if (dp[n].set) {
            for (auto &t : dp[n].toks) {
                if (!t.zh && !out.empty() && !out.back().zh) {
                    out.back().v += t.v;
                } else {
                    out.push_back(t);
                }
            }
        }
        // a lone English token that is exactly one valid syllable IS zhuyin
        if (out.size() == 1 && !out[0].zh) {
            if (auto v = wholeSyllable(out[0].v); !v.empty()) {
                return {{true, v}};
            }
        }
        return out;
    }

private:
    struct ZhCand {
        int len, syms;
        std::string v;
        bool hard, typo;
    };

    static bool isAlnum(char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
               (c >= 'A' && c <= 'Z');
    }

    std::vector<ZhCand> zhAt(const std::string &keys, int i) const {
        std::vector<ZhCand> res;
        std::string bopo;
        int lastSlot = -1;
        bool dig = false;
        for (int L = 0; L < 3 && i + L < static_cast<int>(keys.size()); L++) {
            const char k = keys[i + L];
            auto it = dachenMap().find(k);
            if (it == dachenMap().end()) break;
            const int slot = static_cast<int>(it->second.second);
            if (slot <= lastSlot) break; // initial < medial < final, each once
            lastSlot = slot;
            bopo += it->second.first;
            if (k >= '0' && k <= '9') dig = true;
            const char *tk = (i + L + 1 < static_cast<int>(keys.size()))
                                 ? segToneMark(keys[i + L + 1])
                                 : nullptr;
            if (validBase_.count(bopo)) {
                res.push_back({L + 1, L + 1, bopo, dig, false});
                if (tk) res.push_back({L + 2, L + 1, bopo + tk, true, false});
            } else if (tk) {
                // typo tolerance: unknown base + tone = zhuyin intent; the
                // decoder repairs it (model-scored edit-distance-1).
                res.push_back({L + 2, L + 1, bopo + tk, true, true});
            }
        }
        return res;
    }

    std::string wholeSyllable(const std::string &keys) const {
        for (const auto &s : zhAt(keys, 0)) {
            if (s.len == static_cast<int>(keys.size())) return s.v;
        }
        return {};
    }

    std::set<std::string> validBase_;
};

} // namespace slothing

#endif
