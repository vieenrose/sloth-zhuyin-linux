// 聯想 (next-word association) engine — frontend-free, shared by Android,
// IBus, fcitx5 (and mirrored in JS for the web demo). After the user commits
// text ending in character X, predict what follows: the user's OWN bigrams
// first (learned from every commit), then the dictionary completions of
// high-frequency words starting with X (model/assoc_tc.tsv, built from
// libchewing's tsi.src by model/build_assoc.py — 微軟新注音-style 聯想).
//
// Single-threaded by design (drive it from the frontend's commit path; the
// Android session wraps calls in its own mutex). Offline-testable: see
// core_test.cpp.
#ifndef _SLOTHING_COMMON_ASSOC_H_
#define _SLOTHING_COMMON_ASSOC_H_

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace slothing {

class AssocEngine {
public:
    static constexpr size_t kPredictions = 8;

    // dict: assoc_tc.tsv bytes (head-char \t completions, space-separated).
    // userPath: personal bigram store ("" = in-memory only).
    void load(const std::string &dictTsv, std::string userPath = "") {
        loadDict(dictTsv);
        loadUser(std::move(userPath));
    }

    // Feed committed text: counts adjacent CJK bigrams (persisted) and
    // advances the prediction tail. Punctuation/latin breaks adjacency and
    // clears the tail when the text ends non-CJK.
    void record(const std::string &s) {
        std::string prev = tail_;
        bool dirty = false;
        for (const auto &cp : splitCps(s)) {
            if (!isCjk(cp)) {
                prev.clear();
                continue;
            }
            if (!prev.empty()) {
                user_[{prev, cp}]++;
                dirty = true;
            }
            prev = cp;
        }
        tail_ = prev;
        if (dirty) saveUser();
    }

    // Predictions for what follows the tail: personal (count-ranked) first,
    // then dictionary completions, deduped, capped. Empty when no tail.
    std::vector<std::string> predictions() const {
        std::vector<std::string> out;
        if (tail_.empty()) return out;
        std::vector<std::pair<int, std::string>> mine;
        for (const auto &kv : user_)
            if (kv.first.first == tail_)
                mine.push_back({kv.second, kv.first.second});
        std::sort(mine.begin(), mine.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
        for (auto &p : mine) {
            if (out.size() >= kPredictions) break;
            out.push_back(p.second);
        }
        auto it = dict_.find(tail_);
        if (it != dict_.end())
            for (const auto &c : it->second) {
                if (out.size() >= kPredictions) break;
                if (std::find(out.begin(), out.end(), c) == out.end())
                    out.push_back(c);
            }
        return out;
    }

    bool hasTail() const { return !tail_.empty(); }

    // Field/application switch: predictions must not carry over.
    void clearTail() { tail_.clear(); }

private:
    static std::vector<std::string> splitCps(const std::string &s) {
        std::vector<std::string> out;
        for (size_t i = 0; i < s.size();) {
            unsigned char c = s[i];
            size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            out.push_back(s.substr(i, n));
            i += n;
        }
        return out;
    }
    static bool isCjk(const std::string &cp) {
        // CJK Unified Ideographs (U+4E00..U+9FFF): E4B880..E9BFBF in UTF-8
        return cp.size() == 3 &&
               ((static_cast<unsigned char>(cp[0]) == 0xE4 &&
                 static_cast<unsigned char>(cp[1]) >= 0xB8) ||
                (static_cast<unsigned char>(cp[0]) >= 0xE5 &&
                 static_cast<unsigned char>(cp[0]) <= 0xE9));
    }

    void loadDict(const std::string &tsv) {
        size_t i = 0;
        while (i < tsv.size()) {
            size_t nl = tsv.find('\n', i);
            std::string line = tsv.substr(
                i, nl == std::string::npos ? std::string::npos : nl - i);
            i = (nl == std::string::npos) ? tsv.size() : nl + 1;
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string head = line.substr(0, tab);
            std::vector<std::string> comps;
            std::string rest = line.substr(tab + 1);
            size_t j = 0;
            while (j < rest.size()) {
                size_t sp = rest.find(' ', j);
                std::string c = rest.substr(
                    j, sp == std::string::npos ? std::string::npos : sp - j);
                if (!c.empty()) comps.push_back(c);
                if (sp == std::string::npos) break;
                j = sp + 1;
            }
            if (!comps.empty()) dict_.emplace(std::move(head), std::move(comps));
        }
    }

    void loadUser(std::string path) {
        userFile_ = std::move(path);
        if (userFile_.empty()) return;
        std::ifstream f(userFile_);
        std::string line;
        while (std::getline(f, line)) {
            auto t1 = line.find('\t');
            if (t1 == std::string::npos) continue;
            auto t2 = line.find('\t', t1 + 1);
            if (t2 == std::string::npos) continue;
            int n = std::atoi(line.c_str() + t2 + 1);
            if (n > 0)
                user_[{line.substr(0, t1), line.substr(t1 + 1, t2 - t1 - 1)}] = n;
        }
    }

    void saveUser() {
        if (userFile_.empty()) return;
        std::ofstream f(userFile_);
        for (const auto &kv : user_)
            f << kv.first.first << "\t" << kv.first.second << "\t" << kv.second
              << "\n";
    }

    std::unordered_map<std::string, std::vector<std::string>> dict_;
    std::map<std::pair<std::string, std::string>, int> user_;
    std::string userFile_;
    std::string tail_;
};

} // namespace slothing

#endif // _SLOTHING_COMMON_ASSOC_H_
