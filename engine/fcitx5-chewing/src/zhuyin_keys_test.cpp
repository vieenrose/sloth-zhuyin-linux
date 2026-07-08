// Scale test for the zhuyin FSM: drive it with the key sequences for every
// sentence in eval/testset.tsv and confirm it reconstructs exactly the typed
// syllables. This validates keys->FSM->syllables (the engine's Stage-B input
// path) against real data, complementing run_decode_eval.py which tests
// syllables->daemon->sentence.
//
//   g++ -std=c++17 zhuyin_keys_test.cpp -o /tmp/zkt && /tmp/zkt ../../../eval/testset.tsv
#include "zhuyin.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using slothing::ZhuyinBuffer;

// bopomofo symbol -> Dachen key (inverse of zhuyin.h's map, incl. tones).
static std::unordered_map<std::string, char> bopomofoToKey() {
    std::unordered_map<std::string, char> m;
    for (auto &kv : slothing::dachenMap()) {
        m[kv.second.first] = kv.first;
    }
    m["ˊ"] = '6'; m["ˇ"] = '3'; m["ˋ"] = '4'; m["˙"] = '7';
    return m;
}

// Split a UTF-8 bopomofo syllable into codepoints.
static std::vector<std::string> chars(const std::string &s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        size_t len = (c & 0xF8) == 0xF0 ? 4 : (c & 0xF0) == 0xE0 ? 3
                     : (c & 0xE0) == 0xC0                        ? 2
                                                                 : 1;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "eval/testset.tsv";
    std::ifstream f(path);
    if (!f) { printf("cannot open %s\n", path); return 2; }
    auto b2k = bopomofoToKey();

    int total = 0, ok = 0, skipped = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::string bopomofo = line.substr(0, line.find('\t'));
        if (bopomofo.empty()) continue;
        // Build the key sequence + the expected syllable list.
        std::vector<std::string> want;
        std::string keys;
        bool bad = false;
        std::istringstream ss(bopomofo);
        std::string syl;
        const std::string tones = "ˊˇˋ˙";
        while (ss >> syl) {
            want.push_back(syl);
            std::string lastTone;
            for (auto &ch : chars(syl)) {
                auto it = b2k.find(ch);
                if (it == b2k.end()) { bad = true; break; }
                if (tones.find(ch) != std::string::npos) lastTone = it->second;
                else keys += it->second;
            }
            if (bad) break;
            keys += lastTone.empty() ? ' ' : lastTone[0]; // tone / space=tone1
        }
        if (bad) { skipped++; continue; }
        // Normalize want: tone-1 syllables have no mark (FSM emits none).
        total++;
        ZhuyinBuffer buf;
        for (char c : keys) buf.key(c);
        bool match = buf.syllables().size() == want.size();
        for (size_t i = 0; match && i < want.size(); i++)
            if (buf.syllables()[i] != want[i]) match = false;
        if (match) ok++;
        else if (total - ok <= 8) {
            std::string got;
            for (auto &s : buf.syllables()) got += s + " ";
            printf("  MISS keys='%s' want='%s' got='%s'\n", keys.c_str(),
                   bopomofo.c_str(), got.c_str());
        }
    }
    printf("\nkeys->FSM->syllables: %d/%d exact (%d skipped, unmapped symbol)\n",
           ok, total, skipped);
    return ok == total ? 0 : 1;
}
