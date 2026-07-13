// Offline unit test for the keystream segmenter — mirrors
// space-static/test-segment.mjs so the C++ port stays in lock-step with
// segment.js.
//   g++ -std=c++17 segment_test.cpp -o /tmp/st && /tmp/st
// Needs the phonetic table for valid bases:
//   ../../..//model/phonetic_table.tsv (path taken from argv[1] if given)
#include "segment.h"
#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using slothing::SegTok;
using slothing::Segmenter;

static int failures = 0;

static std::string show(const Segmenter &seg, const std::string &keys) {
    std::string out;
    for (const auto &t : seg.segment(keys)) {
        if (!out.empty()) out += " | ";
        out += (t.zh ? "zh:" : "en:") + t.v;
    }
    return out;
}

int main(int argc, char **argv) {
    const char *table =
        argc > 1 ? argv[1] : "../../../model/phonetic_table.tsv";
    std::set<std::string> validBase;
    {
        std::ifstream in(table);
        if (!in) {
            fprintf(stderr, "cannot open %s\n", table);
            return 2;
        }
        std::string line;
        while (std::getline(in, line)) {
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string syl = line.substr(0, tab), base;
            // strip tone marks (multi-byte UTF-8)
            for (size_t i = 0; i < syl.size();) {
                size_t len = 1;
                unsigned char c = syl[i];
                if (c >= 0xF0) len = 4;
                else if (c >= 0xE0) len = 3;
                else if (c >= 0xC0) len = 2;
                std::string ch = syl.substr(i, len);
                if (ch != "ˊ" && ch != "ˇ" && ch != "ˋ" && ch != "˙")
                    base += ch;
                i += len;
            }
            if (!base.empty()) validBase.insert(base);
        }
    }
    Segmenter seg(std::move(validBase));

    // Mirrors test-segment.mjs (keep in sync).
    const std::vector<std::array<std::string, 3>> T = {
        {"python5k4ek7", "en:python | zh:ㄓㄜˋ | zh:ㄍㄜ˙", "English then zhuyin"},
        {"driving5j;4dj;4", "en:driving | zh:ㄓㄨㄤˋ | zh:ㄎㄨㄤˋ", "English then zhuyin 2"},
        {"happya87", "en:happy | zh:ㄇㄚ˙", "English then neutral-tone"},
        {"happya8", "en:happy | zh:ㄇㄚ", "English then tone-1"},
        {"5k4python", "zh:ㄓㄜˋ | en:python", "zhuyin then English"},
        {"test123", "en:test123", "English+digits stays English"},
        {"hello", "en:hello", "pure English"},
        {"is", "en:is", "short English"},
        {"he", "en:he", "short English 2"},
        {"ishe", "en:ishe", "is-he run stays English"},
        {"5k4", "zh:ㄓㄜˋ", "plain zhuyin syllable"},
        {"rm,6", "zh:ㄐㄩㄝˊ", "3-symbol + tone"},
        {"w8", "zh:ㄊㄚ", "tone-1 (space stripped by caller)"},
        {"api2u4", "en:api | zh:ㄉㄧˋ", "English then 地"},
        {"model", "en:model", "unknown word not chopped"},
        {"world", "en:world", "unknown word not chopped 2"},
        {"banana", "en:banana", "unknown word not chopped 3"},
        {"ek", "zh:ㄍㄜ", "standalone pure-letter syllable"},
        {"vp3", "zh:ㄒㄣˇ", "typo syllable + tone accepted"},
        {"ji3vp3", "zh:ㄨㄛˇ | zh:ㄒㄣˇ", "typo syllable in context"},
        {"Python", "en:Python", "capital preserved, word whole"},
        {"ji3m/4Python", "zh:ㄨㄛˇ | zh:ㄩㄥˋ | en:Python",
         "capital mid-stream in auto zh/en"},
        {"iPhone", "en:iPhone", "capital not first, still one word"},
        {"do", "zh:ㄎㄟ",
         "zhuyin-wins: a word that IS a syllable -> zhuyin (Shift for English)"},
        {"let's", "en:let's", "apostrophe rides the English run"},
        {"upgjbj4", "zh:ㄧㄣ | zh:ㄕㄨ | zh:ㄖㄨˋ",
         "toneless zh run stays zhuyin (音輸入), not English 'upgj'"},
        {"upgj", "zh:ㄧㄣ | zh:ㄕㄨ", "confident multi-syllable run -> zhuyin"},
        {"hello", "en:hello",
         "English that coincidentally tiles (via 1-symbol syllables) stays en"},
        // en punctuation inside en text: a dachen-punct key ('-'=ㄦ, etc.)
        // flanked by English/number context is a literal char, not zhuyin.
        {"7-11", "en:7-11", "digit-hyphen-digit is literal (not 7兒11)"},
        {"3-5", "en:3-5", "lone-digit hyphen range stays literal (not 3-|ㄓ)"},
        {"2024-01-15", "en:2024-01-15",
         "ISO date stays one literal (not ㄉㄢ|24-01-15)"},
        {"COVID-19", "en:COVID-19",
         "word-hyphen-number stays literal (not COVID-|ㄅㄞ)"},
        {"a-b", "en:a-b", "letter-hyphen-letter is literal English"},
        {"0912-345", "en:0912-345", "phone-number hyphen stays literal"},
        // but a genuine zhuyin final ㄦ after a tone digit is preserved: a tone
        // key (3/4/6/7) is NOT English context, so 這兒/女兒 still parse.
        {"5k4-", "zh:ㄓㄜˋ | zh:ㄦ", "trailing ㄦ preserved (這兒)"},
        {"sm3-6", "zh:ㄋㄩˇ | zh:ㄦˊ", "ㄦˊ preserved between tone digits (女兒)"},
    };
    for (const auto &[keys, want, desc] : T) {
        std::string got = show(seg, keys);
        const bool ok = got == want;
        if (!ok) failures++;
        printf("%s  %s\n", ok ? "PASS" : "FAIL", desc.c_str());
        if (!ok) {
            printf("      keys %s\n      got  %s\n      want %s\n",
                   keys.c_str(), got.c_str(), want.c_str());
        }
    }
    printf("\n%zu/%zu passed\n", T.size() - failures, T.size());
    return failures ? 1 : 0;
}
