// Offline unit test for the zhuyin keyboard FSM (no fcitx5/libchewing).
//   g++ -std=c++17 zhuyin_test.cpp -o /tmp/zt && /tmp/zt
#include "zhuyin.h"
#include <cstdio>
#include <string>
#include <vector>

using slothing::ZhuyinBuffer;

static int failures = 0;

// Type a key string, return the committed syllables joined with '|'.
static std::string type(const std::string &keys) {
    ZhuyinBuffer b;
    for (char c : keys) {
        b.key(c);
    }
    std::string out;
    for (size_t i = 0; i < b.syllables().size(); i++) {
        if (i) out += "|";
        out += b.syllables()[i];
    }
    return out;
}

static void eq(const std::string &keys, const std::string &want) {
    std::string got = type(keys);
    bool ok = got == want;
    printf("  [%s] '%s' -> '%s'%s\n", ok ? "OK" : "!!", keys.c_str(),
           got.c_str(), ok ? "" : (" want '" + want + "'").c_str());
    if (!ok) failures++;
}

int main() {
    // 我在重新考慮: ㄨㄛˇ ㄗㄞˋ ㄔㄨㄥˊ ㄒㄧㄣ ㄎㄠˇ ㄌㄩˋ
    //   ㄨㄛˇ = j i 3 ; ㄗㄞˋ = y 9 4 ; ㄔㄨㄥˊ = t j / 6 (tone1? no: ˊ=6)
    //   ㄒㄧㄣ = v u p (space) ; ㄎㄠˇ = d l 3 ; ㄌㄩˋ = x m 4
    printf("zhuyin FSM test\n");
    eq("ji3", "ㄨㄛˇ");
    eq("y94", "ㄗㄞˋ");
    eq("tj/6", "ㄔㄨㄥˊ");
    eq("vup ", "ㄒㄧㄣ");           // space = tone 1, no mark
    eq("dl3", "ㄎㄠˇ");
    eq("xm4", "ㄌㄩˋ");
    // full sentence in one go
    eq("ji3y94tj/6vup dl3xm4",
       "ㄨㄛˇ|ㄗㄞˋ|ㄔㄨㄥˊ|ㄒㄧㄣ|ㄎㄠˇ|ㄌㄩˋ");
    // 你好: ㄋㄧˇ ㄏㄠˇ = s u 3 , c l 3
    eq("su3cl3", "ㄋㄧˇ|ㄏㄠˇ");
    // initial-only + tone: ㄗ = y (space)
    eq("y ", "ㄗ");
    // retype initial overwrites: 1 then 2 -> ㄉ
    eq("2i ", "ㄉㄛ");

    // backspace + preedit checks
    {
        ZhuyinBuffer b;
        b.key('j'); b.key('i'); // pending ㄨㄛ
        bool pe = b.preedit() == "ㄨㄛ";
        printf("  [%s] pending preedit '%s'\n", pe ? "OK" : "!!",
               b.preedit().c_str());
        if (!pe) failures++;
        b.backspace(); // removes ㄛ
        bool bs = b.pending() == "ㄨ";
        printf("  [%s] backspace slot -> '%s'\n", bs ? "OK" : "!!",
               b.pending().c_str());
        if (!bs) failures++;
        b.key('3'); // commit ㄨˇ
        b.backspace(); // removes committed syllable
        bool bc = b.empty();
        printf("  [%s] backspace committed -> empty=%d\n", bc ? "OK" : "!!",
               b.empty());
        if (!bc) failures++;
    }
    // non-zhuyin key is rejected (engine passthrough)
    {
        ZhuyinBuffer b;
        bool consumed = b.key('A');
        printf("  [%s] 'A' consumed=%d (want 0)\n", !consumed ? "OK" : "!!",
               consumed);
        if (consumed) failures++;
    }

    printf(failures ? "\nFAILED: %d\n" : "\nall passed\n", failures);
    return failures ? 1 : 0;
}
