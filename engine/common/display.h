// Frontend-free display/formatting helpers shared by the fcitx5 and IBus
// engines (extracted from eim.cpp): UTF-8 slicing, zh/en join spacing rules,
// fullwidth conversion, the 微軟-style symbol categories and the punctuation
// map. No fcitx/ibus/glib types.
#ifndef _SLOTHING_COMMON_DISPLAY_H_
#define _SLOTHING_COMMON_DISPLAY_H_

#include "segment.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace slothing {

// ---- UTF-8 (dependency-free replacements for fcitx-utils/utf8) ----------

inline size_t utf8SeqLen(unsigned char c) {
    if (c >= 0xF0) return 4;
    if (c >= 0xE0) return 3;
    if (c >= 0xC0) return 2;
    return 1;
}

inline size_t utf8Length(const std::string &s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); i += utf8SeqLen(s[i])) n++;
    return n;
}

// Byte offset of the `nchars`-th character.
inline size_t utf8ByteOffset(const std::string &s, size_t nchars) {
    size_t i = 0;
    for (size_t k = 0; k < nchars && i < s.size(); k++) i += utf8SeqLen(s[i]);
    return i;
}

// Slice UTF-8 chars [from, to) of s (char indices, not bytes).
inline std::string utf8CharSlice(const std::string &s, int from, int to) {
    size_t fromByte = utf8ByteOffset(s, from);
    size_t toByte = utf8ByteOffset(s, to);
    return s.substr(fromByte, toByte - fromByte);
}

inline std::vector<std::string> splitUtf8(const std::string &s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t len = utf8SeqLen(s[i]);
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

// ---- zh/en display joining ------------------------------------------------

inline bool isAsciiRun(const std::string &v) {
    for (unsigned char c : v) {
        if (c >= 0x80) return false;
    }
    return true;
}

// Display string for a token list: zh -> bopomofo (or a supplied char),
// en -> the literal with spaces around it. Doubles collapsed at the end.
struct JoinResult {
    std::string text;
    size_t cursorBytes = 0;
};

// Join per-token display strings with the web demo's spacing rules (ASCII
// English runs spaced, fullwidth punctuation hugging, zh plain), inserting
// `tail` at token index `cursorTok` (-1 = end) and reporting the caret's
// byte offset (after the tail).
inline JoinResult joinDisplay(const std::vector<SegTok> &toks,
                              const std::vector<std::string> &disp,
                              int cursorTok, const std::string &tail) {
    JoinResult r;
    auto append = [&](const std::string &piece, bool zh) {
        if (piece.empty()) return;
        if (zh) {
            r.text += piece;
        } else if (isAsciiRun(piece)) {
            if (!r.text.empty() && r.text.back() != ' ') r.text += ' ';
            r.text += piece;
            r.text += ' ';
        } else { // fullwidth punctuation
            if (!r.text.empty() && r.text.back() == ' ') r.text.pop_back();
            r.text += piece;
        }
    };
    const int n = static_cast<int>(toks.size());
    const int cur = (cursorTok < 0 || cursorTok > n) ? n : cursorTok;
    for (int i = 0; i <= n; i++) {
        if (i == cur) {
            r.text += tail; // already display-formed (mixed zh/en possible)
            r.cursorBytes = r.text.size();
        }
        if (i == n) break;
        const std::string &d =
            (i < static_cast<int>(disp.size()) && !disp[i].empty()) ? disp[i]
                                                                    : toks[i].v;
        append(d, toks[i].zh);
    }
    if (!r.text.empty() && r.text.back() == ' ') {
        r.text.pop_back();
        if (r.cursorBytes > r.text.size()) r.cursorBytes = r.text.size();
    }
    return r;
}

inline std::string toksDisplay(const std::vector<SegTok> &toks) {
    std::string out;
    for (const auto &t : toks) {
        if (t.zh) {
            out += t.v;
        } else if (isAsciiRun(t.v)) {
            out += " " + t.v + " ";
        } else { // fullwidth punctuation: no spaces, eat one before it
            if (!out.empty() && out.back() == ' ') out.pop_back();
            out += t.v;
        }
    }
    return out;
}

// ASCII -> fullwidth (Ｆｕｌｌ　ｗｉｄｔｈ), UTF-8 encoded.
inline std::string toFullWidth(char c) {
    unsigned int cp;
    if (c == ' ') {
        cp = 0x3000;
    } else if (c > 32 && c < 127) {
        cp = 0xFF01 + (static_cast<unsigned char>(c) - 33);
    } else {
        return std::string(1, c);
    }
    std::string out;
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
    return out;
}

inline std::string tidySpaces(std::string s) {
    std::string out;
    for (char c : s) {
        if (c == ' ' && (out.empty() || out.back() == ' ')) continue;
        out += c;
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// ---- 微軟新注音-style ` symbol menu & punctuation --------------------------

struct SymbolCat {
    const char *name;
    const char *syms;
};

// Same set as the web demo's SYMBOLS.
inline const std::vector<SymbolCat> &symbolCats() {
    static const std::vector<SymbolCat> cats = {
        {"常用", "，、。．？！；：…—～‧「」『』（）"},
        {"括號", "（）「」『』〔〕【】《》〈〉｛｝︵︶﹁﹂"},
        {"數學", "＋－×÷＝≠≒±√＜＞≦≧∞∩∪∫∵∴"},
        {"單位", "℃℉°′″＄％＠＃＆＊§￥"},
        {"箭號", "↑↓←→↖↗↙↘"},
        {"圖形", "○●△▲☆★◇◆□■▽▼◎⊙※"},
    };
    return cats;
}

// Punctuation, 微軟新注音/chewing conventions: Shift+,.…/;1 -> fullwidth
// marks, \ -> 、.
inline const std::unordered_map<char, const char *> &punctMap() {
    static const std::unordered_map<char, const char *> m = {
        {'<', "，"}, {'>', "。"}, {'?', "？"}, {'!', "！"},
        {':', "："}, {'"', "；"}, {'(', "（"}, {')', "）"},
        {'\\', "、"},
    };
    return m;
}

} // namespace slothing

#endif // _SLOTHING_COMMON_DISPLAY_H_
