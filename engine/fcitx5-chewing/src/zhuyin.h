// A dependency-free zhuyin (bopomofo) keyboard state machine — the piece that
// replaces libchewing's keystroke→syllable parsing on the way to dropping
// libchewing entirely (see MIGRATION.md).
//
// A bopomofo syllable has up to three slots filled left-to-right:
//   initial (聲母, ㄅ..ㄙ) · medial (介音, ㄧㄨㄩ) · final (韻母, ㄚ..ㄦ)
// followed by a tone key that commits the syllable. This models the Dachen
// (default) layout, the same one eval/harvest and prepare_data.py key on.
//
// The engine feeds ASCII keys in; ZhuyinBuffer accumulates the in-progress
// syllable and a list of committed syllables (each a bopomofo string with its
// tone mark, e.g. "ㄨㄛˇ") — exactly the form slothingd's decode mode wants.
// Header-only so it can be unit-tested offline with no fcitx5/libchewing.
#ifndef _FCITX5_CHEWING_ZHUYIN_H_
#define _FCITX5_CHEWING_ZHUYIN_H_

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace slothing {

enum class Slot { Initial, Medial, Final, None };

// Dachen key → (bopomofo symbol, slot). Tone keys map to Slot count via a
// separate table. Derived by inverting prepare_data.py's KEYMAP.
inline const std::unordered_map<char, std::pair<const char *, Slot>> &
dachenMap() {
    static const std::unordered_map<char, std::pair<const char *, Slot>> m = {
        {'1', {"ㄅ", Slot::Initial}}, {'q', {"ㄆ", Slot::Initial}},
        {'a', {"ㄇ", Slot::Initial}}, {'z', {"ㄈ", Slot::Initial}},
        {'2', {"ㄉ", Slot::Initial}}, {'w', {"ㄊ", Slot::Initial}},
        {'s', {"ㄋ", Slot::Initial}}, {'x', {"ㄌ", Slot::Initial}},
        {'e', {"ㄍ", Slot::Initial}}, {'d', {"ㄎ", Slot::Initial}},
        {'c', {"ㄏ", Slot::Initial}}, {'r', {"ㄐ", Slot::Initial}},
        {'f', {"ㄑ", Slot::Initial}}, {'v', {"ㄒ", Slot::Initial}},
        {'5', {"ㄓ", Slot::Initial}}, {'t', {"ㄔ", Slot::Initial}},
        {'g', {"ㄕ", Slot::Initial}}, {'b', {"ㄖ", Slot::Initial}},
        {'y', {"ㄗ", Slot::Initial}}, {'h', {"ㄘ", Slot::Initial}},
        {'n', {"ㄙ", Slot::Initial}},
        {'u', {"ㄧ", Slot::Medial}},  {'j', {"ㄨ", Slot::Medial}},
        {'m', {"ㄩ", Slot::Medial}},
        {'8', {"ㄚ", Slot::Final}},   {'i', {"ㄛ", Slot::Final}},
        {'k', {"ㄜ", Slot::Final}},   {',', {"ㄝ", Slot::Final}},
        {'9', {"ㄞ", Slot::Final}},   {'o', {"ㄟ", Slot::Final}},
        {'l', {"ㄠ", Slot::Final}},   {'.', {"ㄡ", Slot::Final}},
        {'0', {"ㄢ", Slot::Final}},   {'p', {"ㄣ", Slot::Final}},
        {';', {"ㄤ", Slot::Final}},   {'/', {"ㄥ", Slot::Final}},
        {'-', {"ㄦ", Slot::Final}},
    };
    return m;
}

// Tone keys → mark. Space is tone 1 (no mark, empty string).
inline const std::unordered_map<char, const char *> &toneMap() {
    static const std::unordered_map<char, const char *> m = {
        {' ', ""}, {'6', "ˊ"}, {'3', "ˇ"}, {'4', "ˋ"}, {'7', "˙"},
    };
    return m;
}

class ZhuyinBuffer {
public:
    // Feed one ASCII key. Returns true if the key was a zhuyin/tone key the
    // buffer consumed; false if it isn't part of zhuyin input (the engine then
    // handles it as punctuation/passthrough).
    bool key(char c) {
        auto &dm = dachenMap();
        auto it = dm.find(c);
        if (it != dm.end()) {
            setSlot(it->second.second, it->second.first);
            return true;
        }
        auto tt = toneMap().find(c);
        if (tt != toneMap().end()) {
            // A tone with no symbols pending is a no-op (a bare space is not
            // zhuyin input — let the engine pass it through).
            if (!hasPending()) {
                return c != ' ' ? true : false;
            }
            commit(tt->second);
            return true;
        }
        return false;
    }

    // Remove the last thing typed: an in-progress slot if any, else the last
    // committed syllable. Returns false if there was nothing to remove.
    bool backspace() {
        if (cur_[2].size()) { cur_[2].clear(); return true; }
        if (cur_[1].size()) { cur_[1].clear(); return true; }
        if (cur_[0].size()) { cur_[0].clear(); return true; }
        if (!committed_.empty()) { committed_.pop_back(); return true; }
        return false;
    }

    bool empty() const { return committed_.empty() && !hasPending(); }

    // Committed syllables (each with tone mark), for slothingd decode mode.
    const std::vector<std::string> &syllables() const { return committed_; }

    // Human-readable preedit: committed syllables joined, plus the pending
    // (un-toned) syllable being composed.
    std::string preedit() const {
        std::string out;
        for (const auto &s : committed_) {
            out += s;
        }
        out += pending();
        return out;
    }

    // The in-progress syllable without a tone (what the user is mid-typing).
    std::string pending() const {
        return cur_[0] + cur_[1] + cur_[2];
    }

    void clear() {
        committed_.clear();
        cur_ = {"", "", ""};
    }

private:
    bool hasPending() const {
        return cur_[0].size() || cur_[1].size() || cur_[2].size();
    }

    void setSlot(Slot s, const char *sym) {
        // Overwrite the slot (retyping an initial replaces it, matching
        // chewing). Filling an earlier slot after a later one is allowed; the
        // slots always render in initial·medial·final order.
        cur_[static_cast<int>(s)] = sym;
    }

    void commit(const char *tone) {
        std::string syl = cur_[0] + cur_[1] + cur_[2] + tone;
        committed_.push_back(std::move(syl));
        cur_ = {"", "", ""};
    }

    std::array<std::string, 3> cur_{"", "", ""}; // initial, medial, final
    std::vector<std::string> committed_;
};

} // namespace slothing

#endif // _FCITX5_CHEWING_ZHUYIN_H_
