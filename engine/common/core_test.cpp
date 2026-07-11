// Offline unit test for the shared IME state machine (core.h) — the
// chewing-parity contracts that used to live only inside eim.cpp. Runs with
// no daemon (daemon calls fail-fast; phrase lists come back empty, rescore
// keeps selections), no fcitx, no ibus.
//   g++ -std=c++17 -I . -I ../fcitx5-chewing/src core_test.cpp -o /tmp/ct && /tmp/ct
#include "assoc.h"
#include "core.h"
#include <cstdio>

using namespace slothing;

static int failures = 0;
#define CHECK(cond, desc)                                                      \
    do {                                                                       \
        const bool ok = (cond);                                                \
        if (!ok) failures++;                                                   \
        printf("%s  %s\n", ok ? "PASS" : "FAIL", desc);                        \
    } while (0)

// A tiny fixture: 我在寫程式 -> 5 zh segments with homophone alternatives.
static ChoosingCore fixture(int pendingFocus = -1) {
    ChoosingCore c;
    std::vector<std::vector<std::string>> pos = {
        {"我", "窩"}, {"在", "再", "載"}, {"寫", "血"},
        {"程", "成", "城"}, {"式", "是", "事"}};
    std::vector<std::pair<int, int>> ivals = {
        {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};
    std::vector<std::string> syls = {"ㄨㄛˇ", "ㄗㄞˋ", "ㄒㄧㄝˇ", "ㄔㄥˊ",
                                     "ㄕˋ"};
    std::vector<SegTok> toks;
    for (const auto &s : syls) toks.push_back({true, s});
    c.begin(pos, ivals, syls, toks, "我在寫程式", pendingFocus);
    return c;
}

int main() {
    // Isolate from a live slothingd: point the socket at nowhere so daemon
    // calls fail fast and phrase lists are deterministically empty.
    setenv("SLOTHINGD_SOCKET", "/nonexistent/slothingd.sock", 1);
    // ---- matchesPositions ----
    {
        std::vector<std::vector<std::string>> pos = {{"我", "窩"}, {"在"}};
        CHECK(matchesPositions("我在", pos), "matchesPositions accepts legal");
        CHECK(!matchesPositions("你在", pos), "matchesPositions rejects illegal");
        CHECK(!matchesPositions("我在了", pos), "matchesPositions rejects extra");
    }

    // ---- ComposingCore ----
    {
        ComposingCore comp;
        comp.insertToken({true, "ㄨㄛˇ"});
        comp.insertToken({false, "python"});
        CHECK(comp.toks.size() == 2 && comp.tokCursor == -1,
              "insertToken appends at end");
        comp.moveCursor(ComposingCore::Move::Home);
        comp.insertToken({true, "ㄋㄧˇ"});
        CHECK(comp.toks[0].v == "ㄋㄧˇ" && comp.tokCursor == 1,
              "insert at cursor advances cursor");
        comp.backspace();
        CHECK(comp.toks.size() == 2 && comp.toks[0].v == "ㄨㄛˇ" &&
                  comp.tokCursor == 0,
              "backspace removes token before cursor");
        comp.rawKeys = "ㄨ"; // multi-byte pending symbol
        comp.backspace();
        CHECK(comp.rawKeys.empty() && comp.toks.size() == 2,
              "backspace pops one UTF-8 char of raw run first");
        comp.moveCursor(ComposingCore::Move::End);
        CHECK(comp.tokCursor == -1, "End -> cursor -1 (end sentinel)");
    }

    // ---- staleDisplay (the no-flicker contract) ----
    {
        std::vector<SegTok> old = {{true, "a"}, {true, "b"}, {true, "c"}};
        std::vector<std::string> oldDisp = {"甲", "乙", "丙"};
        // mid-sentence edit: b replaced by x -> prefix a + suffix c preserved
        std::vector<SegTok> cur = {{true, "a"}, {true, "x"}, {true, "c"}};
        auto d = staleDisplay(cur, old, oldDisp);
        CHECK(d.size() == 3 && d[0] == "甲" && d[1].empty() && d[2] == "丙",
              "staleDisplay keeps unchanged prefix+suffix, blanks the edit");
        // appended token: prefix fully preserved
        std::vector<SegTok> app = {{true, "a"}, {true, "b"}, {true, "c"},
                                   {true, "d"}};
        d = staleDisplay(app, old, oldDisp);
        CHECK(d.size() == 4 && d[0] == "甲" && d[2] == "丙" && d[3].empty(),
              "staleDisplay: append keeps whole old prefix");
        CHECK(staleDisplay(old, old, oldDisp) == oldDisp,
              "staleDisplay: identical toks -> fresh display");
    }

    // ---- ChoosingCore seeding ----
    {
        auto c = fixture();
        CHECK(c.segSel == std::vector<int>({0, 0, 0, 0, 0}),
              "begin seeds selections from best sentence");
        CHECK(c.segFocus == 0 && c.candListOpen && c.phraseHl == -1,
              "begin: focus first ambiguous, window open, no phrase hl");
        auto c2 = fixture(3);
        CHECK(c2.segFocus == 3, "begin honors pendingFocus (↓ at cursor)");
    }

    // ---- composedSentence spacing ----
    {
        ChoosingCore c;
        c.positions = {{"我"}, {"python"}, {"好"}, {"，"}};
        c.segSel = {0, 0, 0, 0};
        c.toks = {{true, "ㄨㄛˇ"}, {false, "python"}, {true, "ㄏㄠˇ"},
                  {false, "，"}};
        CHECK(c.composedSentence() == "我 python 好，",
              "composedSentence: en spaced, fullwidth punct hugs");
    }

    // ---- pick semantics (chewing: pick closes the window) ----
    {
        auto c = fixture();
        c.pickSegment(1);
        CHECK(c.segSel[0] == 1 && !c.candListOpen && c.userFixed.count(0),
              "pickSegment selects, closes window, records userFixed");
        c.reopen();
        CHECK(c.candListOpen && c.chCursor == 1 && c.phraseHl == -1,
              "reopen puts char cursor on current selection");
    }
    {
        auto c = fixture();
        c.segFocus = 1;
        c.pickPhrase(1, "再寫");
        CHECK(c.segSel[1] == 1 && c.segSel[2] == 0 && !c.candListOpen &&
                  c.segFocus == 1 && c.userFixed.count(1) &&
                  c.userFixed.count(2),
              "pickPhrase sets both chars, closes window, focus on word");
    }

    // ---- combined highlight loop (no daemon -> 0 phrases, chars only) ----
    {
        auto c = fixture();
        c.segFocus = 1; // 3 candidates
        c.chCursor = c.segSel[1];
        c.moveHighlight(+1);
        CHECK(c.chCursor == 1 && c.phraseHl == -1, "→ moves char highlight");
        c.moveHighlight(+1);
        c.moveHighlight(+1);
        CHECK(c.chCursor == 0, "highlight loop wraps");
        c.moveHighlight(-1);
        CHECK(c.chCursor == 2, "← wraps backward");
        c.confirmHighlight();
        CHECK(c.segSel[1] == 2 && !c.candListOpen,
              "Enter confirms highlighted char and closes window");
    }

    // ---- focus movement skips unambiguous segments ----
    {
        auto c = fixture();
        c.positions[1] = {"在"}; // make segment 1 unambiguous
        c.segFocus = 0;
        c.moveFocus(+1);
        CHECK(c.segFocus == 2, "moveFocus skips single-candidate segments");
        c.moveFocus(-1);
        CHECK(c.segFocus == 0, "moveFocus back skips too");
    }

    // ---- learn payload ----
    {
        auto c = fixture();
        c.segSel[1] = 1;
        c.segSel[2] = 1;
        c.userFixed = {1, 2};
        json p = c.learnPayload();
        CHECK(p["chars"].size() == 2 && p["chars"][0][1] == "再",
              "learnPayload records changed chars");
        CHECK(p["phrases"].size() == 1 &&
                  p["phrases"][0][1] == "再血",
              "learnPayload records adjacent pair as phrase");
    }

    // ---- AssocEngine (聯想) ----
    {
        AssocEngine a;
        a.load("電\t腦 話 影\n注\t意 音\n", "");
        CHECK(a.predictions().empty(), "assoc: no tail -> no predictions");
        a.record("我要買電");
        auto p = a.predictions();
        CHECK(p.size() >= 3 && p[0] == "腦" && p[1] == "話",
              "assoc: dict completions after tail 電");
        a.record("腦");            // user picked 腦 -> tail chains + bigram 電->腦
        a.record("我要買電");      // tail 電 again
        p = a.predictions();
        CHECK(!p.empty() && p[0] == "腦",
              "assoc: personal bigram outranks dictionary order");
        a.record("好。");          // ends non-CJK
        CHECK(a.predictions().empty(), "assoc: punctuation clears the tail");
        a.record("注");
        p = a.predictions();
        CHECK(!p.empty() && p[0] == "意", "assoc: tail 注 -> dict 意 first");
        a.clearTail();
        CHECK(a.predictions().empty(), "assoc: clearTail drops predictions");
    }

    printf("\n%s\n", failures ? "FAILURES" : "all passed");
    return failures ? 1 : 0;
}
