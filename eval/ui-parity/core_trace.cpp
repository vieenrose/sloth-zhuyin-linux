// core_trace: feed a key sequence to the SHARED C++ core (the exact
// SlothSession state machine that fcitx5, IBus and Android all drive) and
// dump the observable UI state after every keystroke — same JSON schema as
// demo_trace.mjs and chewing_trace.c, so compare_traces.py / compare_core_web.py
// can diff the web reimplementation against the C++ core structurally.
//
// The three C++ frontends share engine/common/core.h by construction; this
// harness pins the one true reimplementation (space-static/ime.js) to them.
// A stub Decoder stands in for the neural model — parity is STRUCTURAL (token
// counts, pending run, candidate window, cursor, commit), never the chosen
// characters, so the model's ranking is irrelevant. The stub returns the
// phonetic table's top char per syllable so beginConvert() opens the window
// exactly as the real (WASM/ONNX) decoder does.
//
//   g++ -std=c++17 -I ../../android/app/cpp -I ../../engine/common \
//       core_trace.cpp -o /tmp/core_trace
//   /tmp/core_trace "su3cl3<D>1<E>" ../../model/phonetic_table.tsv
//
// Key DSL (identical to demo_trace.mjs): literal chars; <D>own <U>p <L>eft
// <R>ight <E>nter <ESC> <B>ackspace <H>ome e<N>d; space = ' '.
#include "session.h"
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace sloth;

// A model-free Decoder: returns the phonetic table's top char per syllable so
// the conversion path (beginConvert / commitLive) behaves structurally like the
// real decoder without any weights. Per-position candidate lists are left empty
// so the core falls back to the phonetic table order (same as a socket reply
// with no "candidates" field, and same as the web when the model omits them).
struct StubDecoder : Decoder {
    std::map<std::string, std::vector<std::string>> table;
    std::string best(const std::vector<std::string> &syls) {
        std::string s;
        for (const auto &syl : syls) {
            auto it = table.find(syl);
            if (it == table.end() || it->second.empty()) return {};
            s += it->second.front();
        }
        return s;
    }
    DecodeResult decode(const std::vector<std::string> &syls, int,
                        const std::string &) override {
        DecodeResult r;
        std::string s = best(syls);
        if (!s.empty()) r.sentences.push_back(s);
        return r;
    }
    DecodeResult decodeWithHints(const std::vector<std::string> &syls,
                                 const std::map<int, std::string> &) override {
        return decode(syls, 1, "");
    }
    std::vector<std::pair<double, std::string>>
    phrasesScored(const std::vector<std::string> &, int, int) override {
        return {};
    }
};

static std::string slurp(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// codepoints in a UTF-8 string (for the commit-length field)
static int cpCount(const std::string &s) {
    int n = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80) n++;
    return n;
}

struct Key {
    std::string label; // what we print
    char ch = 0;       // printable ascii, or 0 for a named key
    std::string named; // "D","U","L","R","E","ESC","B","H","N"
};

static std::vector<Key> parseKeys(const std::string &s) {
    std::vector<Key> out;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '<') {
            size_t e = s.find('>', i);
            std::string nm = s.substr(i + 1, e - i - 1);
            out.push_back({nm, 0, nm});
            i = e;
        } else {
            out.push_back({std::string(1, s[i]), s[i], ""});
        }
    }
    return out;
}

static bool isTone(char c) { return c == '3' || c == '4' || c == '6' || c == '7'; }

int main(int argc, char **argv) {
    // Isolate from any live slothd (the shared core never touches a socket
    // here — the injected StubDecoder is the only decode path).
    setenv("SLOTHD_SOCKET", "/nonexistent/slothd.sock", 1);

    std::string keystr = argc > 1 ? argv[1] : "";
    std::string tablePath = argc > 2 ? argv[2] : "../../model/phonetic_table.tsv";
    std::string assocPath = argc > 3 ? argv[3] : "";

    std::string table = slurp(tablePath);
    if (table.empty()) {
        fprintf(stderr, "core_trace: empty/missing phonetic table at %s\n",
                tablePath.c_str());
        return 2;
    }
    // parse the table for the stub (syllable \t char char char ...)
    auto stub = std::make_unique<StubDecoder>();
    {
        std::stringstream ss(table);
        std::string line;
        while (std::getline(ss, line)) {
            size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string syl = line.substr(0, tab);
            // chars are concatenated UTF-8 (same file loadPhoneticTable reads);
            // split into individual codepoints with the shared splitUtf8.
            std::vector<std::string> chars = splitUtf8(line.substr(tab + 1));
            if (!chars.empty()) stub->table[syl] = chars;
        }
    }

    std::string assoc = assocPath.empty() ? std::string() : slurp(assocPath);
    SlothSession sess(std::move(stub), table, assoc);

    auto keys = parseKeys(keystr);
    bool enMode = false; // lone-Shift forced-English toggle (<S>)
    for (const auto &k : keys) {
        if (k.named == "S") {
            enMode = !enMode;
            sess.setEnglishMode(enMode);
        } else if (k.ch) {
            // printable ascii — mirror SlothImeService.onKey():
            // space / tone finalize the run via toneOrSpace, everything else
            // (bopomofo, punctuation, digit, latin) via feedKey. In the
            // candidate window a digit picks; Enter/Esc handled below.
            auto cand = sess.getCandidates();
            if (cand.open && k.ch >= '1' && k.ch <= '9') {
                sess.pickSegment(k.ch - '1'); // chewing: pick closes the window
            } else if (!enMode && (k.ch == ' ' || isTone(k.ch))) {
                sess.toneOrSpace(static_cast<uint32_t>(k.ch));
            } else {
                // English mode: EVERY key (space/tone included) goes through
                // feedKey's passthrough branch — matches SlothImeService.
                sess.feedKey(static_cast<uint32_t>(k.ch));
            }
        } else if (k.named == "B") {
            sess.backspace();
        } else if (k.named == "L" || k.named == "R" || k.named == "H" ||
                   k.named == "N") {
            auto cand = sess.getCandidates();
            int dir = k.named == "L" ? -1 : k.named == "R" ? 1
                      : k.named == "H"                     ? -2
                                                           : 2;
            if (cand.open && (dir == -1 || dir == 1))
                sess.moveHighlight(dir); // web deviation: ←→ move highlight
            else
                sess.moveCursor(dir);
        } else if (k.named == "D" || k.named == "U") {
            auto cand = sess.getCandidates();
            if (cand.open)
                sess.moveHighlight(k.named == "D" ? 1 : -1);
            else if (k.named == "D")
                sess.beginConvert(-1, /*commitDirect=*/false); // ↓ opens window
        } else if (k.named == "E") {
            // Enter: in Choosing confirm/commit; in Composing commit the live
            // conversion, else convert-and-commit directly (新注音).
            if (!sess.confirmChoosing()) {
                if (!sess.commitLive())
                    sess.beginConvert(-1, /*commitDirect=*/true);
            }
        } else if (k.named == "ESC") {
            sess.escapeChoosing(); // no-op outside Choosing
        }

        // refresh the fast (model-free) live display so an all-English Enter can
        // commitLive(); zh/bopo/cand/cursor don't depend on the heavy decode.
        sess.refreshLiveFast();

        int commit = cpCount(sess.getCommit());
        auto u = sess.traceState();
        printf("{\"key\":\"%s\",\"zh\":%d,\"bopo\":%d,\"cand\":%d,"
               "\"cursor\":%d,\"commit\":%d}\n",
               k.label.c_str(), u.zh, u.bopo, u.cand, u.cursor, commit);
    }
    return 0;
}
