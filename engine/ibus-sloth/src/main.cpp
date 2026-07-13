// Sloth IME IBus engine — the IBus adapter over the shared frontend-free IME
// core (engine/common/core.h), the same state machine the fcitx5 addon uses,
// so the chewing-parity behavior is identical: live 微軟新注音-style
// conversion, auto zh/en code-switch, ↓ candidate window with 詞 chips
// (⇧1-9 / ←→+⏎ combined highlight loop), pick-closes-window, hint-based
// re-scoring, one-Enter commit, lone-Shift English passthrough, ` symbol
// menu, Shift+Space fullwidth. No libchewing.
//
// This file only: decodes IBus key events into core calls, runs the async
// decode worker (std::thread + g_idle_add back to the GLib loop), and paints
// (preedit text, IBusLookupTable, auxiliary text).
#include "assoc.h"
#include "core.h"
#include "display.h"
#include "segment.h"
#include "zhuyin.h"
#include <atomic>
#include <cstring>
#include <fstream>
#include <functional>
#include <ibus.h>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace sloth;

// ---------------------------------------------------------------------------

#define IBUS_TYPE_SLOTHING_ENGINE (ibus_sloth_engine_get_type())

typedef struct _IBusSlothEngine IBusSlothEngine;
typedef struct _IBusSlothEngineClass IBusSlothEngineClass;

struct SlothImpl; // C++ state, owned by the GObject

struct _IBusSlothEngine {
    IBusEngine parent;
    SlothImpl *impl;
};

struct _IBusSlothEngineClass {
    IBusEngineClass parent;
};

GType ibus_sloth_engine_get_type();

// ---------------------------------------------------------------------------

static constexpr int kPageSize = 10;

struct SlothImpl {
    IBusSlothEngine *obj = nullptr;

    // Shared frontend-free state machine.
    ComposingCore comp;
    ChoosingCore choosing;

    enum class ConvertState { Composing, Converting, Choosing };
    ConvertState state = ConvertState::Composing;

    bool enMode = false;
    bool fullWidth = false;
    bool shiftAlone = false;
    bool symbolMode = false;
    int symCat = 0;
    int pendingFocus = -1;

    std::unordered_map<std::string, std::vector<std::string>> phoneticTable;
    std::unique_ptr<Segmenter> segmenter;

    // Decode staging (handed to choosing.begin() when the reply arrives).
    std::string convertBuffer;
    std::vector<std::vector<std::string>> convertPositions;
    std::vector<std::pair<int, int>> convertIntervals;
    std::vector<std::string> convertSyllables;
    std::vector<SegTok> convertToks;

    // Live (modeless) conversion state.
    std::string livePreedit;
    std::vector<std::string> liveDisp;
    std::vector<SegTok> liveToks;
    uint64_t liveGeneration = 0;
    uint64_t convertGeneration = 0;

    std::string notice;

    // 聯想 next-word predictions (shared AssocEngine; 微軟新注音-style).
    AssocEngine assoc;
    bool predicting = false;
    int predictChain = 0;

    std::thread worker;
    std::atomic<bool> workerStop{false};
    std::atomic<int> inflightFd{-1};

    ~SlothImpl() { stopWorker(); }

    void stopWorker() {
        workerStop.store(true);
        int fd = inflightFd.exchange(-1);
        if (fd >= 0) {
            shutdown(fd, SHUT_RDWR);
        }
        if (worker.joinable()) {
            worker.join();
        }
        workerStop.store(false);
    }

    // Schedule a closure onto the GLib main loop (worker -> UI thread).
    void schedule(std::function<void()> fn) {
        auto *heap = new std::function<void()>(std::move(fn));
        g_object_ref(obj);
        struct Ctx {
            IBusSlothEngine *obj;
            std::function<void()> *fn;
        };
        auto *ctx = new Ctx{obj, heap};
        g_idle_add(
            [](gpointer data) -> gboolean {
                auto *c = static_cast<Ctx *>(data);
                (*c->fn)();
                delete c->fn;
                g_object_unref(c->obj);
                delete c;
                return G_SOURCE_REMOVE;
            },
            ctx);
    }

    void loadPhoneticTable() {
        std::string path;
        if (const char *env = std::getenv("SLOTHING_PHONETIC_TABLE")) {
            path = env;
        } else {
            // XDG data dirs, same file the fcitx addon installs.
            const char *xdg = std::getenv("XDG_DATA_HOME");
            std::vector<std::string> dirs;
            dirs.push_back((xdg ? std::string(xdg)
                                : std::string(g_get_home_dir()) +
                                      "/.local/share") +
                           "/sloth/phonetic_table.tsv");
            dirs.push_back("/usr/local/share/sloth/phonetic_table.tsv");
            dirs.push_back("/usr/share/sloth/phonetic_table.tsv");
            for (const auto &d : dirs) {
                if (std::ifstream(d).good()) {
                    path = d;
                    break;
                }
            }
        }
        if (path.empty()) {
            return;
        }
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            auto tab = line.find('\t');
            if (tab == std::string::npos) {
                continue;
            }
            std::string syl = line.substr(0, tab);
            std::vector<std::string> chars = splitUtf8(line.substr(tab + 1));
            if (!chars.empty()) {
                phoneticTable.emplace(std::move(syl), std::move(chars));
            }
        }
        std::set<std::string> validBase;
        for (const auto &[syl, chars] : phoneticTable) {
            std::string base;
            for (const auto &ch : splitUtf8(syl)) {
                if (ch != "ˊ" && ch != "ˇ" && ch != "ˋ" && ch != "˙") {
                    base += ch;
                }
            }
            if (!base.empty()) {
                validBase.insert(std::move(base));
            }
        }
        segmenter = std::make_unique<Segmenter>(std::move(validBase));
    }

    void loadAssoc() {
        std::string dictTsv;
        std::string path;
        if (const char *env = std::getenv("SLOTHING_ASSOC_TABLE")) {
            path = env;
        } else {
            const char *xdg = std::getenv("XDG_DATA_HOME");
            std::string home = xdg ? std::string(xdg)
                                   : std::string(g_get_home_dir()) +
                                         "/.local/share";
            std::vector<std::string> dirs = {
                home + "/sloth/assoc_tc.tsv",
                "/usr/local/share/sloth/assoc_tc.tsv",
                "/usr/share/sloth/assoc_tc.tsv",
            };
            for (const auto &d : dirs) {
                if (std::ifstream(d).good()) {
                    path = d;
                    break;
                }
            }
        }
        if (!path.empty()) {
            std::ifstream f(path);
            dictTsv.assign(std::istreambuf_iterator<char>(f),
                           std::istreambuf_iterator<char>());
        }
        const char *xdg = std::getenv("XDG_DATA_HOME");
        std::string dir = (xdg ? std::string(xdg)
                               : std::string(g_get_home_dir()) +
                                     "/.local/share") +
                          "/sloth";
        g_mkdir_with_parents(dir.c_str(), 0700);
        assoc.load(dictTsv, dir + "/assoc_user.tsv");
    }

    // ---- painting ---------------------------------------------------------

    IBusEngine *engine() { return reinterpret_cast<IBusEngine *>(obj); }

    void clearPanels() {
        ibus_engine_hide_lookup_table(engine());
        ibus_engine_hide_auxiliary_text(engine());
    }

    void setPreedit(const std::string &text, size_t cursorBytes,
                    int hlFromByte = -1, int hlToByte = -1) {
        if (text.empty()) {
            // Blank the preedit explicitly (empty text, invisible, CLEAR mode)
            // THEN hide. A bare hide is not enough for terminal/CLI clients:
            // they don't clear on hide, and under the old COMMIT mode they
            // committed the last char instead of dropping it — so backspacing
            // the final character re-landed it in the buffer. GUI/Electron
            // clients cleared fine, which is why it was CLI-only.
            IBusText *empty = ibus_text_new_from_string("");
            ibus_engine_update_preedit_text_with_mode(
                engine(), empty, 0, FALSE, IBUS_ENGINE_PREEDIT_CLEAR);
            ibus_engine_hide_preedit_text(engine());
            return;
        }
        IBusText *t = ibus_text_new_from_string(text.c_str());
        const guint len = g_utf8_strlen(text.c_str(), -1);
        ibus_text_append_attribute(t, IBUS_ATTR_TYPE_UNDERLINE,
                                   IBUS_ATTR_UNDERLINE_SINGLE, 0, len);
        if (hlFromByte >= 0 && hlToByte > hlFromByte) {
            const guint from =
                g_utf8_strlen(text.c_str(), hlFromByte);
            const guint to = g_utf8_strlen(text.c_str(), hlToByte);
            // focused segment: reverse video (background highlight)
            ibus_text_append_attribute(t, IBUS_ATTR_TYPE_BACKGROUND, 0x3584e4,
                                       from, to);
            ibus_text_append_attribute(t, IBUS_ATTR_TYPE_FOREGROUND, 0xffffff,
                                       from, to);
        }
        const guint cursor = g_utf8_strlen(
            text.substr(0, std::min(cursorBytes, text.size())).c_str(), -1);
        // CLEAR, not COMMIT: COMMIT tells the client to commit the preedit
        // whenever it ends (hide / focus-out), so backspacing the last char
        // would land it in the document instead of clearing it (both en+zh).
        // Every real commit path here is explicit (acceptConversion,
        // flushPending, passthrough, symbols), so the preedit must be pure
        // display that vanishes when cleared.
        ibus_engine_update_preedit_text_with_mode(
            engine(), t, cursor, TRUE, IBUS_ENGINE_PREEDIT_CLEAR);
    }

    void setAux(const std::string &text) {
        if (text.empty()) {
            ibus_engine_hide_auxiliary_text(engine());
            return;
        }
        ibus_engine_update_auxiliary_text(
            engine(), ibus_text_new_from_string(text.c_str()), TRUE);
    }

    void commit(const std::string &s) {
        if (!s.empty()) {
            ibus_engine_commit_text(engine(),
                                    ibus_text_new_from_string(s.c_str()));
            assoc.record(s); // feed 聯想 (bigrams + prediction tail)
        }
    }

    // 聯想 strip: after a commit, aux shows 聯: ⇧1 腦 ⇧2 子… — ⇧1-9 picks
    // (digits stay typeable, 微軟 convention), any other key dismisses.
    void renderPredictions() {
        auto preds = assoc.predictions();
        if (preds.empty() || state != ConvertState::Composing ||
            !comp.empty()) {
            predicting = false;
            return;
        }
        std::string aux = "聯:";
        for (size_t i = 0; i < preds.size() && i < 9; i++) {
            aux += " ⇧" + std::to_string(i + 1) + " " + preds[i];
        }
        ibus_engine_hide_lookup_table(engine());
        setAux(aux);
        predicting = true;
    }

    void renderComposing() {
        std::string tail;
        if (!comp.rawKeys.empty()) {
            tail = enMode ? comp.rawKeys
                          : (segmenter ? tidySpaces(toksDisplay(
                                             segmenter->segment(comp.rawKeys)))
                                       : comp.rawKeys);
        }
        std::vector<std::string> disp =
            staleDisplay(comp.toks, liveToks, liveDisp);
        JoinResult jr = joinDisplay(comp.toks, disp, comp.tokCursor, tail);
        setPreedit(jr.text, jr.cursorBytes);
        clearPanels();
        setAux(notice);
    }

    void renderConverting() {
        setPreedit(tidySpaces(toksDisplay(comp.toks)), 0);
        clearPanels();
        setAux("轉換中…");
    }

    void renderSegments() {
        // preedit: selected char per segment, focused one highlighted
        std::string pre;
        int hlFrom = -1, hlTo = -1;
        for (size_t i = 0; i < choosing.positions.size(); i++) {
            const std::string &seg =
                choosing.positions[i][choosing.segSel[i]];
            if (static_cast<int>(i) == choosing.segFocus) {
                hlFrom = static_cast<int>(pre.size());
                hlTo = static_cast<int>(pre.size() + seg.size());
            }
            pre += seg;
        }
        setPreedit(pre, hlFrom >= 0 ? hlFrom : pre.size(), hlFrom, hlTo);

        if (!choosing.candListOpen) {
            ibus_engine_hide_lookup_table(engine());
            setAux("←→ 移動　↓ 選字　⏎ 上字　Esc 取消");
            return;
        }

        IBusLookupTable *table =
            ibus_lookup_table_new(kPageSize, 0, TRUE, FALSE);
        const auto &cands = choosing.positions[choosing.segFocus];
        for (const auto &c : cands) {
            ibus_lookup_table_append_candidate(
                table, ibus_text_new_from_string(c.c_str()));
        }
        ibus_lookup_table_set_cursor_pos(
            table, static_cast<guint>(choosing.chCursor));
        ibus_lookup_table_set_orientation(table, IBUS_ORIENTATION_HORIZONTAL);
        ibus_engine_update_lookup_table(engine(), table, TRUE);

        // 詞 chips ride the auxiliary text (⇧1-9 / ←→+⏎, highlight marked)
        const auto &phrases = choosing.ensurePhrases();
        std::string aux;
        if (!phrases.empty()) {
            aux = "詞:";
            for (size_t j = 0; j < phrases.size() && j < 9; j++) {
                const bool hl =
                    static_cast<int>(j) == choosing.phraseHl;
                aux += std::string(" ") + (hl ? "【" : "") + "⇧" +
                       std::to_string(j + 1) + " " + phrases[j].second +
                       (hl ? "】" : "");
            }
            aux += "　";
        }
        aux += "1-9 選字　⇧1-9 選詞　←→ 移動　⏎ 確認　Esc 取消";
        setAux(aux);
    }

    void renderSymbols() {
        const auto &cats = symbolCats();
        std::string aux;
        for (size_t i = 0; i < cats.size(); i++) {
            if (static_cast<int>(i) == symCat) {
                aux += "【" + std::string(cats[i].name) + "】";
            } else {
                aux += " " + std::string(cats[i].name) + " ";
            }
        }
        aux += "　←→ 分類　1-9 選取　Esc/` 關閉";
        IBusLookupTable *table =
            ibus_lookup_table_new(kPageSize, 0, FALSE, TRUE);
        for (auto &sym : splitUtf8(cats[symCat].syms)) {
            ibus_lookup_table_append_candidate(
                table, ibus_text_new_from_string(sym.c_str()));
        }
        ibus_lookup_table_set_orientation(table, IBUS_ORIENTATION_HORIZONTAL);
        ibus_engine_update_lookup_table(engine(), table, TRUE);
        setAux(aux);
    }

    void render() {
        switch (state) {
        case ConvertState::Choosing: renderSegments(); break;
        case ConvertState::Converting: renderConverting(); break;
        case ConvertState::Composing: renderComposing(); break;
        }
    }

    // ---- decode orchestration (mirrors the fcitx adapter) -----------------

    std::string surroundingContext() {
        IBusText *text = nullptr;
        guint cursor = 0, anchor = 0;
        ibus_engine_get_surrounding_text(engine(), &text, &cursor, &anchor);
        std::string ctx;
        if (text) {
            const gchar *s = ibus_text_get_text(text);
            if (s) {
                const gchar *end = g_utf8_offset_to_pointer(s, cursor);
                ctx.assign(s, end - s);
            }
        }
        return ctx;
    }

    void clearLive() {
        livePreedit.clear();
        liveDisp.clear();
        liveToks.clear();
    }

    void scheduleLiveDecode() {
        if (comp.toks.empty() || phoneticTable.empty() || !segmenter) {
            clearLive();
            return;
        }
        bool anyZh = false;
        for (const auto &t : comp.toks) {
            anyZh |= t.zh;
        }
        if (!anyZh) {
            liveDisp.clear();
            for (const auto &t : comp.toks) {
                liveDisp.push_back(t.v);
            }
            liveToks = comp.toks;
            livePreedit =
                joinDisplay(comp.toks, liveDisp, -1, std::string()).text;
            return;
        }
        std::string ctx = surroundingContext();
        stopWorker();
        const uint64_t generation = ++liveGeneration;
        worker = std::thread([this, generation, ctx = std::move(ctx),
                              toks = comp.toks]() {
            std::vector<std::string> disp;
            std::string runCtx = ctx;
            size_t i = 0;
            bool allOk = true;
            while (i < toks.size() && !workerStop.load()) {
                if (!toks[i].zh) {
                    disp.push_back(toks[i].v);
                    i++;
                    continue;
                }
                std::vector<std::string> run;
                const size_t start = i;
                while (i < toks.size() && toks[i].zh) {
                    run.push_back(toks[i].v);
                    i++;
                }
                DaemonError err = DaemonError::None;
                auto sentences = queryDecoder(run, 1, runCtx, inflightFd, err);
                if (!sentences.empty() &&
                    utf8Length(sentences[0]) == run.size()) {
                    const std::string &sent = sentences[0];
                    runCtx += sent;
                    for (size_t k = 0, off = 0; k < run.size(); k++) {
                        size_t len = utf8SeqLen(sent[off]);
                        disp.push_back(sent.substr(off, len));
                        off += len;
                    }
                } else {
                    allOk = false;
                    for (size_t k = start; k < start + run.size(); k++) {
                        disp.push_back(toks[k].v);
                    }
                }
            }
            if (workerStop.load()) {
                return;
            }
            schedule([this, generation, toks = std::move(toks),
                      disp = std::move(disp), allOk]() {
                if (generation != liveGeneration ||
                    state != ConvertState::Composing) {
                    return;
                }
                if (allOk) {
                    liveDisp = disp;
                    liveToks = toks;
                    livePreedit =
                        joinDisplay(toks, disp, -1, std::string()).text;
                }
                renderComposing();
            });
        });
    }

    void acceptConversion(const std::string &sentence) {
        commit(sentence);
        state = ConvertState::Composing;
        convertGeneration++;
        liveGeneration++;
        clearLive();
        comp.clear();
        choosing.clear();
        pendingFocus = -1;
        convertBuffer.clear();
        convertPositions.clear();
        convertIntervals.clear();
        convertSyllables.clear();
        convertToks.clear();
        ibus_engine_hide_preedit_text(engine());
        clearPanels();
        predictChain = 0;
        renderPredictions(); // 聯想 strip flips on after the commit
    }

    void cancelConversion(std::string newNotice = {}) {
        stopWorker();
        notice = std::move(newNotice);
        state = ConvertState::Composing;
        convertGeneration++;
        choosing.clear();
        convertPositions.clear();
        convertIntervals.clear();
        convertSyllables.clear();
        convertToks.clear();
        render();
    }

    void startDecode(bool commitDirect = false) {
        comp.commitRun(segmenter.get(), enMode);
        if (comp.empty() || phoneticTable.empty()) {
            notice = phoneticTable.empty() ? "無音表" : "";
            render();
            return;
        }
        std::vector<std::vector<std::string>> positions;
        std::vector<std::pair<int, int>> intervals;
        std::vector<std::string> syllables;
        int at = 0;
        for (const auto &t : comp.toks) {
            int span = 1;
            if (t.zh) {
                auto it = phoneticTable.find(t.v);
                if (it != phoneticTable.end() && !it->second.empty()) {
                    positions.push_back(it->second);
                } else {
                    positions.push_back({t.v}); // unknown reading: literal
                    span = static_cast<int>(utf8Length(t.v));
                }
                syllables.push_back(t.v);
            } else {
                positions.push_back({t.v});
                span = static_cast<int>(utf8Length(t.v));
                syllables.push_back("");
            }
            intervals.emplace_back(at, at + span);
            at += span;
        }
        std::string context = surroundingContext();

        stopWorker();
        state = ConvertState::Converting;
        convertBuffer.clear();
        convertPositions = positions;
        convertIntervals = intervals;
        convertSyllables = syllables;
        convertToks = comp.toks;
        const uint64_t generation = ++convertGeneration;
        const int n = 5;
        notice.clear();
        render();

        const bool pureZh =
            std::all_of(comp.toks.begin(), comp.toks.end(),
                        [this](const SegTok &t) {
                            return t.zh && phoneticTable.count(t.v);
                        });

        worker = std::thread([this, generation, n, pureZh, commitDirect,
                              toks = comp.toks,
                              positions = std::move(positions),
                              context = std::move(context)]() {
            DaemonError err = DaemonError::None;
            std::vector<std::string> verified;
            std::vector<std::vector<std::string>> ranked;
            if (pureZh) {
                std::vector<std::string> syls;
                for (const auto &t : toks) {
                    syls.push_back(t.v);
                }
                json full;
                for (auto &sentence :
                     queryDecoder(syls, n, context, inflightFd, err, &full)) {
                    if (!matchesPositions(sentence, positions)) {
                        continue;
                    }
                    if (std::find(verified.begin(), verified.end(),
                                  sentence) == verified.end()) {
                        verified.push_back(std::move(sentence));
                    }
                }
                if (full.contains("candidates")) {
                    try {
                        auto r =
                            full["candidates"]
                                .get<std::vector<std::vector<std::string>>>();
                        if (r.size() == positions.size()) {
                            bool ok = true;
                            for (size_t i = 0; i < r.size(); i++) {
                                if (r[i].size() != positions[i].size()) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (ok) {
                                ranked = std::move(r);
                            }
                        }
                    } catch (const std::exception &) {
                    }
                }
            } else {
                // mixed zh/en: decode each zh run, keep en literals — but
                // build a per-token display and join it with joinDisplay so
                // English runs get the same spacing as the preedit (a raw
                // `sentence += tok.v` dropped the spaces the user typed
                // between words: "web app" -> "webapp").
                std::vector<std::string> disp;
                bool ok = true;
                size_t i = 0;
                while (i < toks.size() && !workerStop.load()) {
                    if (!toks[i].zh) {
                        disp.push_back(toks[i].v);
                        i++;
                        continue;
                    }
                    std::vector<std::string> run;
                    while (i < toks.size() && toks[i].zh) {
                        run.push_back(toks[i].v);
                        i++;
                    }
                    auto sentences =
                        queryDecoder(run, 1, "", inflightFd, err);
                    if (!sentences.empty() &&
                        utf8Length(sentences[0]) == run.size()) {
                        const std::string &sent = sentences[0];
                        for (size_t k = 0, off = 0; k < run.size(); k++) {
                            size_t len = utf8SeqLen(sent[off]);
                            disp.push_back(sent.substr(off, len));
                            off += len;
                        }
                    } else {
                        ok = false;
                        break;
                    }
                }
                if (ok && !disp.empty()) {
                    std::string sentence =
                        joinDisplay(toks, disp, -1, std::string()).text;
                    if (!sentence.empty()) {
                        verified.push_back(std::move(sentence));
                    }
                }
            }
            if (workerStop.load()) {
                return;
            }
            std::string failNotice;
            if (verified.empty()) {
                switch (err) {
                case DaemonError::Connect:
                    failNotice = "slothd 未執行";
                    break;
                case DaemonError::Io:
                    failNotice = "slothd 無回應";
                    break;
                default:
                    failNotice = "無法解碼";
                    break;
                }
            }
            schedule([this, generation, commitDirect,
                      verified = std::move(verified),
                      ranked = std::move(ranked),
                      failNotice = std::move(failNotice)]() {
                if (generation != convertGeneration ||
                    state != ConvertState::Converting) {
                    return;
                }
                if (verified.empty()) {
                    cancelConversion(failNotice);
                    return;
                }
                if (!ranked.empty()) {
                    convertPositions = ranked;
                }
                convertBuffer = verified.front();
                if (commitDirect) { // Enter: one keypress commits (新注音)
                    acceptConversion(verified.front());
                    return;
                }
                state = ConvertState::Choosing;
                choosing.begin(convertPositions, convertIntervals,
                               convertSyllables, convertToks,
                               verified.front(), pendingFocus);
                pendingFocus = -1;
                renderSegments();
            });
        });
    }

    // ---- key handling (mirrors the fcitx adapter contract-for-contract) ---

    bool processKey(guint keyval, guint modifiers) {
        const bool isRelease = modifiers & IBUS_RELEASE_MASK;
        if (isRelease) {
            if ((keyval == IBUS_KEY_Shift_L || keyval == IBUS_KEY_Shift_R) &&
                shiftAlone) {
                shiftAlone = false;
                enMode = !enMode;
                notice = enMode ? "英文模式（Shift 切回）" : "";
                if (state == ConvertState::Composing) {
                    renderComposing();
                }
                return true;
            }
            return false;
        }
        if (keyval == IBUS_KEY_Shift_L || keyval == IBUS_KEY_Shift_R) {
            shiftAlone = true;
            return false;
        }
        shiftAlone = false;

        // real modifier chords go to the app
        if (modifiers & (IBUS_CONTROL_MASK | IBUS_MOD1_MASK |
                         IBUS_SUPER_MASK)) {
            return false;
        }
        const bool shift = modifiers & IBUS_SHIFT_MASK;
        const bool printable = keyval >= 0x20 && keyval < 0x7f;
        const char c = printable ? static_cast<char>(keyval) : 0;

        // 聯想 predictions showing (post-commit, empty buffer): ⇧1-9 picks
        // (digits stay typeable — 微軟 convention); any other key dismisses
        // the strip and is then processed normally.
        if (predicting && state == ConvertState::Composing) {
            if (shift && printable) {
                static const char *shifted = "!@#$%^&*(";
                auto preds = assoc.predictions();
                for (int i = 0; i < 9; i++) {
                    if (shifted[i] == c && i < static_cast<int>(preds.size())) {
                        commit(preds[i]); // records + advances the tail
                        if (++predictChain < 5) {
                            renderPredictions();
                        } else {
                            predicting = false;
                            predictChain = 0;
                            clearPanels();
                        }
                        return true;
                    }
                }
            }
            predicting = false;
            clearPanels();
            // fall through: the key is processed normally
        }

        // -- Choosing --------------------------------------------------------
        if (state == ConvertState::Choosing) {
            if (choosing.segFocus < 0 || choosing.empty()) {
                cancelConversion();
                return true;
            }
            if (keyval == IBUS_KEY_Escape) {
                if (choosing.candListOpen) { // close the window first
                    choosing.candListOpen = false;
                    renderSegments();
                    return true;
                }
                cancelConversion();
                return true;
            }
            if (keyval == IBUS_KEY_Return || keyval == IBUS_KEY_KP_Enter) {
                if (choosing.candListOpen) {
                    choosing.confirmHighlight();
                    renderSegments();
                    return true;
                }
                json payload = choosing.learnPayload();
                if (!payload["chars"].empty() ||
                    !payload["phrases"].empty()) {
                    sendLearn(payload);
                }
                acceptConversion(choosing.composedSentence());
                return true;
            }
            if (keyval == IBUS_KEY_Right || keyval == IBUS_KEY_Left) {
                const int d = keyval == IBUS_KEY_Right ? 1 : -1;
                if (choosing.candListOpen) {
                    choosing.moveHighlight(d);
                } else {
                    choosing.moveFocus(d);
                }
                renderSegments();
                return true;
            }
            if (keyval == IBUS_KEY_Down || keyval == IBUS_KEY_Up ||
                keyval == IBUS_KEY_space || keyval == IBUS_KEY_Page_Down ||
                keyval == IBUS_KEY_Page_Up) {
                if (!choosing.candListOpen) {
                    choosing.reopen();
                } else {
                    // page the char list; the core cursor pages with it
                    const int ncand = static_cast<int>(
                        choosing.positions[choosing.segFocus].size());
                    const int d = (keyval == IBUS_KEY_Up ||
                                   keyval == IBUS_KEY_Page_Up)
                                      ? -kPageSize
                                      : kPageSize;
                    int cur = choosing.chCursor + d;
                    if (cur >= 0 && cur < ncand + kPageSize - 1) {
                        choosing.chCursor =
                            std::max(0, std::min(cur, ncand - 1));
                        choosing.phraseHl = -1;
                    }
                }
                renderSegments();
                return true;
            }
            if (shift && c >= '!' && c <= '~') {
                // ⇧1-9 picks a 詞 (word) option from the aux row
                static const char *shifted = "!@#$%^&*(";
                const auto &ph = choosing.ensurePhrases();
                for (int i = 0; i < 9; i++) {
                    if (shifted[i] == c &&
                        i < static_cast<int>(ph.size())) {
                        choosing.pickPhrase(ph[i].first, ph[i].second);
                        renderSegments();
                        return true;
                    }
                }
                return true;
            }
            if (c >= '0' && c <= '9') {
                // number keys select from the VISIBLE page
                const int idx = (c == '0' ? 9 : c - '1') +
                                (choosing.chCursor / kPageSize) * kPageSize;
                if (idx < static_cast<int>(
                              choosing.positions[choosing.segFocus].size())) {
                    choosing.pickSegment(idx);
                }
                renderSegments();
                return true;
            }
            return true; // swallow everything else while Choosing
        }

        // -- Converting ------------------------------------------------------
        if (state == ConvertState::Converting) {
            if (keyval == IBUS_KEY_Escape) {
                cancelConversion();
            }
            return true; // swallow while decoding
        }

        // -- Symbol menu -----------------------------------------------------
        if (symbolMode) {
            if (keyval == IBUS_KEY_Escape || c == '`') {
                symbolMode = false;
                render();
                return true;
            }
            const int ncat = static_cast<int>(symbolCats().size());
            if (keyval == IBUS_KEY_Left) {
                symCat = (symCat - 1 + ncat) % ncat;
                renderSymbols();
                return true;
            }
            if (keyval == IBUS_KEY_Right) {
                symCat = (symCat + 1) % ncat;
                renderSymbols();
                return true;
            }
            if (c >= '0' && c <= '9') {
                const int idx = c == '0' ? 9 : c - '1';
                auto syms = splitUtf8(symbolCats()[symCat].syms);
                if (idx < static_cast<int>(syms.size())) {
                    pickSymbol(syms[idx]);
                }
                return true;
            }
            return true;
        }

        // -- Composing -------------------------------------------------------
        notice.clear();

        if (keyval == IBUS_KEY_Left || keyval == IBUS_KEY_Right ||
            keyval == IBUS_KEY_Home || keyval == IBUS_KEY_End) {
            if (comp.empty()) {
                return false; // nothing composed: let the app have the key
            }
            if (!comp.rawKeys.empty()) {
                return true; // chewing: arrows ignored mid-syllable
            }
            using Move = ComposingCore::Move;
            comp.moveCursor(keyval == IBUS_KEY_Left    ? Move::Left
                            : keyval == IBUS_KEY_Right ? Move::Right
                            : keyval == IBUS_KEY_Home  ? Move::Home
                                                       : Move::End);
            scheduleLiveDecode();
            renderComposing();
            return true;
        }

        if (keyval == IBUS_KEY_Down) {
            if (!comp.empty()) {
                comp.commitRun(segmenter.get(), enMode);
                const int n = static_cast<int>(comp.toks.size());
                pendingFocus =
                    comp.tokCursor < 0
                        ? n - 1
                        : std::max(0, std::min(comp.tokCursor, n - 1));
                startDecode();
                return true;
            }
            return false;
        }

        if (keyval == IBUS_KEY_Escape) {
            // chewing: Esc clears only the pending bopomofo
            if (!comp.rawKeys.empty()) {
                comp.rawKeys.clear();
                renderComposing();
                return true;
            }
            return !comp.empty(); // swallow, keep the sentence
        }

        if (keyval == IBUS_KEY_BackSpace) {
            if (comp.empty()) {
                return false;
            }
            const bool hadRaw = !comp.rawKeys.empty();
            comp.backspace();
            if (!hadRaw) {
                scheduleLiveDecode();
            }
            renderComposing();
            return true;
        }

        if (keyval == IBUS_KEY_Return || keyval == IBUS_KEY_KP_Enter) {
            if (comp.empty()) {
                return false;
            }
            comp.commitRun(segmenter.get(), enMode);
            if (!livePreedit.empty() && liveToks == comp.toks) {
                acceptConversion(livePreedit);
                return true;
            }
            startDecode(/*commitDirect=*/true);
            return true;
        }

        if (!printable) {
            return !comp.empty(); // swallow noise while composing
        }

        // Shift+Space: 全形/半形
        if (c == ' ' && shift) {
            fullWidth = !fullWidth;
            notice = fullWidth ? "全形" : "半形";
            renderComposing();
            return true;
        }

        // Forced-English mode (lone Shift): PASSTHROUGH — keys go straight
        // to the app (flush composed text first so ordering is preserved).
        if (enMode) {
            if (c == ' ' || (c >= 33 && c < 127)) {
                if (!comp.empty()) {
                    comp.commitRun(segmenter.get(), enMode);
                    commit((!livePreedit.empty() && liveToks == comp.toks)
                               ? livePreedit
                               : tidySpaces(toksDisplay(comp.toks)));
                    comp.clear();
                    clearLive();
                    renderComposing();
                }
                commit(fullWidth ? toFullWidth(c) : std::string(1, c));
            }
            return true;
        }

        // Space finalizes the current run (tone 1 handled by the segmenter);
        // a space between two English words is kept as a literal (faithful).
        if (c == ' ') {
            if (!comp.rawKeys.empty()) {
                comp.commitRunKeepSpace(segmenter.get(), enMode);
                scheduleLiveDecode();
                renderComposing();
                return true;
            }
            return !comp.toks.empty(); // swallow: don't type into app
        }

        // A tone key completes the last syllable -> finalize + live decode.
        if (segToneMark(c) && !comp.rawKeys.empty()) {
            comp.rawKeys += c;
            comp.commitRun(segmenter.get(), enMode);
            scheduleLiveDecode();
            renderComposing();
            return true;
        }

        // ` opens the categorized symbol menu
        if (c == '`') {
            symbolMode = true;
            renderSymbols();
            return true;
        }

        // Punctuation, 微軟新注音/chewing conventions.
        if (auto pit = punctMap().find(c); pit != punctMap().end()) {
            if (comp.empty()) {
                commit(pit->second);
                return true;
            }
            comp.commitRun(segmenter.get(), enMode);
            const int ins = comp.tokCursor < 0
                                ? static_cast<int>(comp.toks.size())
                                : comp.tokCursor;
            comp.insertToken({false, sloth::punctMark(c, pit->second,
                                                         comp.toks, ins)});
            scheduleLiveDecode();
            renderComposing();
            return true;
        }

        // Any zhuyin-mappable or alphanumeric key extends the raw run; the
        // segmenter re-decides zh/en live (auto code-switch, no mode key).
        const bool feeds = dachenMap().count(c) ||
                           (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '\'';
        if (feeds) {
            comp.rawKeys += c;
            renderComposing();
            return true;
        }

        return !comp.empty(); // swallow printable noise while composing
    }

    void pickSymbol(const std::string &sym) {
        symbolMode = false;
        if (comp.empty() && state == ConvertState::Composing) {
            commit(sym);
            ibus_engine_hide_preedit_text(engine());
            clearPanels();
            return;
        }
        comp.commitRun(segmenter.get(), enMode);
        comp.insertToken({false, sym});
        scheduleLiveDecode();
        renderComposing();
    }

    // chewing behavior: losing focus COMMITS the pending text rather than
    // dropping it (no data loss on stray clicks).
    void flushPending() {
        std::string pending;
        if (state == ConvertState::Choosing) {
            pending = choosing.composedSentence();
        } else if (!comp.empty()) {
            comp.commitRun(segmenter.get(), enMode);
            pending = (!livePreedit.empty() && liveToks == comp.toks)
                          ? livePreedit
                          : tidySpaces(toksDisplay(comp.toks));
        }
        if (!pending.empty()) {
            commit(pending);
        }
        stopWorker();
        state = ConvertState::Composing;
        convertGeneration++;
        liveGeneration++;
        comp.clear();
        choosing.clear();
        clearLive();
        notice.clear();
        pendingFocus = -1;
        symbolMode = false;
        assoc.clearTail(); // stale predictions must not cross focus changes
        predicting = false;
        ibus_engine_hide_preedit_text(engine());
        clearPanels();
    }

    void candidateClicked(guint index) {
        if (state == ConvertState::Choosing && choosing.candListOpen) {
            const int idx =
                (choosing.chCursor / kPageSize) * kPageSize +
                static_cast<int>(index);
            if (idx < static_cast<int>(
                          choosing.positions[choosing.segFocus].size())) {
                choosing.pickSegment(idx);
                renderSegments();
            }
        } else if (symbolMode) {
            auto syms = splitUtf8(symbolCats()[symCat].syms);
            if (index < syms.size()) {
                pickSymbol(syms[index]);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// GObject plumbing

G_DEFINE_TYPE(IBusSlothEngine, ibus_sloth_engine, IBUS_TYPE_ENGINE)

static gboolean ibus_sloth_engine_process_key_event(IBusEngine *engine,
                                                       guint keyval,
                                                       guint keycode,
                                                       guint modifiers) {
    (void)keycode;
    auto *self = reinterpret_cast<IBusSlothEngine *>(engine);
    return self->impl->processKey(keyval, modifiers) ? TRUE : FALSE;
}

static void ibus_sloth_engine_focus_out(IBusEngine *engine) {
    auto *self = reinterpret_cast<IBusSlothEngine *>(engine);
    self->impl->flushPending();
    IBUS_ENGINE_CLASS(ibus_sloth_engine_parent_class)->focus_out(engine);
}

static void ibus_sloth_engine_reset(IBusEngine *engine) {
    auto *self = reinterpret_cast<IBusSlothEngine *>(engine);
    self->impl->flushPending();
    IBUS_ENGINE_CLASS(ibus_sloth_engine_parent_class)->reset(engine);
}

static void ibus_sloth_engine_disable(IBusEngine *engine) {
    auto *self = reinterpret_cast<IBusSlothEngine *>(engine);
    self->impl->flushPending();
    IBUS_ENGINE_CLASS(ibus_sloth_engine_parent_class)->disable(engine);
}

static void ibus_sloth_engine_candidate_clicked(IBusEngine *engine,
                                                   guint index, guint button,
                                                   guint modifiers) {
    (void)button;
    (void)modifiers;
    auto *self = reinterpret_cast<IBusSlothEngine *>(engine);
    self->impl->candidateClicked(index);
}

static void ibus_sloth_engine_init(IBusSlothEngine *self) {
    self->impl = new SlothImpl();
    self->impl->obj = self;
    self->impl->loadPhoneticTable();
    self->impl->loadAssoc();
}

static void ibus_sloth_engine_destroy(IBusObject *obj) {
    auto *self = reinterpret_cast<IBusSlothEngine *>(obj);
    delete self->impl;
    self->impl = nullptr;
    IBUS_OBJECT_CLASS(ibus_sloth_engine_parent_class)->destroy(obj);
}

static void ibus_sloth_engine_class_init(IBusSlothEngineClass *klass) {
    IBusEngineClass *engine_class = IBUS_ENGINE_CLASS(klass);
    engine_class->process_key_event = ibus_sloth_engine_process_key_event;
    engine_class->focus_out = ibus_sloth_engine_focus_out;
    engine_class->reset = ibus_sloth_engine_reset;
    engine_class->disable = ibus_sloth_engine_disable;
    engine_class->candidate_clicked = ibus_sloth_engine_candidate_clicked;
    IBUS_OBJECT_CLASS(klass)->destroy = ibus_sloth_engine_destroy;
}

// ---------------------------------------------------------------------------

static IBusBus *bus = nullptr;

static void ibus_disconnected_cb(IBusBus *, gpointer) { ibus_quit(); }

int main(int argc, char **argv) {
    bool byIbus = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ibus") == 0) {
            byIbus = true;
        }
    }

    ibus_init();
    bus = ibus_bus_new();
    if (!ibus_bus_is_connected(bus)) {
        g_printerr("ibus-engine-sloth: cannot connect to the IBus daemon "
                   "(is ibus running?)\n");
        return 1;
    }
    g_signal_connect(bus, "disconnected", G_CALLBACK(ibus_disconnected_cb),
                     nullptr);

    IBusFactory *factory = ibus_factory_new(ibus_bus_get_connection(bus));
    ibus_factory_add_engine(factory, "sloth", IBUS_TYPE_SLOTHING_ENGINE);

    if (byIbus) {
        ibus_bus_request_name(bus, "org.freedesktop.IBus.Sloth", 0);
    } else {
        // standalone/dev: register the component directly
        IBusComponent *component = ibus_component_new(
            "org.freedesktop.IBus.Sloth", "Sloth IME (LLM zhuyin)", "0.1",
            "LGPL-2.1-or-later", "sloth-zhuyin-linux",
            "https://github.com/vieenrose/sloth-zhuyin-linux", "", "");
        ibus_component_add_engine(
            component,
            ibus_engine_desc_new("sloth", "Sloth IME 注音",
                                 "Sloth IME: LLM-powered zhuyin IME "
                                 "(libchewing-free)",
                                 "zh_TW", "LGPL-2.1-or-later",
                                 "sloth-zhuyin-linux",
                                 "ibus-sloth", "us")); // icon name (hicolor)
        ibus_bus_register_component(bus, component);
    }

    ibus_main();
    return 0;
}
