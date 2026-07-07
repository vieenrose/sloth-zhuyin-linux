/*
 * SPDX-FileCopyrightText: 2012~2012 Tai-Lin Chu <tailinchu@gmail.com>
 * SPDX-FileCopyrightText: 2012~2017 CSSlayer <wengxt@gmail.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 */
#include "eim.h"
#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fcitx-utils/utf8.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/statusarea.h>
#include <fcitx/text.h>
#include <fcitx/userinterfacemanager.h>
#include <optional>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

FCITX_DEFINE_LOG_CATEGORY(chewing_log, "chewing");
#define CHEWING_DEBUG() FCITX_LOGC(chewing_log, Debug)

namespace fcitx {

namespace {

constexpr int CHEWING_MAX_LEN = 18;

std::string builtin_keymaps[] = {
    "KB_DEFAULT",     "KB_HSU",          "KB_IBM",    "KB_GIN_YEIH",
    "KB_ET",          "KB_ET26",         "KB_DVORAK", "KB_DVORAK_HSU",
    "KB_DACHEN_CP26", "KB_HANYU_PINYIN", "KB_CARPALX"};

static const char *builtin_selectkeys[] = {
    "1234567890", "asdfghjkl;", "asdfzxcv89", "asdfjkl789",
    "aoeuhtn789", "1234qweras", "dstnaeo789",
};

static_assert(FCITX_ARRAY_SIZE(builtin_selectkeys) ==
                  ChewingSelectionKeyI18NAnnotation::enumLength,
              "Enum mismatch");

class ChewingCandidateWord : public CandidateWord {
public:
    ChewingCandidateWord(ChewingEngine *engine, std::string str, int index)
        : CandidateWord(Text(str)), engine_(engine), index_(index) {}

    void select(InputContext *inputContext) const override {
        auto pageSize = engine_->instance()->globalConfig().defaultPageSize();
        auto ctx = engine_->context();
        int page = index_ / pageSize + chewing_cand_CurrentPage(ctx);
        int off = index_ % pageSize;
        if (page < 0 || page >= chewing_cand_TotalPage(ctx))
            return;
        int lastPage = chewing_cand_CurrentPage(ctx);
        while (page != chewing_cand_CurrentPage(ctx)) {
            if (page < chewing_cand_CurrentPage(ctx)) {
                chewing_handle_Left(ctx);
            }
            if (page > chewing_cand_CurrentPage(ctx)) {
                chewing_handle_Right(ctx);
            }
            /* though useless, but take care if there is a bug cause freeze */
            if (lastPage == chewing_cand_CurrentPage(ctx)) {
                break;
            }
            lastPage = chewing_cand_CurrentPage(ctx);
        }
        chewing_handle_Default(ctx, builtin_selectkeys[static_cast<int>(
                                        *engine_->config().SelectionKey)][off]);

        if (chewing_keystroke_CheckAbsorb(ctx)) {
            engine_->updateUI(inputContext);
        } else if (chewing_keystroke_CheckIgnore(ctx)) {
            return;
        } else if (chewing_commit_Check(ctx)) {
            UniqueCPtr<char, chewing_free> str(chewing_commit_String(ctx));
            inputContext->commitString(str.get());
            engine_->updateUI(inputContext);
        } else {
            engine_->updateUI(inputContext);
        }
    }

private:
    ChewingEngine *engine_;
    int index_;
};

class ChewingCandidateList : public CandidateList,
                             public PageableCandidateList {
public:
    ChewingCandidateList(ChewingEngine *engine, InputContext *ic)
        : engine_(engine), ic_(ic) {
        auto ctx = engine_->context();
        setPageable(this);
        int index = 0;
        // get candidate word
        int pageSize = chewing_get_candPerPage(ctx);
        chewing_cand_Enumerate(ctx);
        while (chewing_cand_hasNext(ctx) && index < pageSize) {
            UniqueCPtr<char, chewing_free> str(chewing_cand_String(ctx));
            candidateWords_.emplace_back(std::make_unique<ChewingCandidateWord>(
                engine_, str.get(), index));
            if (index < 10) {
                const char label[] = {
                    builtin_selectkeys[static_cast<int>(
                        *engine_->config().SelectionKey)][index],
                    '.', '\0'};
                labels_.emplace_back(label);
            } else {
                labels_.emplace_back();
            }
            index++;
        }

        hasPrev_ = chewing_cand_CurrentPage(ctx) > 0;
        hasNext_ =
            chewing_cand_CurrentPage(ctx) + 1 < chewing_cand_TotalPage(ctx);
    }

    const Text &label(int idx) const override {
        if (idx < 0 || idx >= size()) {
            throw std::invalid_argument("Invalid index");
        }
        return labels_[idx];
    }
    const CandidateWord &candidate(int idx) const override {
        if (idx < 0 || idx >= size()) {
            throw std::invalid_argument("Invalid index");
        }
        return *candidateWords_[idx];
    }

    int size() const override { return candidateWords_.size(); }
    int cursorIndex() const override { return -1; }
    CandidateLayoutHint layoutHint() const override {
        return *engine_->config().CandidateLayout;
    }

    // Need for paging
    bool hasPrev() const override { return hasPrev_; }
    bool hasNext() const override { return hasNext_; }
    void prev() override { paging(true); }
    void next() override { paging(false); }

    bool usedNextBefore() const override { return true; }

private:
    void paging(bool prev) {
        if (candidateWords_.empty()) {
            return;
        }

        auto ctx = engine_->context();
        if (prev) {
            chewing_handle_Left(ctx);
        } else {
            chewing_handle_Right(ctx);
        }

        if (chewing_keystroke_CheckAbsorb(ctx)) {
            engine_->updateUI(ic_);
        }
    }

    bool hasPrev_ = false;
    bool hasNext_ = false;
    ChewingEngine *engine_;
    InputContext *ic_;
    std::vector<std::unique_ptr<ChewingCandidateWord>> candidateWords_;
    std::vector<Text> labels_;
};

void logger(void *, int, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// Reranking only runs on explicit commit (Enter / focus-out), a much less
// frequent and more latency-tolerant event than per-keystroke decoding, so
// this can afford to be generous. With LFM2.5-230M at Q4_0 (see
// MODEL_BENCHMARKS.md), observed latency is ~200-350ms on an idle machine;
// this timeout leaves headroom for heavier system load rather than tightly
// chasing the typical case. Note this is still a *blocking* wait inside
// keyEvent() -- see README for the known follow-up to make this
// asynchronous instead.
constexpr int kSlothingdTimeoutMs = 2000;

std::string slothingdSocketPath() {
    if (const char *env = std::getenv("SLOTHINGD_SOCKET")) {
        return env;
    }
    return "/tmp/slothingd.sock";
}

// Walks the already-decided phrase segmentation of the current buffer via
// libchewing's interval API and, for each segment, non-invasively opens its
// candidate list (chewing_cand_open/close, not a keystroke) to harvest the
// alternatives libchewing itself considers for that segment. Cursor position
// and the phrase-choice-rearward mode are saved and restored, so this has no
// visible side effect on editing state. Returns an empty vector if the
// buffer is empty or any segment yields no usable candidates (abort rather
// than send a partial/malformed request).
//
// Candidates whose length differs from the interval's span are discarded:
// chewing_cand_open can surface shorter-than-segment candidates too, and a
// first live test showed what happens if those leak through -- the reranker
// rebuilt the sentence from fragments ('我再重新考慮' became '我爞考').
// maybeRerank() additionally refuses to run unless the original sentence
// reconstructs from what we harvested, so even a subtly wrong harvest
// degrades to a no-op instead of corrupting the user's text.
std::vector<std::vector<std::string>> collectCandidatePositions(ChewingContext *ctx) {
    std::vector<std::vector<std::string>> positions;
    if (chewing_buffer_Len(ctx) <= 0) {
        return positions;
    }

    int origCursor = chewing_cursor_Current(ctx);
    // Harvest with forward phrase choice regardless of the user's
    // ChoiceBackward setting, so cand_open at a segment's start yields that
    // segment's candidates rather than the one ending at the cursor.
    int origRearward = chewing_get_phraseChoiceRearward(ctx);
    chewing_set_phraseChoiceRearward(ctx, 0);

    chewing_handle_Home(ctx);
    std::vector<IntervalType> intervals;
    chewing_interval_Enumerate(ctx);
    while (chewing_interval_hasNext(ctx)) {
        IntervalType iv;
        chewing_interval_Get(ctx, &iv);
        intervals.push_back(iv);
    }

    for (const auto &interval : intervals) {
        // Re-derive the cursor from libchewing each round instead of
        // assuming chewing_cand_open/close left it where we put it.
        while (chewing_cursor_Current(ctx) < interval.from) {
            chewing_handle_Right(ctx);
        }
        const size_t span = static_cast<size_t>(interval.to - interval.from);
        chewing_cand_open(ctx);
        chewing_cand_Enumerate(ctx);
        std::vector<std::string> cands;
        while (chewing_cand_hasNext(ctx)) {
            UniqueCPtr<char, chewing_free> str(chewing_cand_String(ctx));
            if (str && utf8::lengthValidated(std::string_view(str.get())) == span) {
                cands.emplace_back(str.get());
            }
        }
        chewing_cand_close(ctx);
        CHEWING_DEBUG() << "harvest interval [" << interval.from << ","
                        << interval.to << ") -> " << cands.size()
                        << " candidates";
        if (cands.empty()) {
            positions.clear();
            break;
        }
        positions.push_back(std::move(cands));
    }

    chewing_set_phraseChoiceRearward(ctx, origRearward);
    chewing_handle_Home(ctx);
    for (int i = 0; i < origCursor; i++) {
        chewing_handle_Right(ctx);
    }

    return positions;
}

std::string jsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        default:
            out += static_cast<char>(c);
        }
    }
    return out;
}

std::string buildRerankRequest(const std::vector<std::vector<std::string>> &positions) {
    std::string req = "{\"positions\":[";
    for (size_t i = 0; i < positions.size(); i++) {
        if (i) {
            req += ",";
        }
        req += "[";
        for (size_t j = 0; j < positions[i].size(); j++) {
            if (j) {
                req += ",";
            }
            req += "\"" + jsonEscape(positions[i][j]) + "\"";
        }
        req += "]";
    }
    req += "],\"n\":3}\n";
    return req;
}

// Minimal extractor for slothingd's own fixed {"sentences":["...",...]}
// response shape -- not a general JSON parser, and not meant to be one:
// the protocol between eim.cpp and slothingd is closed and controlled by
// us on both ends.
std::vector<std::string> extractStringArray(const std::string &json, const char *key) {
    std::vector<std::string> out;
    std::string needle = std::string("\"") + key + "\":[";
    auto pos = json.find(needle);
    if (pos == std::string::npos) {
        return out;
    }
    pos += needle.size();
    while (pos < json.size() && json[pos] != ']') {
        if (json[pos] != '"') {
            pos++;
            continue;
        }
        pos++; // opening quote
        std::string item;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
            }
            item += json[pos];
            pos++;
        }
        pos++; // closing quote
        out.push_back(std::move(item));
    }
    return out;
}

// Talks to slothingd over its Unix socket with a send/recv timeout so a
// slow or absent daemon can never wedge the worker thread for long. Returns
// an empty list on any failure (daemon not running, timeout, malformed
// response) -- suggestions simply don't appear.
std::vector<std::string> queryReranker(const std::vector<std::vector<std::string>> &positions) {
    if (positions.empty()) {
        return {};
    }

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return {};
    }
    struct timeval tv;
    tv.tv_sec = kSlothingdTimeoutMs / 1000;
    tv.tv_usec = (kSlothingdTimeoutMs % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::string sockPath = slothingdSocketPath();
    strncpy(addr.sun_path, sockPath.c_str(), sizeof(addr.sun_path) - 1);
    if (connect(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return {};
    }

    std::string req = buildRerankRequest(positions);
    size_t off = 0;
    while (off < req.size()) {
        ssize_t n = write(fd, req.data() + off, req.size() - off);
        if (n <= 0) {
            close(fd);
            return {};
        }
        off += static_cast<size_t>(n);
    }
    shutdown(fd, SHUT_WR);

    std::string resp;
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        resp.append(buf, static_cast<size_t>(n));
    }
    close(fd);
    if (resp.empty()) {
        return {};
    }
    return extractStringArray(resp, "sentences");
}

// Verifies `sentence` is exactly the concatenation of one candidate from
// each entry of `positions`, in order -- i.e. that it could only have come
// from real libchewing alternatives, never arbitrary LLM output. Prefers
// the longest matching candidate at each step to avoid a short candidate
// falsely matching a prefix of a longer one.
bool matchesPositions(const std::string &sentence,
                      const std::vector<std::vector<std::string>> &positions) {
    size_t off = 0;
    for (const auto &cands : positions) {
        std::vector<const std::string *> sorted;
        sorted.reserve(cands.size());
        for (const auto &c : cands) {
            sorted.push_back(&c);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const std::string *a, const std::string *b) { return a->size() > b->size(); });

        bool matched = false;
        for (const auto *c : sorted) {
            if (sentence.compare(off, c->size(), *c) == 0) {
                off += c->size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            return false;
        }
    }
    return off == sentence.size();
}

} // namespace

ChewingEngine::ChewingEngine(Instance *instance)
    : instance_(instance), context_(chewing_new()) {
    chewing_set_maxChiSymbolLen(context_.get(), CHEWING_MAX_LEN);
    if (chewing_log().checkLogLevel(Debug)) {
        chewing_set_logger(context_.get(), logger, nullptr);
    }
    dispatcher_.attach(&instance_->eventLoop());
    reloadConfig();
}

ChewingEngine::~ChewingEngine() = default;

void ChewingEngine::dropSuggestion() {
    suggestions_.clear();
    suggestionIndex_ = 0;
    suggestionForBuffer_.clear();
    inflightForBuffer_.clear();
    suggestionGeneration_++;
}

void ChewingEngine::requestSuggestion(InputContext *ic) {
    ChewingContext *ctx = context_.get();

    // Only suggest for a settled multi-char buffer: nothing mid-syllable
    // (zuin pending) and nothing while the candidate window is open.
    // (cand_TotalChoice > 0 is the open-window signal; cand_CheckDone is
    // false even during plain typing, so it can't be used for this.)
    const char *zuin = chewing_bopomofo_String_static(ctx);
    if (chewing_buffer_Len(ctx) < 2 || (zuin && zuin[0]) ||
        chewing_cand_TotalChoice(ctx) > 0) {
        return;
    }

    UniqueCPtr<char, chewing_free> buf(chewing_buffer_String(ctx));
    std::string buffer = buf.get();
    if (buffer == suggestionForBuffer_ || buffer == inflightForBuffer_) {
        return; // already answered / already being asked
    }

    auto positions = collectCandidatePositions(ctx);
    // If chewing's own sentence can't be rebuilt from the harvest, the
    // harvest is unreliable (wrong segments, missing coverage) -- asking
    // the LLM against it would produce nonsense suggestions.
    if (positions.empty() || !matchesPositions(buffer, positions)) {
        return;
    }

    inflightForBuffer_ = buffer;
    const uint64_t generation = ++suggestionGeneration_;
    auto icRef = ic->watch();

    // The worker only touches its own copies plus the Unix socket; all
    // engine state is re-entered through dispatcher_ on the main thread.
    // The engine outlives any worker for practical purposes (the addon is
    // only unloaded when fcitx5 itself exits).
    std::thread([this, icRef, generation, buffer = std::move(buffer),
                 positions = std::move(positions)]() {
        std::vector<std::string> verified;
        for (auto &sentence : queryReranker(positions)) {
            // only sentences provably built from real chewing candidates,
            // and only ones that actually differ from what's on screen
            if (sentence != buffer && matchesPositions(sentence, positions) &&
                std::find(verified.begin(), verified.end(), sentence) ==
                    verified.end()) {
                verified.push_back(std::move(sentence));
            }
        }
        if (verified.empty()) {
            // Unblock retries for this buffer (e.g. daemon was briefly
            // down); without this, a failed request would permanently
            // suppress suggestions for this exact sentence.
            dispatcher_.schedule([this, generation]() {
                if (generation == suggestionGeneration_) {
                    inflightForBuffer_.clear();
                }
            });
            return;
        }
        dispatcher_.schedule(
            [this, icRef, generation, buffer, verified = std::move(verified)]() {
                if (generation != suggestionGeneration_) {
                    return; // user kept typing; stale answer
                }
                inflightForBuffer_.clear();
                suggestions_ = verified;
                suggestionIndex_ = 0;
                suggestionForBuffer_ = buffer;
                CHEWING_DEBUG() << "slothingd suggests " << suggestions_.size()
                                << " alternative(s) for '" << buffer
                                << "', first: '" << suggestions_[0] << "'";
                if (auto *ic = icRef.get()) {
                    updateUI(ic);
                }
            });
    }).detach();
}

void ChewingEngine::reloadConfig() {
    readAsIni(config_, "conf/chewing.conf");
    populateConfig();
}

void ChewingEngine::populateConfig() {
    ChewingContext *ctx = context_.get();

    chewing_set_KBType(
        ctx, chewing_KBStr2Num(
                 builtin_keymaps[static_cast<int>(*config_.Layout)].data()));

    chewing_set_ChiEngMode(ctx, CHINESE_MODE);

    int selkey[10];
    int i = 0;
    for (i = 0; i < 10; i++) {
        selkey[i] =
            builtin_selectkeys[static_cast<int>(*config_.SelectionKey)][i];
    }

    chewing_set_selKey(ctx, selkey, 10);
    chewing_set_candPerPage(ctx, *config_.PageSize);
    chewing_set_addPhraseDirection(ctx, *config_.AddPhraseForward ? 0 : 1);
    chewing_set_phraseChoiceRearward(ctx, *config_.ChoiceBackward ? 1 : 0);
    chewing_set_autoShiftCur(ctx, *config_.AutoShiftCursor ? 1 : 0);
    chewing_set_spaceAsSelection(ctx, *config_.SpaceAsSelection ? 1 : 0);
    chewing_set_escCleanAllBuf(ctx, 1);
}

void ChewingEngine::reset(const InputMethodEntry &, InputContextEvent &event) {
    doReset(event);
}

void ChewingEngine::doReset(InputContextEvent &event) {
    ChewingContext *ctx = context_.get();
    chewing_handle_Esc(ctx);
    dropSuggestion();
    updateUI(event.inputContext());
}

void ChewingEngine::save() {}

void ChewingEngine::activate(const InputMethodEntry &,
                             InputContextEvent &event) {
    // Request chttrans.
    // Fullwidth is not required for chewing.
    chttrans();
    auto *inputContext = event.inputContext();
    if (auto *action =
            instance_->userInterfaceManager().lookupAction("chttrans")) {
        inputContext->statusArea().addAction(StatusGroup::InputMethod, action);
    }
}

void ChewingEngine::deactivate(const InputMethodEntry &entry,
                               InputContextEvent &event) {
    if (event.type() == EventType::InputContextFocusOut ||
        event.type() == EventType::InputContextSwitchInputMethod) {
        flushBuffer(event);
    } else {
        reset(entry, event);
    }
}

void ChewingEngine::keyEvent(const InputMethodEntry &entry,
                             KeyEvent &keyEvent) {
    auto ctx = context_.get();
    if (keyEvent.isRelease()) {
        return;
    }

    // LLM suggestion interactions -- only when suggestions are being shown
    // for exactly the buffer on screen (each was already verified via
    // matchesPositions to be built from real chewing candidates).
    if (!suggestions_.empty()) {
        UniqueCPtr<char, chewing_free> buf(chewing_buffer_String(ctx));
        const char *zuin = chewing_bopomofo_String_static(ctx);
        bool suggestionsLive = suggestionForBuffer_ == buf.get() &&
                               (!zuin || !zuin[0]) &&
                               chewing_cand_TotalChoice(ctx) == 0;

        // Ctrl+Enter commits the highlighted suggestion.
        if (suggestionsLive &&
            keyEvent.key().check(FcitxKey_Return, KeyState::Ctrl)) {
            keyEvent.filterAndAccept();
            keyEvent.inputContext()->commitString(
                suggestions_[suggestionIndex_]);
            chewing_clean_preedit_buf(ctx);
            chewing_clean_bopomofo_buf(ctx);
            dropSuggestion();
            return updateUI(keyEvent.inputContext());
        }

        // Down cycles through the alternatives, but only while the cursor
        // sits at the end of the buffer (its resting position right after
        // typing) and there is more than one to cycle through. Move the
        // cursor anywhere else and Down reverts to chewing's per-phrase
        // candidate window, so the classic correction flow stays intact
        // (Space also opens it when SpaceAsSelection is on).
        if (suggestionsLive && suggestions_.size() > 1 &&
            keyEvent.key().check(FcitxKey_Down) &&
            chewing_cursor_Current(ctx) == chewing_buffer_Len(ctx)) {
            keyEvent.filterAndAccept();
            suggestionIndex_ = (suggestionIndex_ + 1) % suggestions_.size();
            return updateUI(keyEvent.inputContext());
        }
    }

    chewing_set_easySymbolInput(ctx, 0);
    CHEWING_DEBUG() << "KeyEvent: " << keyEvent.key().toString();
    auto ic = keyEvent.inputContext();
    const KeyList keypadKeys{Key{FcitxKey_KP_1}, Key{FcitxKey_KP_2},
                             Key{FcitxKey_KP_3}, Key{FcitxKey_KP_4},
                             Key{FcitxKey_KP_5}, Key{FcitxKey_KP_6},
                             Key{FcitxKey_KP_7}, Key{FcitxKey_KP_8},
                             Key{FcitxKey_KP_9}, Key{FcitxKey_KP_0}};
    if (*config_.UseKeypadAsSelectionKey && ic->inputPanel().candidateList()) {
        if (int index = keyEvent.key().keyListIndex(keypadKeys);
            index >= 0 && index < ic->inputPanel().candidateList()->size()) {
            ic->inputPanel().candidateList()->candidate(index).select(ic);
            return keyEvent.filterAndAccept();
        }
    }

    if (keyEvent.key().check(FcitxKey_space)) {
        chewing_handle_Space(ctx);
    } else if (keyEvent.key().check(FcitxKey_Tab)) {
        chewing_handle_Tab(ctx);
    } else if (keyEvent.key().isSimple()) {
        if (keyEvent.rawKey().states().test(KeyState::Shift)) {
            chewing_set_easySymbolInput(ctx, *config_.EasySymbolInput ? 1 : 0);
        }
        int scan_code = keyEvent.key().sym() & 0xff;
        if (*config_.Layout == ChewingLayout::HanYuPinYin) {
            const char *zuin_str = chewing_bopomofo_String_static(ctx);
            // Workaround a bug in libchewing fixed in 2017 but never has stable
            // release.
            if (std::string_view(zuin_str).size() >= 9) {
                return keyEvent.filterAndAccept();
            }
        }
        chewing_handle_Default(ctx, scan_code);
        chewing_set_easySymbolInput(ctx, 0);
    } else if (keyEvent.key().check(FcitxKey_BackSpace)) {
        const char *zuin_str = chewing_bopomofo_String_static(ctx);
        if (chewing_buffer_Len(ctx) == 0 && !zuin_str[0]) {
            return;
        }
        chewing_handle_Backspace(ctx);
        if (chewing_buffer_Len(ctx) == 0 && !zuin_str[0]) {
            keyEvent.filterAndAccept();
            return reset(entry, keyEvent);
        }
    } else if (keyEvent.key().check(FcitxKey_Escape)) {
        chewing_handle_Esc(ctx);
    } else if (keyEvent.key().check(FcitxKey_Delete)) {
        const char *zuin_str = chewing_bopomofo_String_static(ctx);
        if (chewing_buffer_Len(ctx) == 0 && !zuin_str[0]) {
            return;
        }
        chewing_handle_Del(ctx);
        if (chewing_buffer_Len(ctx) == 0 && !zuin_str[0]) {
            keyEvent.filterAndAccept();
            return reset(entry, keyEvent);
        }
    } else if (keyEvent.key().check(FcitxKey_Up)) {
        chewing_handle_Up(ctx);
    } else if (keyEvent.key().check(FcitxKey_Down)) {
        chewing_handle_Down(ctx);
    } else if (keyEvent.key().check(FcitxKey_Page_Down)) {
        chewing_handle_PageDown(ctx);
    } else if (keyEvent.key().check(FcitxKey_Page_Up)) {
        chewing_handle_PageUp(ctx);
    } else if (keyEvent.key().check(FcitxKey_Right)) {
        chewing_handle_Right(ctx);
    } else if (keyEvent.key().check(FcitxKey_Left)) {
        chewing_handle_Left(ctx);
    } else if (keyEvent.key().check(FcitxKey_Home)) {
        chewing_handle_Home(ctx);
    } else if (keyEvent.key().check(FcitxKey_End)) {
        chewing_handle_End(ctx);
    } else if (keyEvent.key().check(FcitxKey_space, KeyState::Shift)) {
        chewing_handle_ShiftSpace(ctx);
    } else if (keyEvent.key().check(FcitxKey_Left, KeyState::Shift)) {
        chewing_handle_ShiftLeft(ctx);
    } else if (keyEvent.key().check(FcitxKey_Right, KeyState::Shift)) {
        chewing_handle_ShiftRight(ctx);
    } else if (keyEvent.key().check(FcitxKey_Return)) {
        chewing_handle_Enter(ctx);
    } else if (keyEvent.key().states() == KeyState::Ctrl &&
               Key(keyEvent.key().sym()).isDigit()) {
        chewing_handle_CtrlNum(ctx, keyEvent.key().sym());
    } else {
        // to do: more chewing_handle
        return;
    }

    if (chewing_keystroke_CheckAbsorb(ctx)) {
        keyEvent.filterAndAccept();
        return updateUI(ic);
    } else if (chewing_keystroke_CheckIgnore(ctx)) {
        return;
    } else if (chewing_commit_Check(ctx)) {
        keyEvent.filterAndAccept();
        UniqueCPtr<char, chewing_free> str(chewing_commit_String(ctx));
        ic->commitString(str.get());
        return updateUI(ic);
    } else {
        keyEvent.filterAndAccept();
        return updateUI(ic);
    }
}

void ChewingEngine::filterKey(const InputMethodEntry &, KeyEvent &keyEvent) {
    if (keyEvent.isRelease()) {
        return;
    }
    auto ic = keyEvent.inputContext();
    if (ic->inputPanel().candidateList() &&
        (keyEvent.key().isSimple() || keyEvent.key().isCursorMove() ||
         keyEvent.key().check(FcitxKey_space, KeyState::Shift) ||
         keyEvent.key().check(FcitxKey_Tab) ||
         keyEvent.key().check(FcitxKey_Return, KeyState::Shift))) {
        return keyEvent.filterAndAccept();
    }

    if (!ic->inputPanel().candidateList()) {
        // Check if this key will produce something, if so, flush
        if (!keyEvent.key().hasModifier() &&
            Key::keySymToUnicode(keyEvent.key().sym())) {
            flushBuffer(keyEvent);
        }
    }
}

void ChewingEngine::updateUI(InputContext *ic) {
    CHEWING_DEBUG() << "updateUI";
    ChewingContext *ctx = context_.get();

    // clean up window asap
    ic->inputPanel().reset();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);

    UniqueCPtr<char, chewing_free> buf_str(chewing_buffer_String(ctx));
    const char *zuin_str = chewing_bopomofo_String_static(ctx);

    std::string text = buf_str.get();
    std::string_view zuin = zuin_str;
    CHEWING_DEBUG() << "Text: " << text << " Zuin: " << zuin;
    /* if not check done, so there is candidate word */
    if (!chewing_cand_CheckDone(ctx)) {
        ic->inputPanel().setCandidateList(
            std::make_unique<ChewingCandidateList>(this, ic));
        if (!ic->inputPanel().candidateList()->size()) {
            ic->inputPanel().setCandidateList(nullptr);
        }
    }

    /* there is nothing */
    if (zuin.empty() && text.empty() && !ic->inputPanel().candidateList()) {
        ic->updatePreedit();
        return;
    }

    auto len = utf8::lengthValidated(text);
    if (len == utf8::INVALID_LENGTH) {
        return;
    }
    const auto useClientPreedit =
        ic->capabilityFlags().test(CapabilityFlag::Preedit);
    const auto format =
        useClientPreedit ? TextFormatFlag::Underline : TextFormatFlag::NoFlag;
    Text preedit;

    int cur = chewing_cursor_Current(ctx);
    int rcur = text.size();
    if (cur >= 0 && static_cast<size_t>(cur) < len) {
        rcur = utf8::ncharByteLength(text.begin(), cur);
    }
    preedit.setCursor(rcur);

    // insert zuin in the middle
    preedit.append(text.substr(0, rcur), format);
    preedit.append(std::string(zuin), {TextFormatFlag::HighLight, format});
    preedit.append(text.substr(rcur), format);

    if (chewing_aux_Check(ctx)) {
        const char *aux_str = chewing_aux_String_static(ctx);
        std::string aux = aux_str;
        ic->inputPanel().setAuxDown(Text(aux));
    } else if (!suggestions_.empty() && suggestionForBuffer_ == text &&
               zuin.empty()) {
        // Show the LLM's alternatives as a passive hint; one is only ever
        // applied when the user presses Ctrl+Enter. Chewing's own aux
        // messages (errors etc.) take priority over it above.
        std::string hint = "懶 " + suggestions_[suggestionIndex_];
        if (suggestions_.size() > 1) {
            hint += " (" + std::to_string(suggestionIndex_ + 1) + "/" +
                    std::to_string(suggestions_.size()) + " ↓)";
        }
        hint += " [Ctrl+Enter]";
        ic->inputPanel().setAuxDown(Text(hint));
    }

    if (useClientPreedit) {
        ic->inputPanel().setClientPreedit(preedit);
    } else {
        ic->inputPanel().setPreedit(preedit);
    }

    ic->updatePreedit();

    // Fire (or refresh) the async suggestion request for the buffer now on
    // screen. No-op unless the buffer is a settled multi-char sentence.
    requestSuggestion(ic);
}

void ChewingEngine::flushBuffer(InputContextEvent &event) {
    auto ctx = context_.get();
    // This check is because we ask the client to do the focus out commit.
    if (event.type() != EventType::InputContextFocusOut) {
        chewing_handle_Enter(ctx);
        if (chewing_commit_Check(ctx)) {
            UniqueCPtr<char, chewing_free> str(chewing_commit_String(ctx));
            event.inputContext()->commitString(str.get());
        }
        UniqueCPtr<char, chewing_free> buf_str(chewing_buffer_String(ctx));
        const char *zuin_str = chewing_bopomofo_String_static(ctx);
        std::string text = buf_str.get();
        std::string zuin = zuin_str;
        text += zuin;
        if (!text.empty()) {
            event.inputContext()->commitString(text);
        }
    }
    doReset(event);
}

} // namespace fcitx

FCITX_ADDON_FACTORY(fcitx::ChewingEngineFactory);
