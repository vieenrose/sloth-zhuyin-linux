/*
 * SPDX-FileCopyrightText: 2012~2012 Tai-Lin Chu <tailinchu@gmail.com>
 * SPDX-FileCopyrightText: 2012~2017 CSSlayer <wengxt@gmail.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 */
#include "eim.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fcitx-utils/utf8.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/statusarea.h>
#include <fcitx/text.h>
#include <fcitx/userinterfacemanager.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

using json = nlohmann::json;

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

// One LLM-proposed full sentence in the conversion candidate list. Selecting
// it hands the sentence to the engine, which commits it and (optionally)
// teaches chewing. Held in a CommonCandidateList, so paging/labels/cursor are
// handled for us.
class SlothingCandidateWord : public CandidateWord {
public:
    SlothingCandidateWord(ChewingEngine *engine, std::string sentence)
        : CandidateWord(Text(sentence)), engine_(engine),
          sentence_(std::move(sentence)) {}

    void select(InputContext *inputContext) const override {
        engine_->acceptConversion(inputContext, sentence_);
    }

private:
    ChewingEngine *engine_;
    std::string sentence_;
};

void logger(void *, int, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// Conversion runs on an explicit key press on a background thread, so a slow
// daemon delays only the candidate list, never a keystroke. Generous timeout:
// LFM2.5-230M Q4_0 is ~200-350ms idle (MODEL_BENCHMARKS.md); this leaves
// headroom under load.
constexpr int kSlothingdTimeoutMs = 3000;

// Keep this derivation identical to default_socket_path() in slothingd.cpp
// and the shell logic in packaging/run-slothingd.sh so both ends agree.
std::string slothingdSocketPath() {
    if (const char *env = std::getenv("SLOTHINGD_SOCKET")) {
        return env;
    }
    if (const char *xdg = std::getenv("XDG_RUNTIME_DIR")) {
        return std::string(xdg) + "/slothingd.sock";
    }
    return "/tmp/slothingd.sock";
}

// Byte-substring of a UTF-8 string covering character positions [from,to).
std::string utf8CharSlice(const std::string &s, int from, int to) {
    size_t byteStart = utf8::ncharByteLength(s.begin(), from);
    size_t byteLen = utf8::ncharByteLength(s.begin() + byteStart, to - from);
    return s.substr(byteStart, byteLen);
}

// Walks the current buffer's phrase segmentation via libchewing's interval
// API and harvests, per segment, the phrase-length alternatives libchewing
// considers for it (used to build the LLM's constrained choice set and to
// teach chewing on accept). Runs only on the explicit convert key, not per
// keystroke. Cursor and phrase-choice-rearward mode are saved and restored.
//
// The cursor-advance loop carries a progress guard: with a candidate window
// briefly open, chewing_handle_Right can page instead of moving the cursor,
// which without a guard spins forever on the UI thread. Candidates whose
// UTF-8 length differs from the interval span are dropped (chewing_cand_open
// can surface other lengths; letting them through once turned
// '我再重新考慮' into '我爞考'). An interval with no span-length alternative
// is pinned to the user's own text for that span instead of aborting the
// whole harvest -- aborting made long sentences (more intervals, more chances
// of one dry interval) silently unconvertible. Fills `intervals` with the
// [from,to) spans in lockstep with the returned positions.
std::vector<std::vector<std::string>>
collectCandidatePositions(ChewingContext *ctx,
                          std::vector<std::pair<int, int>> &intervalsOut) {
    intervalsOut.clear();
    std::vector<std::vector<std::string>> positions;
    if (chewing_buffer_Len(ctx) <= 0) {
        return positions;
    }

    int origCursor = chewing_cursor_Current(ctx);
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

    UniqueCPtr<char, chewing_free> bufStr(chewing_buffer_String(ctx));
    std::string bufferText = bufStr ? bufStr.get() : "";

    for (const auto &interval : intervals) {
        // Advance the cursor to the interval start, guarding against a
        // non-progressing step (see the freeze note above).
        int guard = 0;
        while (chewing_cursor_Current(ctx) < interval.from) {
            int before = chewing_cursor_Current(ctx);
            chewing_handle_Right(ctx);
            if (chewing_cursor_Current(ctx) == before || ++guard > CHEWING_MAX_LEN) {
                break;
            }
        }
        const size_t span = static_cast<size_t>(interval.to - interval.from);
        std::vector<std::string> cands;
        if (chewing_cand_open(ctx) == 0) {
            // The candidate window shows one phrase-length at a time;
            // cand_list_first/next cycle through the lengths. Walk every
            // list and keep candidates matching this interval's span --
            // stopping at the first list means a mismatched initial length
            // (common mid-sentence) harvests nothing at all.
            chewing_cand_list_first(ctx);
            int listGuard = 0;
            while (true) {
                chewing_cand_Enumerate(ctx);
                while (chewing_cand_hasNext(ctx)) {
                    UniqueCPtr<char, chewing_free> str(chewing_cand_String(ctx));
                    if (str && utf8::lengthValidated(std::string_view(
                                   str.get())) == span) {
                        cands.emplace_back(str.get());
                    }
                }
                if (!cands.empty() || !chewing_cand_list_has_next(ctx) ||
                    ++listGuard > CHEWING_MAX_LEN) {
                    break;
                }
                chewing_cand_list_next(ctx);
            }
            chewing_cand_close(ctx);
        }
        if (cands.empty()) {
            // Keep the conversion alive: this interval simply offers no
            // alternative, so pin it to what the user typed and let the LLM
            // still rework the other intervals.
            std::string slice = utf8CharSlice(bufferText, interval.from,
                                              interval.to);
            if (slice.empty()) {
                positions.clear();
                intervalsOut.clear();
                break;
            }
            cands.push_back(std::move(slice));
        }
        CHEWING_DEBUG() << "harvest interval [" << interval.from << ","
                        << interval.to << ") -> " << cands.size()
                        << " candidates";
        positions.push_back(std::move(cands));
        intervalsOut.emplace_back(interval.from, interval.to);
    }

    chewing_set_phraseChoiceRearward(ctx, origRearward);
    chewing_handle_Home(ctx);
    for (int i = 0; i < origCursor; i++) {
        chewing_handle_Right(ctx);
    }

    return positions;
}

// Serialize the request with the same JSON library the daemon parses with,
// so escaping can never drift between the two ends.
std::string buildRerankRequest(const std::vector<std::vector<std::string>> &positions,
                               int n, const std::string &context) {
    json req;
    req["positions"] = positions;
    req["n"] = n;
    if (!context.empty()) {
        req["context"] = context;
    }
    return req.dump() + "\n";
}

// Talks to slothingd over its Unix socket. Publishes the connected fd into
// `fdSlot` so the engine's destructor can shutdown() it to unblock this
// thread at teardown; clears it again before returning. Returns an empty
// list on any failure (daemon absent, timeout, malformed/partial response).
std::vector<std::string> queryReranker(const std::vector<std::vector<std::string>> &positions,
                                       int n, const std::string &context,
                                       std::atomic<int> &fdSlot) {
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
    fdSlot.store(fd);

    auto finish = [&](std::vector<std::string> result) {
        fdSlot.store(-1);
        close(fd);
        return result;
    };

    std::string req = buildRerankRequest(positions, n, context);
    size_t off = 0;
    while (off < req.size()) {
        ssize_t w = send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
        if (w <= 0) {
            return finish({});
        }
        off += static_cast<size_t>(w);
    }
    shutdown(fd, SHUT_WR);

    std::string resp;
    char buf[4096];
    ssize_t r;
    while ((r = read(fd, buf, sizeof(buf))) > 0) {
        resp.append(buf, static_cast<size_t>(r));
    }
    if (resp.empty()) {
        return finish({});
    }
    // Parse with the same library the daemon dumped with; a truncated or
    // malformed reply throws and yields no candidates rather than garbage.
    try {
        json parsed = json::parse(resp);
        if (parsed.contains("sentences")) {
            return finish(parsed["sentences"].get<std::vector<std::string>>());
        }
    } catch (const std::exception &) {
    }
    return finish({});
}

// Verifies `sentence` is exactly the concatenation of one candidate from each
// entry of `positions`, in order -- so it can only be a combination of real
// libchewing alternatives, never arbitrary LLM output. Longest-match per
// position so a short candidate can't shadow a longer one at the same offset.
bool matchesPositions(const std::string &sentence,
                      const std::vector<std::vector<std::string>> &positions) {
    size_t off = 0;
    for (const auto &cands : positions) {
        size_t best = 0;
        for (const auto &c : cands) {
            if (c.size() > best && sentence.compare(off, c.size(), c) == 0) {
                best = c.size();
            }
        }
        if (best == 0) {
            return false;
        }
        off += best;
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

ChewingEngine::~ChewingEngine() { stopWorker(); }

void ChewingEngine::stopWorker() {
    // Unblock the worker's socket I/O (if any) and join it, so no background
    // thread can outlive the engine and touch freed state at teardown.
    workerStop_.store(true);
    int fd = inflightFd_.exchange(-1);
    if (fd >= 0) {
        shutdown(fd, SHUT_RDWR);
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    workerStop_.store(false);
}

void ChewingEngine::cancelConversion(InputContext *ic) {
    stopWorker();
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertBopomofo_.clear();
    updateUI(ic);
}

void ChewingEngine::startConversion(InputContext *ic) {
    ChewingContext *ctx = context_.get();
    // Only convert a settled multi-char buffer: nothing mid-syllable, no open
    // candidate window.
    const char *zuin = chewing_bopomofo_String_static(ctx);
    if (chewing_buffer_Len(ctx) < 1 || (zuin && zuin[0]) ||
        chewing_cand_TotalChoice(ctx) > 0) {
        CHEWING_DEBUG() << "startConversion: bail (bufLen="
                        << chewing_buffer_Len(ctx) << " zuin='"
                        << (zuin ? zuin : "") << "' totalChoice="
                        << chewing_cand_TotalChoice(ctx) << ")";
        return;
    }

    std::vector<std::pair<int, int>> intervals;
    auto positions = collectCandidatePositions(ctx, intervals);
    UniqueCPtr<char, chewing_free> buf(chewing_buffer_String(ctx));
    std::string buffer = buf.get();
    // If chewing's own sentence can't be rebuilt from the harvest, the harvest
    // is unreliable -- don't convert (the LLM would be fed a broken choice set).
    if (positions.empty() || !matchesPositions(buffer, positions)) {
        CHEWING_DEBUG() << "startConversion: harvest unusable (positions="
                        << positions.size() << " buffer='" << buffer << "')";
        return;
    }
    bool anyAlternative = false;
    for (const auto &cands : positions) {
        if (cands.size() > 1) {
            anyAlternative = true;
            break;
        }
    }
    if (!anyAlternative) {
        CHEWING_DEBUG() << "startConversion: no interval offers an alternative";
        return;
    }
    CHEWING_DEBUG() << "startConversion: buffer='" << buffer << "' positions="
                    << positions.size();

    // Capture the typed pronunciation (with tones) directly from chewing's
    // phone sequence; used to teach chewing if a correction is accepted.
    // Any conversion failure clears the whole track, which makes
    // teachChewing() a no-op rather than a wrong-pronunciation write.
    std::vector<std::string> bopomofo;
    int phoneLen = chewing_get_phoneSeqLen(ctx);
    if (unsigned short *phones = chewing_get_phoneSeq(ctx)) {
        for (int i = 0; i < phoneLen; i++) {
            char pb[24] = {0};
            // returns 0 on success (verified against libchewing 0.6.0)
            if (chewing_phone_to_bopomofo(phones[i], pb, sizeof(pb)) == 0 &&
                pb[0]) {
                bopomofo.emplace_back(pb);
            } else {
                bopomofo.clear();
                break;
            }
        }
        chewing_free(phones);
    }

    // Document context: the text already committed before the cursor, when
    // the client app exposes it. This is what lets the model resolve
    // homophones by what the user is actually writing, not just the buffer.
    std::string context;
    if (ic->capabilityFlags().test(CapabilityFlag::SurroundingText) &&
        ic->surroundingText().isValid()) {
        const auto &st = ic->surroundingText();
        const std::string &t = st.text();
        auto tLen = utf8::lengthValidated(t);
        if (tLen != utf8::INVALID_LENGTH && st.cursor() <= tLen) {
            context = t.substr(0, utf8::ncharByteLength(t.begin(), st.cursor()));
        }
    }

    // Tear down any prior worker, then start fresh.
    stopWorker();
    convertState_ = ConvertState::Converting;
    convertBuffer_ = buffer;
    convertPositions_ = positions;
    convertIntervals_ = intervals;
    convertBopomofo_ = std::move(bopomofo);
    const uint64_t generation = ++convertGeneration_;
    const int n = *config_.LlmCandidateCount;
    auto icRef = ic->watch();

    updateUI(ic); // show the "converting" indicator

    worker_ = std::thread([this, icRef, generation, n,
                           buffer = std::move(buffer),
                           positions = std::move(positions),
                           context = std::move(context)]() {
        std::vector<std::string> verified;
        for (auto &sentence : queryReranker(positions, n, context, inflightFd_)) {
            // Keep only sentences provably built from real chewing candidates.
            if (matchesPositions(sentence, positions) &&
                std::find(verified.begin(), verified.end(), sentence) ==
                    verified.end()) {
                verified.push_back(std::move(sentence));
            }
        }
        // Always offer chewing's own sentence too (as the last entry when the
        // LLM proposed alternatives), so the list is complete without Esc.
        if (!verified.empty() &&
            std::find(verified.begin(), verified.end(), buffer) ==
                verified.end()) {
            verified.push_back(buffer);
        }
        if (workerStop_.load()) {
            return; // engine tearing down; don't touch it
        }
        dispatcher_.schedule([this, icRef, generation,
                              verified = std::move(verified)]() {
            if (generation != convertGeneration_ ||
                convertState_ != ConvertState::Converting) {
                return; // cancelled / superseded
            }
            auto *ic = icRef.get();
            if (!ic) {
                convertState_ = ConvertState::Composing;
                return;
            }
            if (verified.empty()) {
                cancelConversion(ic); // nothing to offer; back to composing
                return;
            }
            showConversionChoices(ic, verified);
        });
    });
}

void ChewingEngine::showConversionChoices(InputContext *ic,
                                          const std::vector<std::string> &sentences) {
    convertState_ = ConvertState::Choosing;

    auto list = std::make_unique<CommonCandidateList>();
    list->setPageSize(*config_.PageSize);
    list->setLayoutHint(*config_.CandidateLayout);
    const char *selkeys =
        builtin_selectkeys[static_cast<int>(*config_.SelectionKey)];
    KeyList selectionKeys;
    std::vector<std::string> labels;
    for (int i = 0; i < 10 && selkeys[i]; i++) {
        selectionKeys.push_back(Key(std::string(1, selkeys[i])));
        labels.push_back(std::string(1, selkeys[i]) + ".");
    }
    list->setSelectionKey(selectionKeys);
    list->setLabels(labels);
    for (const auto &s : sentences) {
        list->append(std::make_unique<SlothingCandidateWord>(this, s));
    }
    list->setGlobalCursorIndex(0);

    ic->inputPanel().reset();
    // Keep the original sentence visible in the preedit above the choices.
    const auto useClientPreedit =
        ic->capabilityFlags().test(CapabilityFlag::Preedit);
    const auto format =
        useClientPreedit ? TextFormatFlag::Underline : TextFormatFlag::NoFlag;
    Text preedit(convertBuffer_, format);
    if (useClientPreedit) {
        ic->inputPanel().setClientPreedit(preedit);
    } else {
        ic->inputPanel().setPreedit(preedit);
    }
    ic->inputPanel().setCandidateList(std::move(list));
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
    CHEWING_DEBUG() << "showing " << sentences.size() << " LLM candidate(s)";
}

void ChewingEngine::teachChewing(const std::string &chosen) {
    if (!*config_.LlmLearn) {
        return;
    }
    ChewingContext *ctx = context_.get();
    // Only teach with a clean, fully-captured bopomofo track that lines up with
    // the buffer we converted; otherwise skip entirely (never feed chewing a
    // guessed/misaligned pronunciation).
    const int bufChars =
        static_cast<int>(utf8::lengthValidated(std::string_view(convertBuffer_)));
    if (bufChars <= 0 ||
        static_cast<int>(convertBopomofo_.size()) != bufChars) {
        CHEWING_DEBUG() << "learn: bopomofo track misaligned ("
                        << convertBopomofo_.size() << " vs " << bufChars
                        << " chars), skipping";
        return;
    }

    for (const auto &iv : convertIntervals_) {
        int from = iv.first, to = iv.second;
        if (from < 0 || to > bufChars || from >= to) {
            continue;
        }
        std::string orig = utf8CharSlice(convertBuffer_, from, to);
        std::string repl = utf8CharSlice(chosen, from, to);
        if (orig == repl) {
            continue; // unchanged interval, nothing to learn
        }
        std::string bopomofo;
        bool ok = true;
        for (int i = from; i < to; i++) {
            if (convertBopomofo_[i].empty()) {
                ok = false;
                break;
            }
            if (!bopomofo.empty()) {
                bopomofo += " ";
            }
            bopomofo += convertBopomofo_[i];
        }
        if (!ok) {
            continue;
        }
        int rc = chewing_userphrase_add(ctx, repl.c_str(), bopomofo.c_str());
        CHEWING_DEBUG() << "learn: userphrase_add('" << repl << "','" << bopomofo
                        << "') -> " << rc;
    }
}

void ChewingEngine::acceptConversion(InputContext *ic, const std::string &sentence) {
    ChewingContext *ctx = context_.get();
    ic->commitString(sentence);
    teachChewing(sentence);

    // Discard chewing's own composing buffer (we committed our text instead of
    // routing through chewing_handle_Enter) and return to Composing.
    chewing_clean_preedit_buf(ctx);
    chewing_clean_bopomofo_buf(ctx);
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertBopomofo_.clear();
    updateUI(ic);
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
    // Leave chewing's own auto-learn at its default (enabled) -- it encodes
    // its own commits with correct tones. Our experimental teachChewing() is
    // gated separately on LlmLearn and must not disable chewing's learning.
}

void ChewingEngine::reset(const InputMethodEntry &, InputContextEvent &event) {
    doReset(event);
}

void ChewingEngine::doReset(InputContextEvent &event) {
    ChewingContext *ctx = context_.get();
    chewing_handle_Esc(ctx);
    stopWorker();
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertBopomofo_.clear();
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
    CHEWING_DEBUG() << "keyEvent(top): " << keyEvent.key().toString()
                    << " state=" << static_cast<int>(convertState_)
                    << " llmConvert=" << *config_.LlmConvert
                    << " convertKeyMatch="
                    << keyEvent.key().normalize().checkKeyList(
                           *config_.ConvertKey);

    // --- LLM conversion state machine (pull model) -------------------------
    // Composing is stock chewing except that the convert key starts a
    // conversion. Converting and Choosing are modal: keys act on the
    // conversion, never leak into chewing or the application.
    if (convertState_ == ConvertState::Converting) {
        keyEvent.filterAndAccept();
        if (keyEvent.key().check(FcitxKey_Escape)) {
            cancelConversion(keyEvent.inputContext());
        }
        return; // swallow everything else while the worker runs
    }
    if (convertState_ == ConvertState::Choosing) {
        auto ic = keyEvent.inputContext();
        auto candList = ic->inputPanel().candidateList();
        if (!candList) { // shouldn't happen; fail safe to composing
            cancelConversion(ic);
            return;
        }
        keyEvent.filterAndAccept();
        if (keyEvent.key().check(FcitxKey_Escape)) {
            cancelConversion(ic);
            return;
        }
        // Selection keys (same set as chewing's own window).
        const char *selkeys =
            builtin_selectkeys[static_cast<int>(*config_.SelectionKey)];
        if (keyEvent.key().isSimple()) {
            char sym = static_cast<char>(keyEvent.key().sym() & 0xff);
            for (int i = 0; i < 10 && selkeys[i]; i++) {
                if (selkeys[i] == sym && i < candList->size()) {
                    candList->candidate(i).select(ic);
                    return;
                }
            }
        }
        auto *movable = candList->toCursorMovable();
        auto *pageable = candList->toPageable();
        if (keyEvent.key().check(FcitxKey_Down) && movable) {
            movable->nextCandidate();
        } else if (keyEvent.key().check(FcitxKey_Up) && movable) {
            movable->prevCandidate();
        } else if (keyEvent.key().check(FcitxKey_Page_Down) && pageable &&
                   pageable->hasNext()) {
            pageable->next();
        } else if (keyEvent.key().check(FcitxKey_Page_Up) && pageable &&
                   pageable->hasPrev()) {
            pageable->prev();
        } else if (keyEvent.key().check(FcitxKey_Return) ||
                   keyEvent.key().check(FcitxKey_space)) {
            // Commit the highlighted candidate.
            int idx = candList->cursorIndex();
            if (idx >= 0 && idx < candList->size()) {
                candList->candidate(idx).select(ic);
                return;
            }
        }
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        return; // all other keys swallowed while choosing
    }
    // Composing: the convert key starts a conversion -- but only when there is
    // a settled buffer to convert. With no preedit (or a pending syllable /
    // open candidate window) the key is NOT consumed, so a modifier combo like
    // Ctrl+Return still reaches the application (e.g. "send message").
    if (*config_.LlmConvert &&
        keyEvent.key().normalize().checkKeyList(*config_.ConvertKey)) {
        const char *cz = chewing_bopomofo_String_static(ctx);
        if (chewing_buffer_Len(ctx) >= 1 && !(cz && cz[0]) &&
            chewing_cand_TotalChoice(ctx) == 0) {
            keyEvent.filterAndAccept();
            startConversion(keyEvent.inputContext());
            return;
        }
    }
    // ------------------------------------------------------------------------

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

    // While Choosing, showConversionChoices() owns the panel; don't rebuild it
    // from chewing state here (that would blow away the LLM candidate list).
    if (convertState_ == ConvertState::Choosing) {
        return;
    }
    // While Converting, keep the original sentence visible with a small
    // "converting" indicator; no chewing candidate window.
    if (convertState_ == ConvertState::Converting) {
        ic->inputPanel().reset();
        const auto useClientPreedit =
            ic->capabilityFlags().test(CapabilityFlag::Preedit);
        const auto format =
            useClientPreedit ? TextFormatFlag::Underline : TextFormatFlag::NoFlag;
        Text preedit(convertBuffer_, format);
        if (useClientPreedit) {
            ic->inputPanel().setClientPreedit(preedit);
        } else {
            ic->inputPanel().setPreedit(preedit);
        }
        ic->inputPanel().setAuxUp(Text("轉換中…"));
        ic->updatePreedit();
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        return;
    }

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
    }

    if (useClientPreedit) {
        ic->inputPanel().setClientPreedit(preedit);
    } else {
        ic->inputPanel().setPreedit(preedit);
    }

    ic->updatePreedit();
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
