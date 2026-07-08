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
#include <fcitx-utils/standardpath.h>
#include <fstream>
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

// Renders `sentence` with only the characters that differ from `reference`
// (the typed buffer) carrying the HighLight flag, so the proposed change
// (e.g. 再→在) pops out of an otherwise identical sentence. Sentence and
// buffer normally have the same char count (both are built from the same
// interval spans), but walk defensively in case one runs short.
Text diffHighlightText(const std::string &sentence,
                       const std::string &reference) {
    Text out;
    size_t sentOff = 0;
    size_t refOff = 0;
    while (sentOff < sentence.size()) {
        const size_t sentLen =
            utf8::ncharByteLength(sentence.begin() + sentOff, 1);
        const size_t refLen =
            refOff < reference.size()
                ? utf8::ncharByteLength(reference.begin() + refOff, 1)
                : 0;
        const bool same = refLen > 0 && sentLen == refLen &&
                          sentence.compare(sentOff, sentLen, reference, refOff,
                                           refLen) == 0;
        out.append(sentence.substr(sentOff, sentLen),
                   same ? TextFormatFlag::NoFlag : TextFormatFlag::HighLight);
        sentOff += sentLen;
        refOff += refLen;
    }
    return out;
}

// One alternative for the *focused segment* in segment-conversion. Selecting
// it (number key or click) sets that segment's character and advances focus.
// The candidate list holds the focused phrase's candidates, so paging/labels
// are handled for us.
class SegmentCandidateWord : public CandidateWord {
public:
    SegmentCandidateWord(ChewingEngine *engine, int index, std::string text)
        : CandidateWord(Text(std::move(text))), engine_(engine),
          index_(index) {}

    void select(InputContext *inputContext) const override {
        engine_->pickSegment(inputContext, index_);
    }

private:
    ChewingEngine *engine_;
    int index_;
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

// Decode-mode request: the typed bopomofo syllables themselves, for the
// daemon's libchewing-free path (-t phonetic_table.tsv). The LLM decodes the
// sentence directly rather than reranking libchewing's candidates.
std::string buildDecodeRequest(const std::vector<std::string> &syllables,
                               int n, const std::string &context) {
    json req;
    req["syllables"] = syllables;
    req["n"] = n;
    if (!context.empty()) {
        req["context"] = context;
    }
    return req.dump() + "\n";
}

// Why a query came back empty, so the UI can tell the user instead of
// silently dropping back to composing (the old behaviour, which read as
// "the key did nothing").
enum class RerankError {
    None,    // got sentences
    Connect, // socket/connect failed: daemon not running
    Io,      // connected but send/recv failed or timed out
    Empty,   // daemon answered but offered no usable sentence
};

// Talks to slothingd over its Unix socket. Publishes the connected fd into
// `fdSlot` so the engine's destructor can shutdown() it to unblock this
// thread at teardown; clears it again before returning. Returns an empty
// list on any failure (daemon absent, timeout, malformed/partial response)
// and reports which kind through `err`.
// Sends a pre-built request string to slothingd and returns its "sentences".
// Shared by rerank and decode paths; see queryReranker/queryDecoder.
std::vector<std::string> sendDaemonRequest(const std::string &req,
                                           std::atomic<int> &fdSlot,
                                           RerankError &err) {
    err = RerankError::Empty;

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        err = RerankError::Connect;
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
        err = RerankError::Connect;
        return {};
    }
    fdSlot.store(fd);

    auto finish = [&](std::vector<std::string> result) {
        fdSlot.store(-1);
        close(fd);
        return result;
    };

    size_t off = 0;
    while (off < req.size()) {
        ssize_t w = send(fd, req.data() + off, req.size() - off, MSG_NOSIGNAL);
        if (w <= 0) {
            err = RerankError::Io;
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
        err = RerankError::Io;
        return finish({});
    }
    // Parse with the same library the daemon dumped with; a truncated or
    // malformed reply throws and yields no candidates rather than garbage.
    try {
        json parsed = json::parse(resp);
        if (parsed.contains("sentences")) {
            auto sentences =
                parsed["sentences"].get<std::vector<std::string>>();
            if (!sentences.empty()) {
                err = RerankError::None;
            }
            return finish(std::move(sentences));
        }
    } catch (const std::exception &) {
    }
    return finish({});
}

std::vector<std::string> queryReranker(const std::vector<std::vector<std::string>> &positions,
                                       int n, const std::string &context,
                                       std::atomic<int> &fdSlot,
                                       RerankError &err) {
    if (positions.empty()) {
        err = RerankError::Empty;
        return {};
    }
    return sendDaemonRequest(buildRerankRequest(positions, n, context), fdSlot,
                             err);
}

std::vector<std::string> queryDecoder(const std::vector<std::string> &syllables,
                                      int n, const std::string &context,
                                      std::atomic<int> &fdSlot,
                                      RerankError &err) {
    if (syllables.empty()) {
        err = RerankError::Empty;
        return {};
    }
    return sendDaemonRequest(buildDecodeRequest(syllables, n, context), fdSlot,
                             err);
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
    loadPhoneticTable();
}

void ChewingEngine::loadPhoneticTable() {
    // Env override (dev/testing), else the installed data location.
    std::string path;
    if (const char *env = std::getenv("SLOTHING_PHONETIC_TABLE")) {
        path = env;
    } else {
        path = StandardPath::global().locate(StandardPath::Type::Data,
                                             "slothing/phonetic_table.tsv");
    }
    if (path.empty()) {
        CHEWING_DEBUG() << "phonetic table not found; DecodeMode unavailable";
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
        std::vector<std::string> chars;
        // one Han character per UTF-8 codepoint (3 bytes in the BMP, but read
        // the lead byte to stay correct for any width)
        const std::string &rest = line.substr(tab + 1);
        for (size_t i = 0; i < rest.size();) {
            unsigned char c = rest[i];
            size_t len = (c & 0xF8) == 0xF0   ? 4
                         : (c & 0xF0) == 0xE0 ? 3
                         : (c & 0xE0) == 0xC0 ? 2
                                              : 1;
            chars.push_back(rest.substr(i, len));
            i += len;
        }
        if (!chars.empty()) {
            phoneticTable_.emplace(std::move(syl), std::move(chars));
        }
    }
    CHEWING_DEBUG() << "loaded phonetic table: " << phoneticTable_.size()
                    << " syllables from " << path;
}

ChewingEngine::~ChewingEngine() {
    hintTimer_.reset();
    stopWorker();
}

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

void ChewingEngine::clearHint() {
    hintTimer_.reset();
    hintText_.clear();
    hintForBuffer_.clear();
}

// Re-armed from updateUI on every Composing repaint, so it only fires after
// the user pauses. Cheap pre-checks here; the real harvest is in fireHint().
void ChewingEngine::armHintTimer(InputContext *ic) {
    if (!*config_.LlmConvert) {
        return;
    }
    ChewingContext *ctx = context_.get();
    const char *zuin = chewing_bopomofo_String_static(ctx);
    if (chewing_buffer_Len(ctx) < 2 || (zuin && zuin[0]) ||
        chewing_cand_TotalChoice(ctx) > 0) {
        clearHint();
        return;
    }
    UniqueCPtr<char, chewing_free> buf(chewing_buffer_String(ctx));
    std::string buffer = buf ? buf.get() : "";
    // Nothing to do if we already have (or are fetching) a hint for this exact
    // buffer -- avoids re-arming on cursor moves / repeated repaints.
    if (buffer == hintForBuffer_ || buffer == hintInflightBuffer_) {
        return;
    }
    convertIc_ = ic->watch();
    hintTimer_ = instance_->eventLoop().addTimeEvent(
        CLOCK_MONOTONIC, now(CLOCK_MONOTONIC) + 400000, 10000,
        [this](EventSourceTime *, uint64_t) {
            if (convertState_ == ConvertState::Composing) {
                if (auto *ic = convertIc_.get()) {
                    fireHint(ic);
                }
            }
            return true;
        });
}

void ChewingEngine::fireHint(InputContext *ic) {
    hintTimer_.reset();
    ChewingContext *ctx = context_.get();
    const char *zuin = chewing_bopomofo_String_static(ctx);
    if (chewing_buffer_Len(ctx) < 2 || (zuin && zuin[0]) ||
        chewing_cand_TotalChoice(ctx) > 0) {
        return;
    }
    std::vector<std::pair<int, int>> intervals;
    auto positions = collectCandidatePositions(ctx, intervals);
    UniqueCPtr<char, chewing_free> buf(chewing_buffer_String(ctx));
    std::string buffer = buf ? buf.get() : "";
    if (positions.empty() || !matchesPositions(buffer, positions)) {
        return;
    }
    bool anyAlt = false;
    for (const auto &p : positions) {
        if (p.size() > 1) {
            anyAlt = true;
            break;
        }
    }
    if (!anyAlt) {
        return;
    }
    stopWorker(); // serialise: at most one background request at a time
    hintInflightBuffer_ = buffer;
    const uint64_t generation = ++convertGeneration_;
    auto icRef = ic->watch();
    workerStop_.store(false);
    worker_ = std::thread([this, icRef, generation, buffer,
                           positions = std::move(positions)]() {
        std::string top;
        RerankError err = RerankError::None;
        for (auto &s : queryReranker(positions, 1, "", inflightFd_, err)) {
            if (matchesPositions(s, positions)) {
                top = s;
                break;
            }
        }
        if (workerStop_.load()) {
            return;
        }
        dispatcher_.schedule([this, icRef, generation, buffer, top]() {
            if (generation != convertGeneration_ ||
                convertState_ != ConvertState::Composing) {
                return;
            }
            hintInflightBuffer_.clear();
            if (top.empty() || top == buffer) {
                return; // chewing's own sentence is already the top pick
            }
            hintText_ = top;
            hintForBuffer_ = buffer;
            if (auto *ic = icRef.get()) {
                updateUI(ic);
            }
        });
    });
}

void ChewingEngine::cancelConversion(InputContext *ic, std::string notice) {
    stopWorker();
    convertTimer_.reset();
    convertTicks_ = 0;
    convertNotice_ = std::move(notice);
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertBopomofo_.clear();
    segSel_.clear();
    segFocus_ = 0;
    updateUI(ic);
}

void ChewingEngine::startConversion(InputContext *ic) {
    ChewingContext *ctx = context_.get();
    clearHint(); // leaving Composing; drop any pending/shown hint
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
        convertNotice_ = "無法轉換此句";
        updateUI(ic);
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

    // Long sentences: convert only a tail window of intervals and hand the
    // untouched front of the buffer to the LLM as context instead. This keeps
    // the choice set (and thus latency) bounded, and concentrates the
    // alternatives where the user is still working -- the end of the
    // sentence. The window always ends at the last interval.
    constexpr size_t kMaxLlmIntervals = 4;
    std::string prefixText;
    std::vector<std::vector<std::string>> llmPositions = positions;
    if (intervals.size() > kMaxLlmIntervals) {
        size_t firstIdx = intervals.size() - kMaxLlmIntervals;
        int splitChar = intervals[firstIdx].first;
        prefixText = utf8CharSlice(buffer, 0, splitChar);
        llmPositions.assign(positions.begin() + firstIdx, positions.end());
        context += prefixText;
        CHEWING_DEBUG() << "startConversion: tail window of "
                        << llmPositions.size() << " intervals, prefix='"
                        << prefixText << "'";
    }
    bool anyAlternative = false;
    for (const auto &cands : llmPositions) {
        if (cands.size() > 1) {
            anyAlternative = true;
            break;
        }
    }
    if (!anyAlternative) {
        CHEWING_DEBUG() << "startConversion: window offers no alternative";
        convertNotice_ = "此句無其他候選";
        updateUI(ic);
        return;
    }

    // Tear down any prior worker, then start fresh.
    stopWorker();
    convertState_ = ConvertState::Converting;
    convertBuffer_ = buffer;
    convertPositions_ = positions;
    convertIntervals_ = intervals;
    convertBopomofo_ = std::move(bopomofo);
    const uint64_t generation = ++convertGeneration_;
    // Ask for enough passes that, after dedup, at least a few real
    // alternatives usually survive.
    const int n = std::max(*config_.LlmCandidateCount, 4);
    auto icRef = ic->watch();

    convertNotice_.clear();
    convertTicks_ = 0;
    convertIc_ = ic->watch();
    updateUI(ic); // show the "converting" indicator

    // Repaint the indicator twice a second while the worker runs: animated
    // dots + elapsed time make it obvious the IME is alive, and keep the
    // Esc-to-cancel hint in front of the user.
    convertTimer_ = instance_->eventLoop().addTimeEvent(
        CLOCK_MONOTONIC, now(CLOCK_MONOTONIC) + 500000, 10000,
        [this](EventSourceTime *source, uint64_t) {
            if (convertState_ == ConvertState::Converting) {
                if (auto *timerIc = convertIc_.get()) {
                    convertTicks_++;
                    updateUI(timerIc);
                }
                source->setNextInterval(500000);
                source->setOneShot();
            }
            return true;
        });

    worker_ = std::thread([this, icRef, generation, n,
                           buffer = std::move(buffer),
                           positions = std::move(llmPositions),
                           prefixText = std::move(prefixText),
                           context = std::move(context)]() {
        RerankError err = RerankError::None;
        std::vector<std::string> verified;
        for (auto &sentence :
             queryReranker(positions, n, context, inflightFd_, err)) {
            // Keep only sentences provably built from real chewing candidates
            // for the converted window, then re-attach the untouched front so
            // every choice is a full replacement for the buffer.
            if (!matchesPositions(sentence, positions)) {
                continue;
            }
            std::string full = prefixText + sentence;
            if (std::find(verified.begin(), verified.end(), full) ==
                verified.end()) {
                verified.push_back(std::move(full));
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
        // Never end a conversion silently: name the reason in the panel so
        // an idle daemon or a dry reply doesn't read as a dead key.
        std::string failNotice;
        if (verified.empty()) {
            switch (err) {
            case RerankError::Connect:
                failNotice = "slothingd 未執行";
                break;
            case RerankError::Io:
                failNotice = "slothingd 無回應";
                break;
            default:
                failNotice = "無建議";
                break;
            }
        }
        dispatcher_.schedule([this, icRef, generation,
                              verified = std::move(verified),
                              failNotice = std::move(failNotice)]() {
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
                // nothing to offer; back to composing, but say why
                cancelConversion(ic, failNotice);
                return;
            }
            showConversionChoices(ic, verified);
        });
    });
}

void ChewingEngine::startDecodeConversion(InputContext *ic) {
    ChewingContext *ctx = context_.get();
    clearHint();
    // Same settled-buffer preconditions as startConversion.
    const char *zuin = chewing_bopomofo_String_static(ctx);
    if (chewing_buffer_Len(ctx) < 1 || (zuin && zuin[0]) ||
        chewing_cand_TotalChoice(ctx) > 0) {
        return;
    }

    // The typed syllables, one bopomofo string (with tone) per syllable --
    // exactly chewing's phone sequence. libchewing is used here only as the
    // zhuyin keyboard parser; its sentence decode is discarded.
    std::vector<std::string> syllables;
    int phoneLen = chewing_get_phoneSeqLen(ctx);
    if (unsigned short *phones = chewing_get_phoneSeq(ctx)) {
        for (int i = 0; i < phoneLen; i++) {
            char pb[24] = {0};
            if (chewing_phone_to_bopomofo(phones[i], pb, sizeof(pb)) == 0 &&
                pb[0]) {
                syllables.emplace_back(pb);
            } else {
                syllables.clear();
                break;
            }
        }
        chewing_free(phones);
    }
    if (syllables.empty()) {
        convertNotice_ = "無法轉換此句";
        updateUI(ic);
        return;
    }

    // Each syllable becomes a 1-char segment; its candidates are the phonetic
    // table's legal characters for that reading (frequency-ordered). A
    // syllable missing from the table aborts decode for this buffer.
    std::vector<std::vector<std::string>> positions;
    std::vector<std::pair<int, int>> intervals;
    for (size_t i = 0; i < syllables.size(); i++) {
        auto it = phoneticTable_.find(syllables[i]);
        if (it == phoneticTable_.end() || it->second.empty()) {
            convertNotice_ = "此音無候選";
            updateUI(ic);
            return;
        }
        positions.push_back(it->second);
        intervals.emplace_back(static_cast<int>(i), static_cast<int>(i + 1));
    }

    // Document context (same as startConversion).
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

    stopWorker();
    convertState_ = ConvertState::Converting;
    // convertBuffer_ is seeded to the daemon's decoded sentence once it
    // returns; until then use chewing's own decode as a placeholder so the
    // "converting" indicator has something to show.
    UniqueCPtr<char, chewing_free> buf(chewing_buffer_String(ctx));
    convertBuffer_ = buf.get();
    convertPositions_ = positions;
    convertIntervals_ = intervals;
    convertBopomofo_ = syllables; // 1 syllable per char; aligns for teaching
    const uint64_t generation = ++convertGeneration_;
    const int n = std::max(*config_.LlmCandidateCount, 4);
    auto icRef = ic->watch();
    convertNotice_.clear();
    convertTicks_ = 0;
    convertIc_ = ic->watch();
    updateUI(ic);

    convertTimer_ = instance_->eventLoop().addTimeEvent(
        CLOCK_MONOTONIC, now(CLOCK_MONOTONIC) + 500000, 10000,
        [this](EventSourceTime *source, uint64_t) {
            if (convertState_ == ConvertState::Converting) {
                if (auto *timerIc = convertIc_.get()) {
                    convertTicks_++;
                    updateUI(timerIc);
                }
                source->setNextInterval(500000);
                source->setOneShot();
            }
            return true;
        });

    worker_ = std::thread([this, icRef, generation, n,
                           syllables = std::move(syllables),
                           positions = std::move(positions),
                           context = std::move(context)]() {
        RerankError err = RerankError::None;
        std::vector<std::string> verified;
        for (auto &sentence :
             queryDecoder(syllables, n, context, inflightFd_, err)) {
            // The grammar restricts output to table chars, but verify anyway:
            // every char must be one of its segment's candidates.
            if (!matchesPositions(sentence, positions)) {
                continue;
            }
            if (std::find(verified.begin(), verified.end(), sentence) ==
                verified.end()) {
                verified.push_back(std::move(sentence));
            }
        }
        if (workerStop_.load()) {
            return;
        }
        std::string failNotice;
        if (verified.empty()) {
            switch (err) {
            case RerankError::Connect:
                failNotice = "slothingd 未執行";
                break;
            case RerankError::Io:
                failNotice = "slothingd 無回應";
                break;
            default:
                failNotice = "無法解碼";
                break;
            }
        }
        dispatcher_.schedule([this, icRef, generation,
                              verified = std::move(verified),
                              failNotice = std::move(failNotice)]() {
            if (generation != convertGeneration_ ||
                convertState_ != ConvertState::Converting) {
                return;
            }
            auto *ic = icRef.get();
            if (!ic) {
                convertState_ = ConvertState::Composing;
                return;
            }
            if (verified.empty()) {
                cancelConversion(ic, failNotice);
                return;
            }
            // Seed convertBuffer_ to the best decode so segment defaults and
            // the "original" fallback track the LLM, not chewing.
            convertBuffer_ = verified.front();
            showConversionChoices(ic, verified);
        });
    });
}

void ChewingEngine::showConversionChoices(InputContext *ic,
                                          const std::vector<std::string> &sentences) {
    convertState_ = ConvertState::Choosing;
    convertTimer_.reset();

    // Seed each segment's selection from the LLM's best sentence: for every
    // interval, pick the candidate whose text matches that sentence's span.
    // Defaults to chewing's own choice (index that reconstructs the buffer)
    // when no match is found.
    segSel_.assign(convertPositions_.size(), 0);
    const std::string &best = sentences.empty() ? convertBuffer_ : sentences[0];
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        int from = convertIntervals_[i].first, to = convertIntervals_[i].second;
        std::string span = utf8CharSlice(best, from, to);
        std::string bufSpan = utf8CharSlice(convertBuffer_, from, to);
        int fallback = 0;
        for (size_t j = 0; j < convertPositions_[i].size(); j++) {
            if (convertPositions_[i][j] == span) {
                segSel_[i] = static_cast<int>(j);
                span.clear();
                break;
            }
            if (convertPositions_[i][j] == bufSpan) {
                fallback = static_cast<int>(j);
            }
        }
        if (!span.empty()) { // no LLM match; keep chewing's own char
            segSel_[i] = fallback;
        }
    }
    // Focus the first segment that actually offers a choice.
    segFocus_ = 0;
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        if (convertPositions_[i].size() > 1) {
            segFocus_ = static_cast<int>(i);
            break;
        }
    }
    renderSegments(ic);
    CHEWING_DEBUG() << "segment-convert: " << convertPositions_.size()
                    << " segments, composed '" << composedSentence() << "'";
}

std::string ChewingEngine::composedSentence() const {
    std::string out;
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        int sel = (i < segSel_.size()) ? segSel_[i] : 0;
        if (sel >= 0 && sel < static_cast<int>(convertPositions_[i].size())) {
            out += convertPositions_[i][sel];
        }
    }
    return out;
}

void ChewingEngine::renderSegments(InputContext *ic) {
    ic->inputPanel().reset();

    // Preedit: the whole composed sentence, focused segment highlighted.
    const auto useClientPreedit =
        ic->capabilityFlags().test(CapabilityFlag::Preedit);
    const auto base =
        useClientPreedit ? TextFormatFlag::Underline : TextFormatFlag::NoFlag;
    Text preedit;
    for (size_t i = 0; i < convertPositions_.size(); i++) {
        int sel = segSel_[i];
        const std::string &seg = convertPositions_[i][sel];
        if (static_cast<int>(i) == segFocus_) {
            preedit.append(seg, {TextFormatFlag::HighLight, base});
        } else {
            preedit.append(seg, base);
        }
    }
    if (useClientPreedit) {
        ic->inputPanel().setClientPreedit(preedit);
    } else {
        ic->inputPanel().setPreedit(preedit);
    }

    // Candidate list = the focused segment's alternatives, cursor on current.
    auto list = std::make_unique<CommonCandidateList>();
    list->setPageSize(*config_.PageSize);
    auto layout = *config_.CandidateLayout;
    list->setLayoutHint(layout == CandidateLayoutHint::NotSet
                            ? CandidateLayoutHint::Horizontal
                            : layout);
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
    const auto &cands = convertPositions_[segFocus_];
    for (size_t j = 0; j < cands.size(); j++) {
        list->append(std::make_unique<SegmentCandidateWord>(
            this, static_cast<int>(j), cands[j]));
    }
    list->setGlobalCursorIndex(segSel_[segFocus_]);
    ic->inputPanel().setCandidateList(std::move(list));

    ic->inputPanel().setAuxDown(Text("←→ 選詞　↑↓ 換字　⏎ 確認　Esc 取消"));
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
}

void ChewingEngine::pickSegment(InputContext *ic, int candIdx) {
    if (segFocus_ < 0 || segFocus_ >= static_cast<int>(convertPositions_.size())) {
        return;
    }
    if (candIdx >= 0 &&
        candIdx < static_cast<int>(convertPositions_[segFocus_].size())) {
        segSel_[segFocus_] = candIdx;
    }
    // Advance to the next segment that offers a choice; if none, stay.
    for (int i = segFocus_ + 1; i < static_cast<int>(convertPositions_.size());
         i++) {
        if (convertPositions_[i].size() > 1) {
            segFocus_ = i;
            break;
        }
    }
    renderSegments(ic);
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
    segSel_.clear();
    segFocus_ = 0;
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
    clearHint();
    convertTimer_.reset();
    convertTicks_ = 0;
    convertNotice_.clear();
    convertState_ = ConvertState::Composing;
    convertGeneration_++;
    convertBuffer_.clear();
    convertPositions_.clear();
    convertIntervals_.clear();
    convertBopomofo_.clear();
    segSel_.clear();
    segFocus_ = 0;
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
        // Segment-conversion: ←/→ pick which phrase segment, ↑/↓ (and Space)
        // cycle the focused segment's candidate with the sentence updating
        // live, number keys pick directly, Enter commits, Esc cancels. This
        // unifies "whole-sentence LLM convert" and "per-phrase fix" into one
        // modal flow (Japanese-henkan / Sogou style).
        auto ic = keyEvent.inputContext();
        keyEvent.filterAndAccept();
        if (segFocus_ < 0 || convertPositions_.empty()) {
            cancelConversion(ic);
            return;
        }
        if (keyEvent.key().check(FcitxKey_Escape)) {
            cancelConversion(ic);
            return;
        }
        if (keyEvent.key().check(FcitxKey_Return)) {
            acceptConversion(ic, composedSentence());
            return;
        }
        const int nseg = static_cast<int>(convertPositions_.size());
        auto &foc = segSel_[segFocus_];
        const int ncand = static_cast<int>(convertPositions_[segFocus_].size());
        if (keyEvent.key().check(FcitxKey_Right)) {
            for (int i = segFocus_ + 1; i < nseg; i++) {
                if (convertPositions_[i].size() > 1) { segFocus_ = i; break; }
            }
        } else if (keyEvent.key().check(FcitxKey_Left)) {
            for (int i = segFocus_ - 1; i >= 0; i--) {
                if (convertPositions_[i].size() > 1) { segFocus_ = i; break; }
            }
        } else if (keyEvent.key().check(FcitxKey_Down) ||
                   keyEvent.key().check(FcitxKey_space)) {
            foc = (foc + 1) % ncand;
        } else if (keyEvent.key().check(FcitxKey_Up)) {
            foc = (foc - 1 + ncand) % ncand;
        } else if (keyEvent.key().isSimple()) {
            const char *selkeys =
                builtin_selectkeys[static_cast<int>(*config_.SelectionKey)];
            char sym = static_cast<char>(keyEvent.key().sym() & 0xff);
            for (int i = 0; i < 10 && selkeys[i]; i++) {
                if (selkeys[i] == sym && i < ncand) {
                    pickSegment(ic, i); // sets segment + advances focus
                    return;
                }
            }
        }
        renderSegments(ic);
        return; // all other keys swallowed while choosing
    }
    // Composing: any keystroke clears a lingering conversion notice (it is
    // one-shot feedback, not a status line).
    convertNotice_.clear();
    // The convert key starts a conversion -- but only when there is
    // a settled buffer to convert. With no preedit (or a pending syllable /
    // open candidate window) the key is NOT consumed, so a modifier combo like
    // Ctrl+Return still reaches the application (e.g. "send message").
    if (*config_.LlmConvert &&
        keyEvent.key().normalize().checkKeyList(*config_.ConvertKey)) {
        const char *cz = chewing_bopomofo_String_static(ctx);
        if (chewing_buffer_Len(ctx) >= 1 && !(cz && cz[0]) &&
            chewing_cand_TotalChoice(ctx) == 0) {
            keyEvent.filterAndAccept();
            // DecodeMode bypasses libchewing's decode entirely (LLM decodes
            // the syllables) when the phonetic table is available; otherwise
            // fall back to reranking libchewing's candidates.
            if (*config_.DecodeMode && !phoneticTable_.empty()) {
                startDecodeConversion(keyEvent.inputContext());
            } else {
                startConversion(keyEvent.inputContext());
            }
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
        // Animated dots (repainted by convertTimer_) show the request is
        // alive; elapsed seconds appear once it stops feeling instant.
        std::string aux = "轉換中";
        for (int i = 0; i <= convertTicks_ % 3; i++) {
            aux += "·";
        }
        if (convertTicks_ >= 2) {
            aux += " " + std::to_string(convertTicks_ / 2) + "s";
        }
        aux += "（Esc 取消）";
        ic->inputPanel().setAuxUp(Text(aux));
        ic->updatePreedit();
        ic->updateUserInterface(UserInterfaceComponent::InputPanel);
        return;
    }

    // clean up window asap. The conversion notice / debounced hint are drawn
    // below (as auxDown) so they never collide with the preedit on one line.
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

    // Drop a stale hint (buffer moved on) before deciding what to show.
    if (!hintText_.empty() && hintForBuffer_ != text) {
        hintText_.clear();
        hintForBuffer_.clear();
    }
    // One auxDown line, drawn below the preedit so it never collides with it.
    // Priority: chewing's own message > conversion-outcome notice > hint.
    if (chewing_aux_Check(ctx)) {
        ic->inputPanel().setAuxDown(Text(chewing_aux_String_static(ctx)));
    } else if (!convertNotice_.empty()) {
        ic->inputPanel().setAuxDown(Text(convertNotice_));
    } else if (!hintText_.empty() && hintForBuffer_ == text) {
        Text hint;
        hint.append("建議 ", TextFormatFlag::NoFlag);
        Text diff = diffHighlightText(hintText_, text);
        for (size_t i = 0; i < diff.size(); i++) {
            hint.append(diff.stringAt(i), diff.formatAt(i));
        }
        hint.append("　⏎轉換", TextFormatFlag::NoFlag);
        ic->inputPanel().setAuxDown(hint);
    }

    if (useClientPreedit) {
        ic->inputPanel().setClientPreedit(preedit);
    } else {
        ic->inputPanel().setPreedit(preedit);
    }

    ic->updatePreedit();

    // Arm (or re-arm) the debounced hint for the buffer now on screen. Fires
    // only after the user pauses; a no-op for empty/mid-syllable buffers.
    if (convertState_ == ConvertState::Composing) {
        armHintTimer(ic);
    }
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
