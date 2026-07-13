// The frontend-free decode seam. The shared IME state machine (core.h) reaches
// the model through a Decoder*: on the Linux fcitx5/ibus engines it is left null
// and core.h calls the Unix-socket free functions in daemon.h
// (queryDecoder / queryDecoderWithHints / queryPhrasesScored / sendLearn); on
// Android there is no daemon, so SlothSession injects an in-process
// OnnxDecoder (engine runs ONNX Runtime on-device). Same state machine, two
// transports, no behavioral difference on Linux.
#ifndef _SLOTHING_COMMON_DECODER_H_
#define _SLOTHING_COMMON_DECODER_H_

#include "nlohmann/json.hpp"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace sloth {

using json = nlohmann::json;

// Mirror of what daemon.h's replies carry: the n-best sentences, and
// (optionally) the per-position ranked candidate lists the ↓ window shows.
// `candidates` is empty when the model didn't return a ranking (the caller
// then falls back to the phonetic table's order, exactly like the socket path
// when the reply has no "candidates" field).
struct DecodeResult {
    std::vector<std::string> sentences;                 // best-first
    std::vector<std::vector<std::string>> candidates;   // per position, or empty
};

// Injected into the core so it never touches a socket on-device. All three
// query methods are the heavy ones (an ONNX forward each): callers run them on
// a worker thread — never the IME/UI thread.
class Decoder {
public:
    virtual ~Decoder() = default;

    // n-best decode of one zhuyin-syllable run, optional left context (the
    // text already to the left of the caret, for the LM). == queryDecoder().
    virtual DecodeResult decode(const std::vector<std::string> &syllables,
                                int n, const std::string &context) = 0;

    // Decode conditioned on the user's fixed picks (position -> chosen char);
    // used by ChoosingCore::rescore() to re-rank the untouched segments.
    // == queryDecoderWithHints(). Returns n=1 + re-ranked per-position lists.
    virtual DecodeResult
    decodeWithHints(const std::vector<std::string> &syllables,
                    const std::map<int, std::string> &hints) = 0;

    // Model-ranked 2-char phrase candidates covering positions (at, at+1) with
    // their joint probabilities, best-first. == queryPhrasesScored().
    virtual std::vector<std::pair<double, std::string>>
    phrasesScored(const std::vector<std::string> &syllables, int at,
                  int n) = 0;

    // Persist the user's corrections (ChoosingCore::learnPayload()). On-device
    // this appends to a local learn store the decoder consults; a no-op is a
    // legal implementation. == sendLearn(). Cheap / fire-and-forget.
    virtual void learn(const json & /*payload*/) {}
};

} // namespace sloth

#endif // _SLOTHING_COMMON_DECODER_H_
