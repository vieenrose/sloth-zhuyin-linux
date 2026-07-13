// OnnxDecoder — the on-device Decoder. NOW ggml-BACKED (libslothe / ternary
// GGUF encoder cross-compiled for arm64-v8a); the class name is kept so callers
// (jni_sloth.cpp, session.h) are untouched. Owns one slothe_model loaded
// from a GGUF file PATH handed in from Kotlin (the asset is copied to app cache
// first — see SlothImeService), plus the syllable/char vocabularies and the
// phonetic legality tables parsed from the asset strings. It is a faithful
// in-process port of engine/slothd/slothd_e.py's decode: phonetic-legality
// mask, char-hint channel (INERT — the shipped GGUF has no hints input), ED1
// typo repair with the insertion prior, n-best by lowest-margin flips,
// per-position model-ranked candidates, phrase scoring, and the learn-store
// logit bonuses.
//
// Every public method is safe to call from a worker thread: slothe_logits
// builds its own throwaway ggml graph/context per call and shares only the
// immutable weight buffer, so concurrent calls don't race; we hold no mutable
// decode state across calls (only the learn store has its own mutex).
#ifndef _SLOTHING_ANDROID_ONNX_DECODER_H_
#define _SLOTHING_ANDROID_ONNX_DECODER_H_

#include "decoder.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "slothe.h" // libslothe ggml forward: slothe_load / slothe_logits / ...

namespace sloth {

class OnnxDecoder final : public Decoder {
public:
    // ggufPath: filesystem path to slothe-t-25m.gguf (the asset is copied to app
    // cache by the caller; slothe_load mmaps/reads it here). sylVocabJson/
    // char2idJson: the two vocab files verbatim. tableTsv: the
    // phonetic_table.tsv bytes (syllable \t chars). numThreads: CPU threads for
    // the ggml backend (1-2 on a phone). learnPath: optional on-device learn.tsv
    // (empty = in-memory only, resets per process).
    OnnxDecoder(const std::string &ggufPath,
                const std::string &sylVocabJson, const std::string &char2idJson,
                const std::string &tableTsv, int numThreads,
                std::string learnPath = "");
    ~OnnxDecoder() override;

    OnnxDecoder(const OnnxDecoder &) = delete;
    OnnxDecoder &operator=(const OnnxDecoder &) = delete;

    DecodeResult decode(const std::vector<std::string> &syllables, int n,
                        const std::string &context) override;
    DecodeResult
    decodeWithHints(const std::vector<std::string> &syllables,
                    const std::map<int, std::string> &hints) override;
    std::vector<std::pair<double, std::string>>
    phrasesScored(const std::vector<std::string> &syllables, int at,
                  int n) override;
    void learn(const json &payload) override;

    bool ok() const { return model_ != nullptr; }

private:
    static constexpr int CTX_MAX = 12;
    // calibrated 2026-07-11 (see slothd_e.py): 6/8 over-personalized
    static constexpr double CHAR_BONUS = 2.0, PHRASE_BONUS = 3.0;

    // one forward over B sequences of length T. syl is the flat [B*T] batch
    // (int64 in the decode logic; cast to int32 per row for slothe_logits).
    // hints is IGNORED (the shipped GGUF has no hints input). Returns the flat
    // [B*T*nChar] logits buffer, matching the old ORT layout.
    std::vector<float> runForward(const std::vector<int64_t> &syl, int B, int T,
                                  const std::vector<int64_t> *hints);
    inline float lg(const std::vector<float> &f, int b, int t, int T,
                    int c) const {
        return f[(static_cast<size_t>(b) * T + t) * nChar_ + c];
    }
    int sylId(const std::string &s) const;
    std::vector<std::pair<std::string, int>> cands(const std::string &syl) const;
    std::vector<std::pair<std::string, std::vector<std::string>>>
    typoFixes(const std::string &syl) const;
    std::map<std::pair<int, std::string>, double>
    bonus(const std::vector<std::string> &syl) const;
    DecodeResult decodeImpl(const std::vector<std::string> &syl, int n,
                            const std::map<int, std::string> &hints,
                            const std::string &context);
    std::vector<std::pair<double, std::string>>
    phrases(const std::vector<std::string> &syl, int at, int n);
    void loadTable(const std::string &tsv);
    void loadLearn(const std::string &path);
    void saveLearn();

    slothe_model *model_ = nullptr; // owned; freed in dtor via slothe_free

    bool hasHints_ = false; // always false: the shipped GGUF has no hints input
    int nChar_ = 0;

    std::unordered_map<std::string, int> sylVocab_;
    std::unordered_map<std::string, int> char2id_;
    std::unordered_map<std::string, std::vector<std::string>> tonal_;
    std::unordered_map<std::string, std::vector<std::string>> toneless_;

    std::unordered_map<std::string, std::string> learnChar_;
    std::map<std::pair<std::string, std::string>, std::string> learnPhrase_;
    std::string learnFile_;
    std::mutex learnMu_;
};

} // namespace sloth

#endif // _SLOTHING_ANDROID_ONNX_DECODER_H_
