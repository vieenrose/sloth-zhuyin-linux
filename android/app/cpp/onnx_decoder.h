// OnnxDecoder — the on-device Decoder backed by ONNX Runtime (arm64-v8a).
// Owns one Ort::Env + one warm Ort::Session built from the model BYTES handed
// in from Kotlin/AssetManager (nothing touches the filesystem), plus the
// syllable/char vocabularies and the phonetic legality tables parsed from the
// asset strings. It is a faithful in-process port of engine/slothingd/
// slothingd_e.py's decode: phonetic-legality mask, char-hint channel (inert on
// the shipped hints-off checkpoint), ED1 typo repair with the insertion prior,
// n-best by lowest-margin flips, per-position model-ranked candidates, phrase
// scoring, and the learn-store logit bonuses.
//
// Every public method is safe to call from a worker thread: Ort::Session::Run
// is thread-safe for concurrent calls on one session and we hold no mutable
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

#include "onnxruntime_cxx_api.h"

namespace slothing {

class OnnxDecoder final : public Decoder {
public:
    // modelBytes/modelLen: the .onnx graph copied out of assets (parsed +
    // copied by ORT at construction, so the buffer need not outlive us).
    // sylVocabJson/char2idJson: the two vocab files verbatim. tableTsv: the
    // phonetic_table.tsv bytes (syllable \t chars). numThreads: intra-op ORT
    // threads (1-2 on a phone). learnPath: optional on-device learn.tsv (empty
    // = in-memory only, resets per process).
    OnnxDecoder(const void *modelBytes, size_t modelLen,
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

    bool ok() const { return session_ != nullptr; }

private:
    static constexpr int CTX_MAX = 12;
    static constexpr double CHAR_BONUS = 6.0, PHRASE_BONUS = 8.0;

    // one forward: syl[B,T] int64, amask[B,T] bool, (opt) hints[B,T] int64
    // -> logits[B,T,nChar] float. Returns the flat [B*T*nChar] buffer.
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

    Ort::Env env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memInfo_ =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    bool hasHints_ = false;
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

} // namespace slothing

#endif // _SLOTHING_ANDROID_ONNX_DECODER_H_
