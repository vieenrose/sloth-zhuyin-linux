/*
 * SPDX-FileCopyrightText: 2010~2017 CSSlayer <wengxt@gmail.com>
 * SPDX-FileCopyrightText: 2026 sloth-zhuyin-linux
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 */
#ifndef _FCITX5_CHEWING_EIM_H_
#define _FCITX5_CHEWING_EIM_H_

// Slothing: a libchewing-free zhuyin input method. The keyboard is the
// dependency-free ZhuyinBuffer FSM (zhuyin.h); the decoder is the local
// SlothLM model via slothingd's decode mode. No libchewing.
#include <atomic>
#include <cstdint>
#include <fcitx-config/configuration.h>
#include <fcitx-config/enum.h>
#include <fcitx-config/iniparser.h>
#include <fcitx-utils/event.h>
#include <fcitx-utils/eventdispatcher.h>
#include <fcitx-utils/i18n.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/trackableobject.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/instance.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "segment.h"
#include "zhuyin.h"

namespace fcitx {
FCITX_CONFIG_ENUM_NAME_WITH_I18N(CandidateLayoutHint, N_("Not Set"),
                                 N_("Vertical"), N_("Horizontal"));

enum class ChewingSelectionKey {
    CSK_Digit,
    CSK_asdfghjkl,
    CSK_asdfzxcv89,
    CSK_asdfjkl789,
    CSK_aoeuhtn789,
    CSK_1234qweras,
    CSK_dstnaeo789
};

FCITX_CONFIG_ENUM_NAME(ChewingSelectionKey, "1234567890", "asdfghjkl;",
                       "asdfzxcv89", "asdfjkl789", "aoeuhtn789", "1234qweras",
                       "dstnaeo789");
FCITX_CONFIG_ENUM_I18N_ANNOTATION(ChewingSelectionKey, N_("1234567890"),
                                  N_("asdfghjkl;"), N_("asdfzxcv89"),
                                  N_("asdfjkl789"), N_("aoeuhtn789"),
                                  N_("1234qweras"), N_("dstnaeo789"));

FCITX_CONFIGURATION(
    ChewingConfig,
    OptionWithAnnotation<ChewingSelectionKey, ChewingSelectionKeyI18NAnnotation>
        SelectionKey{this, "SelectionKey", _("Selection Key"),
                     ChewingSelectionKey::CSK_Digit};
    Option<int, IntConstrain> PageSize{this, "PageSize", _("Page Size"), 10,
                                       IntConstrain(3, 10)};
    OptionWithAnnotation<CandidateLayoutHint, CandidateLayoutHintI18NAnnotation>
        CandidateLayout{this, "CandidateLayout", _("Candidate List Layout"),
                        fcitx::CandidateLayoutHint::NotSet};
    Option<bool> SpaceAsSelection{this, "SpaceAsSelection",
                                  _("Space as selection key"), true};
    KeyListOption ConvertKey{
        this, "ConvertKey", _("LLM convert key"), {Key("Control+Return")},
        KeyListConstrain({KeyConstrainFlag::AllowModifierLess})};
    Option<int, IntConstrain> LlmCandidateCount{
        this, "LlmCandidateCount", _("Number of LLM candidates"), 5,
        IntConstrain(1, 8)};);

class ChewingEngine final : public InputMethodEngine {
public:
    ChewingEngine(Instance *instance);
    ~ChewingEngine();
    Instance *instance() { return instance_; }
    const ChewingConfig &config() { return config_; }
    void keyEvent(const InputMethodEntry &entry, KeyEvent &keyEvent) override;
    void reloadConfig() override;
    void reset(const InputMethodEntry &entry,
               InputContextEvent &event) override;

    const Configuration *getConfig() const override { return &config_; }
    void setConfig(const RawConfig &config) override {
        config_.load(config, true);
        safeSaveAsIni(config_, "conf/chewing.conf");
    }

    // Commit the composed sentence, clear the buffer, back to Composing.
    void acceptConversion(InputContext *ic, const std::string &sentence);
    // Segment-conversion: set the focused segment to candidate `candIdx` and
    // advance focus (number key / click on a segment candidate).
    void pickSegment(InputContext *ic, int candIdx);
    // Set the focused segment AND the next one from a 2-char phrase candidate
    // (per-phrase Down-rank), then advance focus past both.
    void pickPhrase(InputContext *ic, int start, const std::string &phrase);

private:
    // Pull-model, three states. Composing: the ZhuyinBuffer accumulates typed
    // bopomofo (shown as the preedit); no LLM work. The convert key sends the
    // syllables to slothingd's decode mode (Converting) and shows the decoded
    // sentence as an editable segment list (Choosing).
    enum class ConvertState { Composing, Converting, Choosing };

    void updateUI(InputContext *ic);
    // Redraw the composing preedit: live-decoded Chinese when fresh, with the
    // pending (incomplete) syllable's bopomofo appended; raw bopomofo as the
    // fallback (decoder unavailable / not yet returned).
    void renderComposing(InputContext *ic);
    // 微軟新注音-style live conversion: after each completed syllable, decode
    // asynchronously and show the sentence as the preedit while Composing.
    void scheduleLiveDecode(InputContext *ic);
    // Begin a decode for the current buffer (spawn the worker).
    // commitDirect: decode then commit immediately (Enter path, 新注音
    // style) instead of opening the segment-choosing window (↓ path).
    void startDecode(InputContext *ic, bool commitDirect = false);
    // Enter segment-conversion for the decoded sentences (best seeds the
    // per-segment selection).
    void showConversionChoices(InputContext *ic,
                               const std::vector<std::string> &sentences);
    std::string composedSentence() const;
    void renderSegments(InputContext *ic);
    void cancelConversion(InputContext *ic, std::string notice = {});
    void stopWorker();
    void loadPhoneticTable();

    Instance *instance_;
    ChewingConfig config_;
    EventDispatcher dispatcher_;

    // Composing state, web-demo style: the raw keystream of the current run
    // (re-segmented live into zh/en tokens) plus the finalized tokens of
    // earlier runs (a run ends on a tone key or space).
    std::string rawKeys_;
    std::vector<slothing::SegTok> committedToks_;
    // Insertion cursor over committedToks_ (token granularity): index into
    // the token list where the current run / new input lands. -1 = end.
    int tokCursor_ = -1;
    // Manual modes (微軟 conventions): lone-Shift toggles forced-English
    // (literal input, no segmentation); Shift+Space toggles 全形/半形.
    bool enMode_ = false;
    bool fullWidth_ = false;
    bool shiftAlone_ = false; // lone-Shift press tracking
    // ` symbol menu (微軟-style categorized symbols)
    bool symbolMode_ = false;
    int symCat_ = 0;
    void renderSymbols(InputContext *ic);
public:
    void pickSymbol(InputContext *ic, const std::string &sym);
private:
    // ↓ focus hint for showConversionChoices (segment at the cursor). -1 =
    // first ambiguous segment.
    int pendingFocus_ = -1;
    std::unique_ptr<slothing::Segmenter> segmenter_; // built from the table
    bool composingEmpty() const {
        return rawKeys_.empty() && committedToks_.empty();
    }
    void commitRun();

    ConvertState convertState_ = ConvertState::Composing;
    // The decode's per-syllable candidate lists, interval spans (1 char each),
    // and the syllables it was started for. Main thread only.
    std::string convertBuffer_; // best decoded sentence (seed / original)
    std::vector<std::vector<std::string>> convertPositions_;
    std::vector<std::pair<int, int>> convertIntervals_;
    std::vector<std::string> convertSyllables_;
    // Token list the conversion was started for (zh syllable / en literal);
    // en tokens become single-candidate segments and get spaces on commit.
    std::vector<slothing::SegTok> convertToks_;
    // One selected candidate index per interval, and which segment the arrows
    // act on.
    std::vector<int> segSel_;
    // Selection state when Choosing began — diffed at commit to learn the
    // user's corrections (sent to the daemon's persistent store).
    std::vector<int> initialSel_;
    // Segments the user explicitly picked (hints for re-scoring; also the
    // only positions the learn store records).
    std::set<int> userFixed_;
    // Re-decode the sentence conditioned on the user's picks (hint-aware
    // model) and update the segments the user has NOT touched.
    void rescoreChoosing(InputContext *ic);
    int segFocus_ = 0;
    // Candidate window visibility (chewing: a pick CLOSES the window; ↓
    // reopens it; Esc closes it before cancelling the whole conversion).
    bool candListOpen_ = true;
    // ←→ highlight over the 詞 (aux) options: -1 = highlight is in the char
    // list; >=0 = phrase index. Enter confirms whichever is highlighted.
    int phraseHl_ = -1;
    // Model-ranked 2-char phrase candidates per focus position, fetched
    // lazily from the daemon while Choosing. Main thread only.
    std::map<int, std::vector<std::pair<int, std::string>>> phraseCands_;

    // Live (modeless) conversion state: the decoded preedit (spaces around
    // English) and the tokens it corresponds to. Used only when it matches
    // the current committedToks_.
    std::string livePreedit_; // joined display (commit form)
    std::vector<std::string> liveDisp_; // per-token display, aligned to toks
    std::vector<slothing::SegTok> liveToks_;
    uint64_t liveGeneration_ = 0;

    std::string convertNotice_;
    std::unique_ptr<EventSourceTime> convertTimer_;
    int convertTicks_ = 0;
    TrackableObjectReference<InputContext> convertIc_;

    // Single background worker for the in-flight decode; joinable + stop flag +
    // shared fd let the destructor tear down safely.
    std::thread worker_;
    std::atomic<bool> workerStop_{false};
    std::atomic<int> inflightFd_{-1};
    uint64_t convertGeneration_ = 0;

    // Phonetic table: bopomofo syllable -> legal Traditional characters, for
    // the per-segment candidate lists. Loaded once at startup.
    std::unordered_map<std::string, std::vector<std::string>> phoneticTable_;
};

class ChewingEngineFactory : public AddonFactory {
public:
    AddonInstance *create(AddonManager *manager) override {
        registerDomain("fcitx5-chewing", FCITX_INSTALL_LOCALEDIR);
        return new ChewingEngine(manager->instance());
    }
};
} // namespace fcitx

#endif // _FCITX5_CHEWING_EIM_H_
