/*
 * SPDX-FileCopyrightText: 2010~2017 CSSlayer <wengxt@gmail.com>
 * SPDX-FileCopyrightText: 2026 sloth-zhuyin-linux
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 */
#ifndef _FCITX5_CHEWING_EIM_H_
#define _FCITX5_CHEWING_EIM_H_

// Slothing: a libchewing-free zhuyin input method. The IME state machine
// (token buffer, segment-conversion window, re-scoring, learning) lives in
// the frontend-free shared core (engine/common/core.h) also used by the
// IBus engine; this class is the fcitx5 adapter: key decoding, async decode
// workers, and painting. No libchewing.
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
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core.h"
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
    // Segment-conversion: set the focused segment to candidate `candIdx`
    // (number key / click on a segment candidate).
    void pickSegment(InputContext *ic, int candIdx);
    // Set the focused segment AND the next one from a 2-char phrase candidate
    // (per-phrase Down-rank).
    void pickPhrase(InputContext *ic, int start, const std::string &phrase);

private:
    // Pull-model, three states. Composing: the token buffer accumulates typed
    // runs (shown as the preedit) with live decode. Enter/↓ send the
    // syllables to slothingd's decode mode (Converting) and either commit
    // directly or show the decoded sentence as an editable segment list
    // (Choosing).
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
    void renderSegments(InputContext *ic);
    void cancelConversion(InputContext *ic, std::string notice = {});
    void stopWorker();
    void loadPhoneticTable();
    // Document context before the cursor, when the app exposes it (conditions
    // the decode via the model's hint channel).
    std::string surroundingContext(InputContext *ic) const;

    Instance *instance_;
    ChewingConfig config_;
    EventDispatcher dispatcher_;

    // Shared frontend-free state machine (engine/common/core.h).
    slothing::ComposingCore comp_;
    slothing::ChoosingCore choosing_;

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

    ConvertState convertState_ = ConvertState::Composing;
    std::string convertBuffer_; // best decoded sentence (seed / original)
    // Positions/intervals/syllables computed at decode start; handed to
    // choosing_.begin() when the reply arrives.
    std::vector<std::vector<std::string>> convertPositions_;
    std::vector<std::pair<int, int>> convertIntervals_;
    std::vector<std::string> convertSyllables_;
    std::vector<slothing::SegTok> convertToks_;

    // Live (modeless) conversion state: the decoded preedit (spaces around
    // English) and the tokens it corresponds to. Used only when it matches
    // the current comp_.toks.
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
