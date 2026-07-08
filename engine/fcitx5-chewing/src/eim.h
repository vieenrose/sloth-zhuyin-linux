/*
 * SPDX-FileCopyrightText: 2010~2017 CSSlayer <wengxt@gmail.com>
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 */
#ifndef _FCITX5_CHEWING_EIM_H_
#define _FCITX5_CHEWING_EIM_H_

#include <atomic>
#include <chewing.h>
#include <cstdint>
#include <fcitx-config/configuration.h>
#include <fcitx-config/enum.h>
#include <fcitx-config/iniparser.h>
#include <fcitx-utils/event.h>
#include <fcitx-utils/eventdispatcher.h>
#include <fcitx-utils/trackableobject.h>
#include <fcitx-utils/i18n.h>
#include <fcitx-utils/key.h>
#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h>
#include <fcitx/candidatelist.h>
#include <fcitx/inputmethodengine.h>
#include <fcitx/instance.h>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

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

enum class ChewingLayout {
    Default,
    Hsu,
    IBM,
    GinYieh,
    ETen,
    ETen26,
    Dvorak,
    DvorakHsu,
    DACHEN_CP26,
    HanYuPinYin,
    Carpalx
};
FCITX_CONFIG_ENUM_NAME(ChewingLayout, "Default Keyboard", "Hsu's Keyboard",
                       "IBM Keyboard", "Gin-Yieh Keyboard", "ETen Keyboard",
                       "ETen26 Keyboard", "Dvorak Keyboard",
                       "Dvorak Keyboard with Hsu's support",
                       "DACHEN_CP26 Keyboard", "Han-Yu PinYin Keyboard",
                       "Carpalx Keyboard");
FCITX_CONFIG_ENUM_I18N_ANNOTATION(ChewingLayout, N_("Default Keyboard"),
                                  N_("Hsu's Keyboard"), N_("IBM Keyboard"),
                                  N_("Gin-Yieh Keyboard"), N_("ETen Keyboard"),
                                  N_("ETen26 Keyboard"), N_("Dvorak Keyboard"),
                                  N_("Dvorak Keyboard with Hsu's support"),
                                  N_("DACHEN_CP26 Keyboard"),
                                  N_("Han-Yu PinYin Keyboard"),
                                  N_("Carpalx Keyboard"));

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
    Option<bool> UseKeypadAsSelectionKey{
        this, "UseKeypadAsSelection", _("Use Keypad as Selection key"), false};
    Option<bool> AddPhraseForward{this, "AddPhraseForward",
                                  _("Add Phrase Forward"), true};
    Option<bool> ChoiceBackward{this, "ChoiceBackward",
                                _("Backward phrase choice"), true};
    Option<bool> AutoShiftCursor{this, "AutoShiftCursor",
                                 _("Automatically shift cursor"), false};
    Option<bool> EasySymbolInput{this, "EasySymbolInput",
                                 _("Enable easy symbol"), false};
    Option<bool> SpaceAsSelection{this, "SpaceAsSelection",
                                  _("Space as selection key"), true};
    OptionWithAnnotation<ChewingLayout, ChewingLayoutI18NAnnotation> Layout{
        this, "Layout", _("Keyboard Layout"), ChewingLayout::Default};
    Option<bool> LlmConvert{
        this, "LlmConvert",
        _("Enable LLM sentence conversion (press the convert key)"), true};
    KeyListOption ConvertKey{
        this, "ConvertKey", _("LLM convert key"), {Key("Control+Return")},
        KeyListConstrain({KeyConstrainFlag::AllowModifierLess})};
    Option<int, IntConstrain> LlmCandidateCount{
        this, "LlmCandidateCount", _("Number of LLM candidates"), 5,
        IntConstrain(1, 8)};
    Option<bool> LlmLearn{
        this, "LlmLearn",
        _("Teach chewing the phrase when an LLM candidate is accepted"),
        true};);

class ChewingEngine final : public InputMethodEngine {
public:
    ChewingEngine(Instance *instance);
    ~ChewingEngine();
    Instance *instance() { return instance_; }
    const ChewingConfig &config() { return config_; }
    void activate(const InputMethodEntry &entry,
                  InputContextEvent &event) override;
    void deactivate(const InputMethodEntry &entry,
                    InputContextEvent &event) override;
    void keyEvent(const InputMethodEntry &entry, KeyEvent &keyEvent) override;
    void filterKey(const InputMethodEntry &, KeyEvent &) override;
    void reloadConfig() override;
    void reset(const InputMethodEntry &entry,
               InputContextEvent &event) override;
    void save() override;

    const Configuration *getConfig() const override { return &config_; }
    void setConfig(const RawConfig &config) override {
        config_.load(config, true);
        populateConfig();
        safeSaveAsIni(config_, "conf/chewing.conf");
    }

    void updateUI(InputContext *ic);

    void flushBuffer(InputContextEvent &event);
    void doReset(InputContextEvent &event);

    ChewingContext *context() { return context_.get(); }

    // Accept an LLM candidate: commit it, optionally teach chewing, reset.
    // Called from the candidate word's select().
    void acceptConversion(InputContext *ic, const std::string &sentence);
    // Segment-conversion: set the focused segment to candidate `candIdx` and
    // advance focus. Called from a segment candidate word's select() (number
    // key or click).
    void pickSegment(InputContext *ic, int candIdx);

private:
    FCITX_ADDON_DEPENDENCY_LOADER(chttrans, instance_->addonManager());

    void populateConfig();

    // Explicit "convert" pull-model. The engine is normally in Composing and
    // behaves exactly like stock chewing. Pressing the convert key on a
    // settled multi-char buffer harvests candidates, fires one background
    // request (Converting), and shows the LLM alternatives as a native
    // candidate list (Choosing). No LLM work happens during ordinary typing.
    enum class ConvertState { Composing, Converting, Choosing };

    // Begin a conversion for the current buffer (harvest + spawn worker).
    void startConversion(InputContext *ic);
    // Enter segment-conversion for the returned sentences (the best sets the
    // initial per-segment selection); the user then tweaks segments.
    void showConversionChoices(InputContext *ic,
                               const std::vector<std::string> &sentences);
    // The sentence currently composed from the per-segment selections.
    std::string composedSentence() const;
    // Repaint the segment view: full sentence with the focused segment
    // highlighted (preedit) + the focused segment's candidates (list).
    void renderSegments(InputContext *ic);
    // Abandon an in-progress/shown conversion, keep the composing buffer.
    // A non-empty `notice` is shown as a brief auxUp message (e.g. why the
    // conversion produced nothing) and clears on the next keystroke.
    void cancelConversion(InputContext *ic, std::string notice = {});
    // Best-effort: register each changed phrase+bopomofo with libchewing so
    // it learns the correction. No-op unless LlmLearn is on and a clean
    // bopomofo was captured for the interval.
    void teachChewing(const std::string &chosen);
    // Join the background worker (if any) after unblocking its socket.
    void stopWorker();

    // Debounced typing-time hint. While Composing, after a short pause,
    // fireHint() runs ONE background check asking whether the LLM would offer
    // a better sentence than chewing's current buffer; if so, a passive auxUp
    // marker appears. It never opens the candidate list or intercepts keys --
    // the user still presses the convert key to act. Reuses worker_/stopWorker
    // and convertGeneration_ so at most one background request runs at a time.
    void armHintTimer(InputContext *ic);
    void fireHint(InputContext *ic);
    void clearHint();

    Instance *instance_;
    ChewingConfig config_;
    UniqueCPtr<ChewingContext, chewing_delete> context_;
    EventDispatcher dispatcher_;

    ConvertState convertState_ = ConvertState::Composing;
    // The buffer text a conversion was started for, its harvested per-interval
    // candidate lists, the interval boundaries, and the returned choices.
    // Main thread only.
    std::string convertBuffer_;
    std::vector<std::vector<std::string>> convertPositions_;
    std::vector<std::pair<int, int>> convertIntervals_;
    // Per-character bopomofo (with tones) of the buffer being converted,
    // captured at startConversion() from chewing_get_phoneSeq -- chewing's own
    // record of the typed pronunciation -- and used to teach chewing when a
    // conversion is accepted. Never reconstructed from keystrokes: an earlier
    // keystroke-tracking attempt captured syllables before their tone key and
    // wrote a wrong-pronunciation userphrase that crashed libchewing's
    // decoder.
    std::vector<std::string> convertBopomofo_;
    // Segment-conversion state (Choosing): one selected candidate index per
    // interval of convertPositions_, and which segment the arrows act on.
    std::vector<int> segSel_;
    int segFocus_ = 0;

    // One-shot feedback line (auxUp) shown after a conversion ends without
    // choices -- "無建議", "slothingd 未執行", ... Cleared on the next
    // keystroke so it never lingers.
    std::string convertNotice_;
    // Repaints the "轉換中" indicator every 500ms while Converting, so the
    // user sees progress (animated dots + elapsed seconds) instead of a
    // frozen label. Destroyed on any state exit.
    std::unique_ptr<EventSourceTime> convertTimer_;
    int convertTicks_ = 0;
    TrackableObjectReference<InputContext> convertIc_;

    // Single background worker for the in-flight request. The joinable thread
    // + stop flag + shared socket fd let the destructor tear down safely
    // instead of leaving a detached thread to touch freed state.
    std::thread worker_;
    std::atomic<bool> workerStop_{false};
    std::atomic<int> inflightFd_{-1};
    uint64_t convertGeneration_ = 0;

    // Debounced-hint state (main thread only).
    std::unique_ptr<EventSourceTime> hintTimer_;
    std::string hintText_;           // suggested sentence, better than buffer
    std::string hintForBuffer_;      // buffer hintText_ applies to
    std::string hintInflightBuffer_; // buffer a hint request is in flight for
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
