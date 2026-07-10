package com.slothing.ime

import android.inputmethodservice.InputMethodService
import android.text.SpannableString
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection
import android.widget.LinearLayout
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Slothing InputMethodService — the Dàqiān bopomofo soft keyboard layered over
 * the shared frontend-free IME core ([Core], engine/common/core.h on-device).
 *
 * Docs: the service returns its keyboard from onCreateInputView() (created once,
 * reused), edits the field only through getCurrentInputConnection()
 * (setComposingText / commitText / finishComposingText), and is bound via
 * BIND_INPUT_METHOD + the android.view.im metadata (AndroidManifest.xml /
 * method.xml). See
 * https://developer.android.com/develop/ui/views/touch-and-input/creating-input-method
 * and the InputMethodService / InputConnection references.
 *
 * The loop mirrors the Linux engines (fcitx5 eim.cpp / ibus main.cpp):
 *   tap  -> Core.feedKey / toneOrSpace        (composing; auto zh/en segmenter)
 *   NEED_LIVE -> Core.decodeLive off-main     -> setComposingText(getLive())    (免選字)
 *   ↓    -> Core.beginConvert                 -> candidate window (ChoosingCore)
 *   字/詞 tap -> Core.pickSegment / pickPhrase -> re-score, repaint preedit
 *   ⏎    -> Core.confirmChoosing / commitLive  -> commitText(getCommit())
 *   ⌫ ⏎ space tone、？，。 = keyboard keys; Shift(中/英) = English passthrough.
 *
 * E-ink: the keyboard is flat, high-contrast, animation-free (see KeyboardView);
 * we never run the model on the UI thread — heavy Core calls suspend onto
 * Dispatchers.Default inside [Core] and we repaint back on Main.
 */
class SlothingImeService : InputMethodService(),
    KeyboardView.Listener, CandidateBar.Listener {

    private companion object {
        const val TAG = "SlothingIME"
        const val MODEL = "slothing/model_quantized.onnx"
        const val SYL_VOCAB = "slothing/syl_vocab.json"
        const val CHAR2ID = "slothing/char2id.json"
        const val TABLE = "slothing/phonetic_table.tsv"
        const val THREADS = 2
        val TONE_KEYS = setOf('3', '4', '6', '7')     // ime.js TONEK; feed via toneOrSpace
        const val CTX_CHARS = 64                       // left-of-caret context handed to the LM
    }

    private val core = Core()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var keyboard: KeyboardView
    private lateinit var candidateBar: CandidateBar
    private var english = false

    override fun onCreate() {
        super.onCreate()
        val ok = core.init(readAsset(MODEL), readAsset(SYL_VOCAB), readAsset(CHAR2ID), readAsset(TABLE), THREADS)
        Log.i(TAG, "core.init = $ok, state = ${core.state()}")
        selfTest()
    }

    /**
     * On-device decode validation: drive the native Core with known ASCII key
     * sequences (the same routing the soft keyboard uses) and log the committed
     * sentence vs the expected one. Exercises the whole port — segmenter, the
     * shared state machine, and the in-process OnnxDecoder — end to end.
     * Readable via `adb logcat -s SlothingIME`.
     */
    private fun selfTest() = scope.launch {
        val cases = listOf(
            "su3cl3" to "你好",   // ㄋㄧˇ ㄏㄠˇ
            "ji3y94" to "我在",   // ㄨㄛˇ ㄗㄞˋ
            "5k4"    to "這",     // ㄓㄜˋ
            // ㄓㄨˋ ㄧㄣ: raw model picks 注因 (因 slightly > 音 for isolated
            // ㄧㄣ). The desktop shows 注音 only because it LEARNED it
            // (learn.tsv: c ㄧㄣ 音 / p ㄓㄨˋ ㄧㄣ 注音, +6/+8 bonus). A fresh
            // install has no learn store — the port's learn logic is present,
            // just empty. So the honest fresh-install expectation is 注因.
            "5j4up"  to "注因",   // 注音 after personalization / seed-learn
        )
        for ((keys, expect) in cases) {
            core.reset()
            for (c in keys) {
                if (c == ' ' || c in TONE_KEYS) core.toneOrSpace(c.code)
                else core.feedKey(c.code)
            }
            if (!core.commitLive()) core.beginConvert(-1, /* commitDirect = */ true)
            val got = core.getCommit()
            val mark = if (got == expect) "OK  " else "FAIL"
            Log.i(TAG, "selftest $mark keys='$keys' got='$got' expect='$expect'")
            core.reset()
        }
    }.let {}

    // Created once and reused (guide: cache the input view).
    override fun onCreateInputView(): View {
        candidateBar = CandidateBar(this).apply {
            listener = this@SlothingImeService
            visibility = View.GONE
        }
        keyboard = KeyboardView(this).apply {
            listener = this@SlothingImeService
            setEnglish(english)
        }
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(getColor(R.color.eink_bg))
            addView(
                candidateBar,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ),
            )
            addView(
                keyboard,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        core.reset()
        english = false
        core.setEnglishMode(false)
        if (::keyboard.isInitialized) keyboard.setEnglish(false)
        setContextFromField()
        hideBar()
        ic()?.finishComposingText()
    }

    // E-ink / large screen: never take the whole screen; keep the field visible.
    override fun onEvaluateFullscreenMode(): Boolean = false

    // ===== KeyboardView.Listener ==========================================

    override fun onKey(ascii: Char) {
        val outcome = if (ascii == ' ' || ascii in TONE_KEYS) {
            core.toneOrSpace(ascii.code)      // tone (3/4/6/7) or space finalizes the run
        } else {
            core.feedKey(ascii.code)          // bopomofo / punctuation / digit / latin
        }
        applyOutcome(outcome)
    }

    override fun onBackspace() {
        if (core.state() == Core.State.CHOOSING) {
            core.escapeChoosing()
            if (core.state() == Core.State.CHOOSING) paintChoosing() else paintComposingRaw()
        } else {
            applyOutcome(core.backspace())
        }
    }

    override fun onEnter() = commitOrConfirm()

    override fun onOpenChoosing() = openChoosing()

    override fun onToggleEnglish() {
        english = !english
        core.setEnglishMode(english)
        keyboard.setEnglish(english)
    }

    // Opens the 微軟新注音-style ` symbol menu. Symbol selection is wired once the
    // Core exposes symbol accessors; for now the menu opens and number keys pick.
    override fun onToggleSymbols() {
        applyOutcome(core.feedKey('`'.code))
    }

    // ===== CandidateBar.Listener ==========================================

    override fun onPickCandidate(index: Int) = scope.launch {
        core.pickSegment(index)               // chewing: pick CLOSES the char window
        if (core.getCommit().isNotEmpty()) commitDrain() else paintChoosing()
    }.let {}

    override fun onPickPhrase(phrase: Phrase) = scope.launch {
        core.pickPhrase(phrase.start, phrase.text)
        if (core.getCommit().isNotEmpty()) commitDrain() else paintChoosing()
    }.let {}

    // ===== loop ============================================================

    private fun applyOutcome(outcome: Core.KeyOutcome) {
        when (outcome) {
            Core.KeyOutcome.NEED_LIVE -> scope.launch {
                setContextFromField()
                if (!core.refreshLiveFast()) core.decodeLive()   // heavy: off-main inside Core
                ic()?.setComposingText(core.getLive(), 1)
                hideBar()
            }
            Core.KeyOutcome.CONSUMED -> paintComposingRaw()
            Core.KeyOutcome.COMMITTED -> commitDrain()
            Core.KeyOutcome.IGNORED -> { /* soft keyboard: nothing to pass through */ }
        }
    }

    private fun openChoosing() = scope.launch {
        setContextFromField()
        val opened = core.beginConvert(/* focus = */ -1, /* commitDirect = */ false)
        if (opened) {
            core.ensurePhrases()              // pre-warm the 詞 chips
            paintChoosing()
        } else {
            commitDrain()                     // single clean decode / failure -> drain
            core.getNotice().takeIf { it.isNotEmpty() }?.let { Log.i(TAG, "convert: $it") }
        }
    }.let {}

    private fun commitOrConfirm() = scope.launch {
        when (core.state()) {
            Core.State.CHOOSING -> {
                if (core.confirmChoosing()) commitDrain() else paintChoosing()
            }
            else -> {
                if (!core.commitLive()) {     // Enter on an already-shown live sentence
                    setContextFromField()
                    core.beginConvert(-1, /* commitDirect = */ true)
                }
                commitDrain()
            }
        }
    }.let {}

    // ===== painting ========================================================

    private fun paintComposingRaw() {
        ic()?.setComposingText(core.getPreedit().text, 1)
        hideBar()
    }

    private fun paintChoosing() {
        val ic = ic() ?: return
        val pe = core.getPreedit()
        ic.setComposingText(highlight(pe.text, pe.hlStart, pe.hlEnd), 1)
        val shown = candidateBar.render(core.getCandidates(), core.getPhrases())
        candidateBar.visibility = if (shown) View.VISIBLE else View.GONE
    }

    private fun commitDrain() {
        val ic = ic() ?: return
        val text = core.getCommit()
        if (text.isNotEmpty()) ic.commitText(text, 1) else ic.finishComposingText()
        hideBar()
    }

    /** Reverse-video the focused segment (BackgroundColorSpan) — matches the
     *  IBus engine's focused-segment highlight. hl* are codepoint indices. */
    private fun highlight(text: String, hlStart: Int, hlEnd: Int): CharSequence {
        if (hlStart < 0 || hlEnd <= hlStart) return text
        return try {
            val a = text.offsetByCodePoints(0, hlStart)
            val b = text.offsetByCodePoints(0, hlEnd)
            SpannableString(text).apply {
                val bg = getColor(R.color.eink_focus_bg)
                val fg = getColor(R.color.eink_focus_ink)
                setSpan(BackgroundColorSpan(bg), a, b, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                setSpan(ForegroundColorSpan(fg), a, b, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            }
        } catch (e: IndexOutOfBoundsException) {
            text
        }
    }

    private fun hideBar() {
        if (::candidateBar.isInitialized) {
            candidateBar.clear()
            candidateBar.visibility = View.GONE
        }
    }

    private fun setContextFromField() {
        val before = ic()?.getTextBeforeCursor(CTX_CHARS, 0)?.toString().orEmpty()
        core.setContext(before)
    }

    private fun ic(): InputConnection? = currentInputConnection

    // ===== lifecycle =======================================================

    override fun onFinishInput() {
        core.flush()
        core.getCommit().takeIf { it.isNotEmpty() }?.let { ic()?.commitText(it, 1) }
        hideBar()
        super.onFinishInput()
    }

    override fun onDestroy() {
        scope.cancel()
        core.destroy()
        super.onDestroy()
    }

    private fun readAsset(name: String): ByteArray = assets.open(name).use { it.readBytes() }
}
