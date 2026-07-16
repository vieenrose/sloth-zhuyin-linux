package com.sloth.ime

import android.inputmethodservice.InputMethodService
import android.text.SpannableString
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.text.style.ForegroundColorSpan
import android.util.Log
import android.view.KeyEvent
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
 * Sloth IME InputMethodService — the Dàqiān bopomofo soft keyboard layered over
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
class SlothImeService : InputMethodService(),
    KeyboardView.Listener, CandidateBar.Listener {

    private companion object {
        const val TAG = "SlothIME"
        const val MODEL = "sloth/slothe-t-25m.gguf"   // ternary GGUF encoder (ggml/libslothe)
        const val SYL_VOCAB = "sloth/syl_vocab.json"
        const val CHAR2ID = "sloth/char2id.json"
        const val TABLE = "sloth/phonetic_table.tsv"
        const val ASSOC = "sloth/assoc_tc.tsv"
        const val THREADS = 2
        const val PREDICT_CHAIN_MAX = 5   // librime-predict-style chain cap
        val TONE_KEYS = setOf('3', '4', '6', '7')     // ime.js TONEK; feed via toneOrSpace
        const val CTX_CHARS = 64                       // left-of-caret context handed to the LM
        // 符 strip: the 常用 row of the desktop ` symbol menu
        val SYMBOLS = arrayOf(
            "，", "、", "。", "？", "！", "；", "：", "…", "—", "～",
            "「", "」", "『", "』", "（", "）", "《", "》",
        )
    }

    private val core = Core()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var keyboard: KeyboardView
    private lateinit var candidateBar: CandidateBar
    private var english = false

    override fun onCreate() {
        super.onCreate()
        val ok = core.init(
            // GGUF is loaded by path (slothe_load opens the file), so copy the
            // asset to app cache once and hand the native side that path.
            copyAssetToCache(MODEL), readAsset(SYL_VOCAB), readAsset(CHAR2ID),
            readAsset(TABLE), THREADS,
            // persistent personalization, same format as the desktop learn.tsv
            filesDir.resolve("learn.tsv").absolutePath,
            // 聯想 dictionary + personal bigram store
            readAsset(ASSOC), filesDir.resolve("assoc_user.tsv").absolutePath,
        )
        Log.i(TAG, "core.init = $ok, state = ${core.state()}")
        selfTest()
    }

    /**
     * On-device decode validation: drive the native Core with known ASCII key
     * sequences (the same routing the soft keyboard uses) and log the committed
     * sentence vs the expected one. Exercises the whole port — segmenter, the
     * shared state machine, and the in-process OnnxDecoder — end to end.
     * Readable via `adb logcat -s SlothIME`.
     */
    private fun selfTest() = scope.launch(Dispatchers.Default) {
        // debuggable builds only, off the UI thread: 230 ONNX forwards would
        // ANR the service, and the staged commits must not race live typing
        // debug builds, OR release when `settings put global sloth_bench 1` is set
        // (adb-only global) — lets us capture BENCH_LAT on real hardware without a
        // debug build (which ANRs the BOOX on model extraction).
        val debuggable = (applicationInfo.flags and
            android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0
        val benchFlag = try {
            android.provider.Settings.Global.getInt(contentResolver, "sloth_bench", 0) == 1
        } catch (_: Exception) { false }
        if (!debuggable && !benchFlag) return@launch
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
        benchmark()
    }.let {}

    /**
     * On-device accuracy benchmark: decode the 230-case 免選字 reference set
     * (bopomofo <TAB> expected) straight through the OnnxDecoder and score
     * whole-sentence top-1 exact match — the same metric as the desktop daemon.
     * Writes every case to filesDir/bench_android.tsv (pull via `adb shell
     * run-as com.sloth.ime cat files/bench_android.tsv`) and logs the total.
     */
    private fun benchmark() {
        val lines = try { assets.open("sloth/reference_mspy.tsv").bufferedReader().readLines() }
                    catch (e: Exception) { Log.i(TAG, "bench: no reference set"); return }
        var ok = 0; var tot = 0
        val sb = StringBuilder()
        for (line in lines) {
            if (line.isBlank() || line.startsWith("#") || !line.contains('\t')) continue
            val (syl, exp) = line.split('\t', limit = 2)
            val got = core.decodeBest(syl)
            tot++; if (got == exp) ok++
            sb.append(syl).append('\t').append(exp).append('\t').append(got).append('\n')
        }
        try { filesDir.resolve("bench_android.tsv").writeText(sb.toString()) } catch (_: Exception) {}
        Log.i(TAG, "BENCH android on-device: $ok/$tot = ${if (tot>0) 100*ok/tot else 0}% top-1 免選字")

        // On-device (ARM) inference-latency micro-benchmark: time repeated
        // whole-sentence decodes of one fixed 6-syllable input. Reports median
        // ms so the 4M vs 10M model swap can be compared on real hardware
        // (x86 said int8>int4; ARM must confirm). Logged as BENCH_LAT.
        val probe = "su3cl3ji3y94"    // 你好我在 — 6 syllables, representative
        repeat(10) { core.decodeBest(probe) }             // warm the session
        val samples = LongArray(40)
        for (i in samples.indices) {
            val t0 = System.nanoTime()
            core.decodeBest(probe)
            samples[i] = System.nanoTime() - t0
        }
        samples.sort()
        val medMs = samples[samples.size / 2] / 1e6
        val p90Ms = samples[(samples.size * 9) / 10] / 1e6
        Log.i(TAG, "BENCH_LAT decode(6-syl) median=%.2fms p90=%.2fms".format(medMs, p90Ms))
    }

    // Created once and reused (guide: cache the input view).
    override fun onCreateInputView(): View {
        candidateBar = CandidateBar(this).apply {
            listener = this@SlothImeService
            visibility = View.INVISIBLE
        }
        keyboard = KeyboardView(this).apply {
            listener = this@SlothImeService
            setEnglish(english)
            // demo recording can stretch the hardware-key echo so a screencap
            // catches the pressed key (`adb shell settings put global
            // sloth_flash_ms 900`). Gated behind an adb/system-only global
            // setting, so it stays inert (default 250ms) in normal use — safe
            // to honour in release builds too (debug builds ANR-extract the
            // model too slowly to record on the e-ink device).
            android.provider.Settings.Global.getString(
                contentResolver, "sloth_flash_ms",
            )?.toLongOrNull()?.let { flashHoldMs = it }
        }
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Skin.color(this@SlothImeService, R.color.eink_bg))
            // fixed-height slot + INVISIBLE (not GONE) so toggling the strip
            // never shifts the keys under the user's finger (e-ink repaint)
            addView(
                candidateBar,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    (56 * resources.displayMetrics.density).toInt(),
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

    /** Password/number fields: force English passthrough + no LM context. */
    private var privateField = false

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        // chewing rule: never DROP a composition — commit it (a bare reset on
        // restart, e.g. rotation, would silently lose the buffer)
        core.flush()
        core.getCommit().takeIf { it.isNotEmpty() }?.let { ic()?.commitText(it, 1) }

        // inputType: passwords must not reach the LM context or learn store;
        // number/phone/datetime fields need literal digits, not bopomofo
        val it = info?.inputType ?: 0
        val cls = it and android.text.InputType.TYPE_MASK_CLASS
        val variation = it and android.text.InputType.TYPE_MASK_VARIATION
        val password = cls == android.text.InputType.TYPE_CLASS_TEXT &&
            (variation == android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD ||
             variation == android.text.InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD ||
             variation == android.text.InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD)
        val nonText = cls == android.text.InputType.TYPE_CLASS_NUMBER ||
            cls == android.text.InputType.TYPE_CLASS_PHONE ||
            cls == android.text.InputType.TYPE_CLASS_DATETIME
        privateField = password
        english = password || nonText
        core.setEnglishMode(english)
        if (::keyboard.isInitialized) keyboard.setEnglish(english)
        setContextFromField()
        symbolsShowing = false
        core.clearPredictions()   // stale predictions must not cross fields
        predicting = false
        hideBar()
        ic()?.finishComposingText()
    }

    // E-ink / large screen: never take the whole screen; keep the field visible.
    override fun onEvaluateFullscreenMode(): Boolean = false

    // ===== KeyboardView.Listener ==========================================

    override fun onKey(ascii: Char) {
        // Choosing window open: digits pick the numbered candidates; anything
        // else must not leak into the composing buffer behind the window.
        if (core.state() == Core.State.CHOOSING) {
            if (ascii in '1'..'9') onPickCandidate(ascii - '1')
            return
        }
        // Symbol strip open: keys close it and type normally.
        if (symbolsShowing) hideSymbols()
        // First phonetic key atomically replaces predictions (convention).
        if (predicting) { hideBar(); predicting = false }
        if (english) {
            // English mode: EVERY key goes through feedKey's passthrough
            // branch (space, digits 3/4/6/7 included — toneOrSpace has no
            // English branch and would swallow them). The Chinese-punct key
            // codes revert to their ASCII meaning.
            val ch = when (ascii) {
                '<' -> ','; '>' -> '.'; '\\' -> '/'
                else -> ascii
            }
            val outcome = core.feedKey(ch.code)
            if (outcome == Core.KeyOutcome.IGNORED) ic()?.commitText(ch.toString(), 1)
            else applyOutcome(outcome)
            return
        }
        val outcome = if (ascii == ' ' || ascii in TONE_KEYS) {
            core.toneOrSpace(ascii.code)      // tone (3/4/6/7) or space finalizes the run
        } else {
            core.feedKey(ascii.code)          // bopomofo / punctuation / digit / latin
        }
        if (outcome == Core.KeyOutcome.IGNORED) {
            // Empty buffer: the key belongs to the app (space, stray ASCII) —
            // a soft keyboard must never swallow it (Gboard/iOS convention).
            ic()?.commitText(ascii.toString(), 1)
            return
        }
        applyOutcome(outcome)
    }

    override fun onBackspace() {
        if (core.state() == Core.State.CHOOSING) {
            core.escapeChoosing()
            if (core.state() == Core.State.CHOOSING) paintChoosing() else paintComposingRaw()
        } else {
            val outcome = core.backspace()
            if (outcome == Core.KeyOutcome.IGNORED) {
                // Empty buffer: delete committed text in the app. KEYCODE_DEL
                // (not deleteSurroundingText) so the editor handles surrogate
                // pairs, selections, and key listeners itself.
                sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL)
            } else {
                applyOutcome(outcome)
            }
        }
    }

    override fun onEnter() = commitOrConfirm()

    override fun onOpenChoosing() = openChoosing()

    override fun onToggleEnglish() {
        english = !english
        core.setEnglishMode(english)
        // hardware-only sessions may toggle before the input view exists
        if (::keyboard.isInitialized) keyboard.setEnglish(english)
    }

    // 符: toggle a symbol strip in the candidate bar (the 常用 row of the
    // desktop ` menu). Tap a symbol -> commit (composing) or insert directly.
    private var symbolsShowing = false

    override fun onToggleSymbols() {
        if (symbolsShowing) { hideSymbols(); return }
        symbolsShowing = true
        candidateBar.renderSuggestions(SYMBOLS)
        candidateBar.visibility = View.VISIBLE
    }

    private fun hideSymbols() {
        symbolsShowing = false
        showSuggestions()   // restore whatever the composing strip should show
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

    override fun onPickLastWord(ch: String) {
        if (core.pickLastWord(ch)) {
            ic()?.setComposingText(core.getComposingBopo(), 1)  // iOS model: field stays bopomofo
            showSuggestions()          // refresh the selected chip
        }
    }

    override fun onPickPrediction(text: String) {
        ic()?.commitText(text, 1)
        core.predicted(text)           // learn transition + move the tail
        predictChain++
        if (predictChain < PREDICT_CHAIN_MAX) showPredictions() else hideBar()
    }

    override fun onMoveFocus(dir: Int) = scope.launch {
        core.moveFocus(dir)
        core.ensurePhrases()                  // pre-warm 詞 for the new focus (worker)
        paintChoosing()
    }.let {}

    override fun onPickSuggestion(sentence: String) {
        if (symbolsShowing) {
            hideSymbols()
            applyOutcome(core.insertSymbol(sentence))   // literal token / direct commit
            return
        }
        core.commitSentence(sentence)         // tap-to-commit (Gboard convention)
        commitDrain()
    }

    // ===== hardware (Bluetooth/USB) keyboard ===============================
    //
    // BOOX users pair BT keyboards; physical keys must drive the SAME state
    // machine as the soft keys, mirroring the desktop engines (fcitx5
    // eim.cpp keyEvent / ibus main.cpp processKey): ↓ opens Choosing, ←→
    // move the preedit cursor / choosing highlight, Esc is two-level, Enter
    // commits/confirms, lone Shift toggles English, ⇧1-9 picks 詞/聯想.
    // Only keys with IME meaning are consumed — everything else falls to
    // super so app shortcuts and navigation keep working.

    /** Desktop shiftAlone_: armed on Shift-down, disarmed by any other key;
     *  a clean press+release toggles 中/英. */
    private var shiftAlone = false

    /** Echo a consumed hardware key on the on-screen keyboard so the user can
     *  see WHICH key was hit (specials: '\b' ⌫, '\n' ⏎, ' ' space). */
    private fun flash(ascii: Char) {
        if (::keyboard.isInitialized) keyboard.flashKey(ascii)
    }

    /** Keep the keyboard + candidate bar visible while a hardware keyboard
     *  is connected — the default policy hides the input view when
     *  config.keyboard != NOKEYS, which would remove the candidate window
     *  (choosing UI, 聯想 strip) entirely. */
    override fun onEvaluateInputViewShown(): Boolean {
        super.onEvaluateInputViewShown()
        return true
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (keyCode == KeyEvent.KEYCODE_SHIFT_LEFT ||
            keyCode == KeyEvent.KEYCODE_SHIFT_RIGHT) {
            shiftAlone = true
            return super.onKeyDown(keyCode, event)   // app keeps the modifier
        }
        shiftAlone = false
        // real modifier chords belong to the app (desktop parity)
        if (event.isCtrlPressed || event.isAltPressed || event.isMetaPressed) {
            return super.onKeyDown(keyCode, event)
        }
        if (handleHardwareKey(keyCode, event)) return true
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        if ((keyCode == KeyEvent.KEYCODE_SHIFT_LEFT ||
             keyCode == KeyEvent.KEYCODE_SHIFT_RIGHT) && shiftAlone) {
            shiftAlone = false
            onToggleEnglish()   // lone Shift = 中/英 (desktop convention)
            if (::keyboard.isInitialized) keyboard.flashEnglishToggle()
            // the release still passes through: the app saw the press
        }
        return super.onKeyUp(keyCode, event)
    }

    /** Route one hardware key through the soft-keyboard code paths.
     *  Returns true iff the key was consumed by the IME. */
    private fun handleHardwareKey(keyCode: Int, event: KeyEvent): Boolean {
        val choosing = core.state() == Core.State.CHOOSING
        when (keyCode) {
            KeyEvent.KEYCODE_ENTER, KeyEvent.KEYCODE_NUMPAD_ENTER -> {
                if (!choosing && core.getPreedit().text.isEmpty()) {
                    return false          // empty buffer: Enter belongs to the app
                }
                flash('\n')
                commitOrConfirm()
                return true
            }
            KeyEvent.KEYCODE_DEL -> {     // system auto-repeat re-enters here
                if (choosing) { flash('\b'); onBackspace(); return true }
                val outcome = core.backspace()
                if (outcome == Core.KeyOutcome.IGNORED) return false
                flash('\b')
                applyOutcome(outcome)
                return true
            }
            KeyEvent.KEYCODE_ESCAPE -> {
                if (choosing) {           // two-level: close window, then cancel
                    core.escapeChoosing()
                    if (core.state() == Core.State.CHOOSING) paintChoosing()
                    else paintComposingRaw()
                    return true
                }
                // Composing: the session exposes no clear-pending-raw surface
                // (desktop Esc clears rawKeys only) — swallow to protect the
                // sentence, pass through when nothing is composed.
                return core.getPreedit().text.isNotEmpty()
            }
            KeyEvent.KEYCODE_DPAD_DOWN -> {
                if (choosing) {
                    if (core.getCandidates().open) core.moveHighlight(1)
                    else core.reopen()    // desktop: ↓ reopens the char window
                    paintChoosing()
                    return true
                }
                if (core.getPreedit().text.isEmpty()) return false
                openChoosing()            // desktop: ↓ starts conversion at cursor
                return true
            }
            KeyEvent.KEYCODE_DPAD_UP -> {
                if (choosing) {
                    if (core.getCandidates().open) {
                        core.moveHighlight(-1)
                        paintChoosing()
                    }
                    return true           // swallow while the window shows
                }
                return false
            }
            KeyEvent.KEYCODE_DPAD_LEFT, KeyEvent.KEYCODE_DPAD_RIGHT -> {
                val d = if (keyCode == KeyEvent.KEYCODE_DPAD_RIGHT) 1 else -1
                if (choosing) {
                    if (core.getCandidates().open) {
                        core.moveHighlight(d)   // desktop: ←→ move the highlight
                        paintChoosing()
                    } else {
                        onMoveFocus(d)          // desktop: ←→ move segment focus
                    }
                    return true
                }
                val outcome = core.moveCursor(d)  // preedit caret (NEED_LIVE)
                if (outcome == Core.KeyOutcome.IGNORED) return false
                applyOutcome(outcome)
                return true
            }
            KeyEvent.KEYCODE_MOVE_HOME, KeyEvent.KEYCODE_MOVE_END -> {
                if (choosing) return true
                val outcome = core.moveCursor(
                    if (keyCode == KeyEvent.KEYCODE_MOVE_HOME) -2 else 2,
                )
                if (outcome == Core.KeyOutcome.IGNORED) return false
                applyOutcome(outcome)
                return true
            }
        }

        // ⇧1-9: pick a 詞 in Choosing / a 聯想 chip while predicting (desktop)
        if (event.isShiftPressed &&
            keyCode in KeyEvent.KEYCODE_1..KeyEvent.KEYCODE_9) {
            val i = keyCode - KeyEvent.KEYCODE_1
            if (choosing) {
                val phrases = core.getPhrases()
                if (i < phrases.size) onPickPhrase(phrases[i])
                return true               // desktop swallows out-of-range ⇧digits
            }
            if (predicting) {
                val preds = core.getPredictions()
                if (i < preds.size) { onPickPrediction(preds[i]); return true }
                // out of range: fall through — the strip dismisses below
            }
        }

        // printable ASCII (letters/digits/punct, incl. shifted like <>?!)
        val uni = event.unicodeChar
        if (uni in 0x20..0x7e) return handleHardAscii(uni.toChar())

        // anything else while the choosing window is open: swallow keyboard
        // noise (desktop parity) — but never system keys (Back/Home/volume)
        return choosing && !event.isSystem
    }

    /** The onKey(ascii) route for a hardware key: identical guards, except an
     *  IGNORED outcome returns false so the ORIGINAL event reaches the app
     *  (instead of the soft keyboard's synthesized commitText). */
    private fun handleHardAscii(ascii: Char): Boolean {
        if (core.state() == Core.State.CHOOSING) {
            if (ascii in '1'..'9') { flash(ascii); onPickCandidate(ascii - '1') }
            return true    // nothing leaks behind the window (onKey parity)
        }
        if (symbolsShowing) hideSymbols()
        if (predicting) { hideBar(); predicting = false }
        if (english) {
            // english mode: feedKey's passthrough branch stages the char
            // (flushes composed text first, preserving order — desktop parity)
            val outcome = core.feedKey(ascii.code)
            if (outcome == Core.KeyOutcome.IGNORED) return false
            flash(ascii)
            applyOutcome(outcome)
            return true
        }
        val outcome = if (ascii == ' ' || ascii in TONE_KEYS) {
            core.toneOrSpace(ascii.code)
        } else {
            core.feedKey(ascii.code)
        }
        if (outcome == Core.KeyOutcome.IGNORED) return false
        flash(ascii)
        applyOutcome(outcome)
        return true
    }

    // ===== loop ============================================================

    private fun applyOutcome(outcome: Core.KeyOutcome) {
        when (outcome) {
            Core.KeyOutcome.NEED_LIVE -> scope.launch {
                setContextFromField()
                if (!core.refreshLiveFast()) core.decodeLive()   // heavy: off-main inside Core
                // Touch/iOS model (matches iPhone 注音): the FIELD shows the raw
                // space-separated bopomofo you typed (verifiable); the sentence
                // CONVERSION lives in the candidate bar (getLiveSuggestions).
                paintComposingRaw()
                showSuggestions()
            }
            Core.KeyOutcome.CONSUMED -> paintComposingRaw()
            Core.KeyOutcome.COMMITTED -> commitDrain()
            Core.KeyOutcome.IGNORED -> { /* handled at the call sites (pass-through) */ }
        }
    }

    /** Refresh the always-visible composing-time suggestion strip: per-position
     *  candidates for the last word (cursor is at the end while typing). Correct
     *  earlier chars by tapping them / the ↓ 字·詞 window — no whole-sentence 句. */
    private fun showSuggestions() {
        if (!::candidateBar.isInitialized) return
        // Touch/iOS model: the candidate bar LEADS with whole-sentence
        // conversions (getLiveSuggestions, [0] = best) — tap to commit. Falls
        // back to last-word char candidates for a lone word / mixed input.
        val sents = core.getLiveSuggestions()
        val shown = if (sents.isNotEmpty()) candidateBar.renderSuggestions(sents)
            else candidateBar.renderComposing(core.getLastWordCands(), core.getLastWordCurrent())
        candidateBar.visibility = if (shown) View.VISIBLE else View.INVISIBLE
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
                if (core.getPreedit().text.isEmpty()) {
                    sendEnterToApp()          // empty buffer: Enter belongs to the app
                    return@launch
                }
                if (!core.commitLive()) {     // Enter on an already-shown live sentence
                    setContextFromField()
                    core.beginConvert(-1, /* commitDirect = */ true)
                }
                commitDrain()
            }
        }
    }.let {}

    /** Enter with nothing composed: perform the field's IME action (search/send/…)
     *  or type a newline — exactly what the stock keyboards do.
     *  sendDefaultEditorAction implements the imeOptions contract for us. */
    private fun sendEnterToApp() {
        if (!sendDefaultEditorAction(true)) {
            sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
        }
    }

    // ===== painting ========================================================

    private fun paintComposingRaw() {
        // Touch/iOS model: field shows raw bopomofo, never the eager conversion.
        ic()?.setComposingText(core.getComposingBopo(), 1)
    }

    private fun paintChoosing() {
        val ic = ic() ?: return
        val pe = core.getPreedit()
        ic.setComposingText(highlight(pe.text, pe.hlStart, pe.hlEnd), 1)
        if (!::candidateBar.isInitialized) return   // hardware-only session
        val shown = candidateBar.render(core.getCandidates(), core.getPhrases())
        candidateBar.visibility = if (shown) View.VISIBLE else View.INVISIBLE
    }

    private fun commitDrain() {
        val ic = ic() ?: return
        val text = core.getCommit()
        if (text.isNotEmpty()) {
            ic.commitText(text, 1)
            predictChain = 0
            showPredictions()   // strip flips to 聯想 (mobile convention)
        } else {
            ic.finishComposingText()
            hideBar()
        }
    }

    // ---- 聯想 prediction strip (empty buffer, after a commit) --------------
    private var predicting = false
    private var predictChain = 0

    private fun showPredictions() {
        if (!::candidateBar.isInitialized) return
        val preds = core.getPredictions()
        predicting = preds.isNotEmpty()
        if (predicting) {
            candidateBar.renderPredictions(preds)
            candidateBar.visibility = View.VISIBLE
        } else {
            hideBar()
        }
    }

    /** Reverse-video the focused segment (BackgroundColorSpan) — matches the
     *  IBus engine's focused-segment highlight. hl* are codepoint indices. */
    private fun highlight(text: String, hlStart: Int, hlEnd: Int): CharSequence {
        if (hlStart < 0 || hlEnd <= hlStart) return text
        return try {
            val a = text.offsetByCodePoints(0, hlStart)
            val b = text.offsetByCodePoints(0, hlEnd)
            SpannableString(text).apply {
                val bg = Skin.color(this@SlothImeService, R.color.eink_focus_bg)
                val fg = Skin.color(this@SlothImeService, R.color.eink_focus_ink)
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
            candidateBar.visibility = View.INVISIBLE
        }
    }

    private fun setContextFromField() {
        // never feed password text into the model's context channel
        val before = if (privateField) "" else
            ic()?.getTextBeforeCursor(CTX_CHARS, 0)?.toString().orEmpty()
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

    /**
     * Copy an (uncompressed) asset to app cache and return its absolute path.
     * The GGUF encoder is loaded from a file by libslothe/ggml, so it needs a
     * real filesystem path rather than a byte buffer. Re-copies only when the
     * cached file is missing or a different size (cheap, once per model bump).
     */
    private fun copyAssetToCache(name: String): String {
        val out = cacheDir.resolve(name.substringAfterLast('/'))
        val want = assets.openFd(name).use { it.length }.takeIf { it > 0 }
        if (!out.exists() || (want != null && out.length() != want)) {
            assets.open(name).use { input -> out.outputStream().use { input.copyTo(it) } }
        }
        return out.absolutePath
    }
}
