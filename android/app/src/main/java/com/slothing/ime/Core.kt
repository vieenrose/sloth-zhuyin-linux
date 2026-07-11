package com.slothing.ime

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Kotlin front door to libslothing.so — the same frontend-free IME state
 * machine the fcitx5/IBus engines use (engine/common/core.h), driven on-device
 * with an ONNX decoder instead of the slothingd Unix socket.
 *
 * There is exactly one [Core] per InputMethodService instance; it owns a native
 * SlothingSession behind [handle]. Call [init] once, then drive the IME loop.
 *
 * ## Threading
 * The native session is internally locked (mutex + generation), so the split is
 * only about not blocking the UI thread on an ONNX forward:
 *
 *  - **Main-safe (cheap):** feedKey, toneOrSpace, backspace, moveCursor,
 *    getPreedit, getCandidates, getPhrases, getCommit, moveHighlight, moveFocus,
 *    reopen, closeCandList, escapeChoosing, refreshLiveFast, reset, flush,
 *    setContext, setEnglishMode, setFullWidth, commitLive.
 *  - **Off-main (heavy — one ONNX forward each), wrapped in
 *    withContext(Dispatchers.Default):** decodeLive, beginConvert, ensurePhrases,
 *    pickSegment, pickPhrase, confirmChoosing.
 *
 * The suspend wrappers below already hop to Default and back; call them from the
 * service's CoroutineScope, then repaint on the main thread. Heavy calls carry a
 * generation guard natively: if the user typed more while a decode was in
 * flight, the stale result is dropped and the wrapper returns false.
 */
class Core {

    private var handle: Long = 0L

    /** Values mirror native slothing::KeyOutcome. */
    enum class KeyOutcome { IGNORED, CONSUMED, NEED_LIVE, COMMITTED;
        companion object { fun of(i: Int) = values()[i] } }

    /** Values mirror native SlothingSession::State. */
    enum class State { COMPOSING, CONVERTING, CHOOSING;
        companion object { fun of(i: Int) = values()[i] } }

    /**
     * @param model     the .onnx graph bytes (read from assets).
     * @param sylVocab  syl_vocab.json bytes (bopomofo syllable -> id).
     * @param char2id   char2id.json bytes (Han char -> logit column).
     * @param table     phonetic_table.tsv bytes (syllable \t chars per line).
     * @param threads   intra-op ORT threads (1–2 on a phone; 0 = ORT decides).
     * @param learnPath on-device learn store (persisted corrections; "" = in-memory only).
     * @param assocTsv  聯想 dictionary bytes (assoc_tc.tsv; empty = predictions off).
     * @param assocUserPath personal bigram store ("" = in-memory only).
     */
    fun init(model: ByteArray, sylVocab: ByteArray, char2id: ByteArray,
             table: ByteArray, threads: Int, learnPath: String = "",
             assocTsv: ByteArray = ByteArray(0), assocUserPath: String = ""): Boolean {
        handle = nativeInit(model, sylVocab, char2id, table, threads, learnPath,
                            assocTsv, assocUserPath)
        return handle != 0L && nativeReady(handle)
    }

    fun destroy() {
        if (handle != 0L) { nativeDestroy(handle); handle = 0L }
    }

    /** Debug/benchmark: space-separated bopomofo syllables -> best sentence. */
    fun decodeBest(syllables: String): String = nativeDecodeBest(handle, syllables)

    // ---- context & modes ---------------------------------------------------
    /** Left-of-caret text for the LM (InputConnection.getTextBeforeCursor). */
    fun setContext(text: String) = nativeSetContext(handle, text)
    fun setEnglishMode(on: Boolean) = nativeSetEnglishMode(handle, on)
    fun setFullWidth(on: Boolean) = nativeSetFullWidth(handle, on)

    // ---- composing input (main-safe) --------------------------------------
    /** A printable ASCII key from the soft keyboard. */
    fun feedKey(codePoint: Int) = KeyOutcome.of(nativeFeedKey(handle, codePoint))
    /** A tone key (3/4/6/7) or space — finalizes the current run. */
    fun toneOrSpace(codePoint: Int) = KeyOutcome.of(nativeToneOrSpace(handle, codePoint))
    fun backspace() = KeyOutcome.of(nativeBackspace(handle))
    /** dir: LEFT=-1, RIGHT=1, HOME=-2, END=2. */
    fun moveCursor(dir: Int) = KeyOutcome.of(nativeMoveCursor(handle, dir))

    fun reset() = nativeReset(handle)
    /** Focus-out / onFinishInput: stage pending text for commit, then clear. */
    fun flush() = nativeFlush(handle)

    // ---- live (modeless) conversion ---------------------------------------
    /** Cheap: resolves empty/all-English live display; false => run decodeLive. */
    fun refreshLiveFast() = nativeRefreshLiveFast(handle)
    /** HEAVY. Returns true if a fresh live display landed (not stale/partial). */
    suspend fun decodeLive(): Boolean = withContext(Dispatchers.Default) { nativeDecodeLive(handle) }
    fun getLive(): String = nativeGetLive(handle)
    /** n-best sentence suggestions for the always-visible strip ([0] = shown inline). */
    fun getLiveSuggestions(): Array<String> = nativeGetLiveSuggestions(handle)
    /** Auto candidates for the LAST word in the buffer (mobile convention). */
    fun getLastWordCands(): Array<String> = nativeGetLastWordCands(handle)
    /** The char currently shown for the last word (selected-chip state). */
    fun getLastWordCurrent(): String = nativeGetLastWordCurrent(handle)
    /** Tap a last-word chip: replace that char in place, keep composing. */
    fun pickLastWord(ch: String): Boolean = nativePickLastWord(handle, ch)

    // ---- 聯想 next-word predictions (empty buffer, after a commit) ---------
    /** Predictions following the last committed char (personal + dictionary). */
    fun getPredictions(): Array<String> = nativeGetPredictions(handle)
    /** A prediction chip was committed: learn the transition, chain the tail. */
    fun predicted(text: String) = nativePredicted(handle, text)
    /** Field switch: stale predictions must not carry over. */
    fun clearPredictions() = nativeClearPredictions(handle)
    /** Tap on a suggestion chip: commit that sentence outright, then drain [getCommit]. */
    fun commitSentence(s: String) = nativeCommitSentence(handle, s)
    /** 符 strip tap: insert a literal symbol (token while composing, else direct commit). */
    fun insertSymbol(s: String) = KeyOutcome.of(nativeInsertSymbol(handle, s))

    // ---- convert / choose --------------------------------------------------
    /**
     * ↓ or Enter-convert. HEAVY. focus = token the ↓ landed on (or -1 for the
     * first ambiguous). commitDirect (Enter): commit a single clean decode
     * outright instead of opening the window. Returns true iff the candidate
     * window opened (false => either committed, drain [getCommit], or failed —
     * check [getNotice]).
     */
    suspend fun beginConvert(focus: Int, commitDirect: Boolean): Boolean =
        withContext(Dispatchers.Default) { nativeBeginConvert(handle, focus, commitDirect) }

    /** Enter when the live conversion is already shown: commit without decoding. */
    fun commitLive(): Boolean = nativeCommitLive(handle)

    /** Pre-warm phrase (詞) candidates for the focused segment. HEAVY. */
    suspend fun ensurePhrases() = withContext(Dispatchers.Default) { nativeEnsurePhrases(handle) }

    /** Pick a char candidate (re-scores untouched segments). HEAVY. */
    suspend fun pickSegment(index: Int) = withContext(Dispatchers.Default) { nativePickSegment(handle, index) }

    /** Pick a 2-char 詞 (from [getPhrases]). HEAVY. */
    suspend fun pickPhrase(start: Int, phrase: String) =
        withContext(Dispatchers.Default) { nativePickPhrase(handle, start, phrase) }

    /** Enter in Choosing. Returns true iff it committed (then drain [getCommit]). */
    suspend fun confirmChoosing(): Boolean =
        withContext(Dispatchers.Default) { nativeConfirmChoosing(handle) }

    // ---- choosing navigation (main-safe) ----------------------------------
    fun escapeChoosing(): Boolean = nativeEscapeChoosing(handle)
    /** Combined 詞+字 highlight loop. dir: +1 →, -1 ←. */
    fun moveHighlight(dir: Int) = nativeMoveHighlight(handle, dir)
    /** Move focus to the next/prev ambiguous segment (window closed). */
    fun moveFocus(dir: Int) = nativeMoveFocus(handle, dir)
    fun reopen() = nativeReopen(handle)
    fun closeCandList() = nativeCloseCandList(handle)

    // ---- read-only views (main-safe) --------------------------------------
    fun getPreedit(): Preedit = nativeGetPreedit(handle)
    fun getCandidates(): Candidates = nativeGetCandidates(handle)
    fun getPhrases(): Array<Phrase> = nativeGetPhrases(handle)
    /** Drain staged commit text; send it via InputConnection.commitText. "" = none. */
    fun getCommit(): String = nativeGetCommit(handle)
    fun getNotice(): String = nativeGetNotice(handle)
    fun state(): State = State.of(nativeState(handle))
    fun symbolMode(): Boolean = nativeSymbolMode(handle)

    // ---- native surface (bound by Java_com_slothing_ime_Core_*) -----------
    private external fun nativeInit(model: ByteArray, sylVocab: ByteArray, char2id: ByteArray, table: ByteArray, threads: Int, learnPath: String, assocTsv: ByteArray, assocUserPath: String): Long
    private external fun nativeReady(handle: Long): Boolean
    private external fun nativeDecodeBest(handle: Long, syls: String): String
    private external fun nativeDestroy(handle: Long)
    private external fun nativeSetContext(handle: Long, ctx: String)
    private external fun nativeSetEnglishMode(handle: Long, on: Boolean)
    private external fun nativeSetFullWidth(handle: Long, on: Boolean)

    private external fun nativeFeedKey(handle: Long, cp: Int): Int
    private external fun nativeToneOrSpace(handle: Long, cp: Int): Int
    private external fun nativeBackspace(handle: Long): Int
    private external fun nativeMoveCursor(handle: Long, dir: Int): Int
    private external fun nativeReset(handle: Long)
    private external fun nativeFlush(handle: Long)

    private external fun nativeRefreshLiveFast(handle: Long): Boolean
    private external fun nativeDecodeLive(handle: Long): Boolean
    private external fun nativeGetLive(handle: Long): String
    private external fun nativeGetLiveSuggestions(handle: Long): Array<String>
    private external fun nativeGetLastWordCands(handle: Long): Array<String>
    private external fun nativeGetLastWordCurrent(handle: Long): String
    private external fun nativePickLastWord(handle: Long, ch: String): Boolean
    private external fun nativeGetPredictions(handle: Long): Array<String>
    private external fun nativePredicted(handle: Long, s: String)
    private external fun nativeClearPredictions(handle: Long)
    private external fun nativeCommitSentence(handle: Long, s: String)
    private external fun nativeInsertSymbol(handle: Long, s: String): Int

    private external fun nativeBeginConvert(handle: Long, focus: Int, commitDirect: Boolean): Boolean
    private external fun nativeCommitLive(handle: Long): Boolean
    private external fun nativeEnsurePhrases(handle: Long)
    private external fun nativePickSegment(handle: Long, idx: Int)
    private external fun nativePickPhrase(handle: Long, start: Int, phrase: String)
    private external fun nativeConfirmChoosing(handle: Long): Boolean

    private external fun nativeEscapeChoosing(handle: Long): Boolean
    private external fun nativeMoveHighlight(handle: Long, dir: Int)
    private external fun nativeMoveFocus(handle: Long, dir: Int)
    private external fun nativeReopen(handle: Long)
    private external fun nativeCloseCandList(handle: Long)

    private external fun nativeGetPreedit(handle: Long): Preedit
    private external fun nativeGetCandidates(handle: Long): Candidates
    private external fun nativeGetPhrases(handle: Long): Array<Phrase>
    private external fun nativeGetCommit(handle: Long): String
    private external fun nativeGetNotice(handle: Long): String
    private external fun nativeState(handle: Long): Int
    private external fun nativeSymbolMode(handle: Long): Boolean

    companion object {
        init { System.loadLibrary("slothing") }
    }
}

/**
 * Composing/choosing display. [cursor], [hlStart], [hlEnd] are codepoint indices
 * into [text] (map to UTF-16 offsets for setComposingText if you commit
 * supplementary CJK). [hlStart] == -1 when nothing is highlighted.
 */
data class Preedit(val text: String, val cursor: Int, val hlStart: Int, val hlEnd: Int)

/** The ↓ candidate window for the focused segment. */
data class Candidates(val focus: Int, val cursor: Int, val open: Boolean, val items: Array<String>) {
    // arrays: value-based equals/hashCode
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Candidates) return false
        return focus == other.focus && cursor == other.cursor && open == other.open &&
            items.contentEquals(other.items)
    }
    override fun hashCode(): Int =
        ((focus * 31 + cursor) * 31 + open.hashCode()) * 31 + items.contentHashCode()
}

/** A 2-char 詞 candidate: [start] is the token index it covers (pickPhrase arg). */
data class Phrase(val start: Int, val text: String)
