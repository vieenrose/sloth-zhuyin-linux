package com.sloth.ime

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.StateListDrawable
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

/**
 * The on-screen Dàqiān (大千) bopomofo keyboard — a faithful port of the iOS
 * ".ioskb" layout in space-static/index.html + space-static/ime.js.
 *
 * Source of truth for the grid (space-static/ime.js): the DACHEN key→glyph map,
 * the four physical QWERTY ROWS, and TONEK (3/4/6/7 → ˇ/ˋ/ˊ/˙). The right-hand
 * FUNCTION COLUMN reproduces the iPhone 注音 keyboard exactly, one per row:
 *   row0 ⌫ (backspace) · row1 、(頓號) · row2 ？· row3 ⏎ (上字/commit).
 * A compact bottom utility row adds what the iOS caps also carry: 中/英 English
 * toggle, 符 symbol menu, ，。, a wide 空白（一聲）space, and ↓ 選字 to open the
 * candidate window.
 *
 * Each bopomofo key DISPLAYS its bopomofo glyph but FEEDS its underlying ASCII
 * (QWERTY) key to the core — the same contract the Linux engines use
 * (processKey / Segmenter key in dachenMap). The tone digits 3/4/6/7 and space
 * are routed by the service to Core.toneOrSpace; every other printable to
 * Core.feedKey.
 *
 * E-ink notes: no ripples, no elevation, no animation. A key is a flat TextView
 * whose pressed state is a StateListDrawable colour swap — a single clean
 * redraw. Large (52dp) high-contrast targets, pure black on white.
 */
class KeyboardView(context: Context) : LinearLayout(context) {

    interface Listener {
        /** A grid key or punctuation ASCII. Service decides feedKey vs toneOrSpace. */
        fun onKey(ascii: Char)
        fun onBackspace()
        fun onEnter()
        /** ↓ 選字 / long-press space: open the ChoosingCore candidate window. */
        fun onOpenChoosing()
        fun onToggleEnglish()
        fun onToggleSymbols()
    }

    var listener: Listener? = null

    // --- Dàqiān map (EXACT copy of ime.js DACHEN + TONEK) ------------------
    private val glyphOf: Map<Char, String> = mapOf(
        '1' to "ㄅ", 'q' to "ㄆ", 'a' to "ㄇ", 'z' to "ㄈ",
        '2' to "ㄉ", 'w' to "ㄊ", 's' to "ㄋ", 'x' to "ㄌ",
        'e' to "ㄍ", 'd' to "ㄎ", 'c' to "ㄏ", 'r' to "ㄐ",
        'f' to "ㄑ", 'v' to "ㄒ", '5' to "ㄓ", 't' to "ㄔ",
        'g' to "ㄕ", 'b' to "ㄖ", 'y' to "ㄗ", 'h' to "ㄘ",
        'n' to "ㄙ", 'u' to "ㄧ", 'j' to "ㄨ", 'm' to "ㄩ",
        '8' to "ㄚ", 'i' to "ㄛ", 'k' to "ㄜ", ',' to "ㄝ",
        '9' to "ㄞ", 'o' to "ㄟ", 'l' to "ㄠ", '.' to "ㄡ",
        '0' to "ㄢ", 'p' to "ㄣ", ';' to "ㄤ", '/' to "ㄥ",
        '-' to "ㄦ",
        // TONEK — shown as their marks, fed as their digit
        '6' to "ˊ", '3' to "ˇ", '4' to "ˋ", '7' to "˙",
    )

    // The four physical rows, in QWERTY order (ime.js ROWS).
    private val rows: List<List<Char>> = listOf(
        listOf('1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-'),
        listOf('q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'),
        listOf('a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';'),
        listOf('z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'),
    )

    // Right-hand iOS function column, one per bopomofo row (ime.js .ioskb fns).
    private data class Fn(val label: String, val run: (Listener) -> Unit)
    private val fnColumn: List<Fn> = listOf(
        Fn("⌫") { it.onBackspace() },
        Fn("、") { it.onKey('\\') },   // punctMap: '\\' → 、 (頓號)
        Fn("？") { it.onKey('?') },    // punctMap: '?'  → ？
        Fn("⏎") { it.onEnter() },
    )

    // Bopomofo keys we repaint when English mode flips (glyph <-> latin).
    // Dual-label caps: big bopomofo + small latin corner hint — the Dàqiān
    // layout IS QWERTY, and auto zh/en means both alphabets are live at once
    // (the segmenter decides), so the latin must be visible, like physical
    // Taiwanese keyboards. iOS hides it only because iOS switches layouts.
    private class BopoKey(
        val main: TextView, val hint: TextView, val ascii: Char, val glyph: String,
    )
    private val bopoKeys = ArrayList<BopoKey>(41)
    private var englishKey: TextView? = null
    private var english = false

    // ascii -> key view, for the hardware-keyboard echo (flashKey). Grid keys
    // map by their QWERTY ascii; specials use control chars: '\b' ⌫, '\n' ⏎,
    // ' ' space, '<' ，, '>' 。, '\\' 、, '?' ？.
    private val keyViews = HashMap<Char, View>()

    private val pad = dp(3)

    init {
        orientation = VERTICAL
        setBackgroundColor(color(R.color.eink_bg))
        setPadding(pad, pad, pad, pad)
        buildBopomofoRows()
        buildUtilityRow()
    }

    private fun buildBopomofoRows() {
        rows.forEachIndexed { ri, keys ->
            val row = newRow()
            for (k in keys) {
                val glyph = glyphOf[k] ?: k.toString()
                val (container, main, hint) = makeDualKey(glyph, k.toString())
                container.setOnClickListener { listener?.onKey(k) }
                row.addView(container)
                bopoKeys.add(BopoKey(main, hint, k, glyph))
                keyViews[k] = container
            }
            // right-hand iOS function key for this row
            fnColumn.getOrNull(ri)?.let { fn ->
                val v = makeKey(fn.label, weight = 1f, fnKey = true)
                v.setOnClickListener { l -> listener?.let { fn.run(it) } }
                if (fn.label == "⌫") {
                    // hold-to-repeat (standard mobile-keyboard behavior)
                    attachRepeat(v) { listener?.onBackspace() }
                }
                when (fn.label) {
                    "⌫" -> keyViews['\b'] = v
                    "、" -> keyViews['\\'] = v
                    "？" -> keyViews['?'] = v
                    "⏎" -> keyViews['\n'] = v
                }
                row.addView(v)
            }
            addView(row)
        }
    }

    private fun buildUtilityRow() {
        val row = newRow()

        englishKey = makeKey(if (english) "英" else "中", weight = 1f, fnKey = true).also {
            it.setOnClickListener { listener?.onToggleEnglish() }
            row.addView(it)
        }
        makeKey("符", weight = 1f, fnKey = true).also {
            it.setOnClickListener { listener?.onToggleSymbols() }
            row.addView(it)
        }
        makeKey("，", weight = 1f, fnKey = true).also {
            it.setOnClickListener { listener?.onKey('<') }   // punctMap: '<' → ，
            keyViews['<'] = it
            row.addView(it)
        }
        makeKey("。", weight = 1f, fnKey = true).also {
            it.setOnClickListener { listener?.onKey('>') }   // punctMap: '>' → 。
            keyViews['>'] = it
            row.addView(it)
        }
        // Wide space = 一聲 (tone 1). Long-press opens the ↓ candidate window.
        makeKey(context.getString(R.string.key_space), weight = 3f, fnKey = false).also {
            it.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            it.setOnClickListener { listener?.onKey(' ') }
            it.setOnLongClickListener { listener?.onOpenChoosing(); true }
            keyViews[' '] = it
            row.addView(it)
        }
        makeKey(context.getString(R.string.key_choose), weight = 1f, fnKey = true).also {
            it.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            it.setOnClickListener { listener?.onOpenChoosing() }
            row.addView(it)
        }
        addView(row)
    }

    /** Hardware-echo hold time. Debug builds may override it via
     *  `adb shell settings put global sloth_flash_ms N` (demo recording
     *  needs the pressed state to survive a screencap round-trip). */
    var flashHoldMs = 250L

    /** Echo a hardware (Bluetooth/USB) key press on the on-screen keyboard:
     *  momentarily paint the matching key's pressed visual so the user (and
     *  demo recordings) can see WHICH key was hit. Two clean redraws — no
     *  animation, e-ink friendly. Uppercase falls back to its lowercase key. */
    fun flashKey(ascii: Char) {
        val v = keyViews[ascii] ?: keyViews[ascii.lowercaseChar()] ?: return
        v.isPressed = true
        v.postDelayed({ v.isPressed = false }, flashHoldMs)
    }

    /** flashKey for the 中/英 utility key (lone-Shift hardware toggle). */
    fun flashEnglishToggle() {
        englishKey?.let {
            it.isPressed = true
            it.postDelayed({ it.isPressed = false }, flashHoldMs)
        }
    }

    /** Repaint the grid for English passthrough vs bopomofo. */
    fun setEnglish(on: Boolean) {
        if (english == on) return
        english = on
        for (b in bopoKeys) {
            // lowercase: that's what feedKey's passthrough actually commits
            b.main.text = if (on) b.ascii.toString() else b.glyph
            b.hint.visibility = if (on) INVISIBLE else VISIBLE
        }
        englishKey?.text = if (on) "英" else "中"
    }

    // --- key + row factories ----------------------------------------------

    private fun newRow(): LinearLayout = LinearLayout(context).apply {
        orientation = HORIZONTAL
        layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
        gravity = Gravity.CENTER_HORIZONTAL
    }

    /** A bopomofo key: big glyph + small latin hint in the top-right corner
     *  (dual-alphabet cap, like physical Taiwanese keyboards). */
    private fun makeDualKey(
        glyph: String,
        latin: String,
    ): Triple<android.widget.FrameLayout, TextView, TextView> {
        val edge = color(R.color.eink_edge)
        val fill = color(R.color.eink_key)
        val pressedBg = color(R.color.eink_pressed_bg)
        val pressedInk = color(R.color.eink_pressed_ink)
        val ink = color(R.color.eink_ink)
        val muted = color(R.color.eink_muted)
        val latinInk = color(R.color.eink_latin)

        fun shape(bg: Int) = GradientDrawable().apply {
            setColor(bg)
            cornerRadius = dp(6).toFloat()
            setStroke(dp(1), edge)
        }
        val bg = StateListDrawable().apply {
            addState(intArrayOf(android.R.attr.state_pressed), shape(pressedBg))
            addState(intArrayOf(), shape(fill))
        }
        val mainColors = ColorStateList(
            arrayOf(intArrayOf(android.R.attr.state_pressed), intArrayOf()),
            intArrayOf(pressedInk, ink),
        )
        val hintColors = ColorStateList(
            arrayOf(intArrayOf(android.R.attr.state_pressed), intArrayOf()),
            intArrayOf(pressedInk, latinInk),
        )
        val main = TextView(context).apply {
            text = glyph
            gravity = Gravity.CENTER
            includeFontPadding = false
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
            setTextColor(mainColors)
            isDuplicateParentStateEnabled = true
            // bopomofo at bottom-right (dual-alphabet keycap: latin top-left, glyph bottom-right)
            layoutParams = android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.WRAP_CONTENT,
                android.widget.FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.BOTTOM or Gravity.END,
            ).apply { setMargins(0, 0, dp(8), dp(4)) }
        }
        val hint = TextView(context).apply {
            text = latin
            includeFontPadding = false
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 20f)   // as prominent as the glyph; green (eink_latin) for distinction
            setTextColor(hintColors)
            isDuplicateParentStateEnabled = true
            // latin at top-left (dual-alphabet keycap)
            layoutParams = android.widget.FrameLayout.LayoutParams(
                android.widget.FrameLayout.LayoutParams.WRAP_CONTENT,
                android.widget.FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.START,
            ).apply { setMargins(dp(7), dp(3), 0, 0) }
        }
        val box = android.widget.FrameLayout(context).apply {
            isClickable = true
            isFocusable = true
            background = bg
            addView(main)
            addView(hint)
            layoutParams = LayoutParams(0, dp(52), 1f).apply {
                val m = dp(2)
                setMargins(m, m, m, m)
            }
        }
        return Triple(box, main, hint)
    }

    private fun makeKey(label: String, weight: Float, fnKey: Boolean): TextView {
        val edge = color(R.color.eink_edge)
        val fill = color(if (fnKey) R.color.eink_fn_key else R.color.eink_key)
        val pressedBg = color(R.color.eink_pressed_bg)
        val pressedInk = color(R.color.eink_pressed_ink)
        val ink = color(R.color.eink_ink)

        fun shape(bg: Int) = GradientDrawable().apply {
            setColor(bg)
            cornerRadius = dp(6).toFloat()
            setStroke(dp(1), edge)
        }
        val bg = StateListDrawable().apply {
            addState(intArrayOf(android.R.attr.state_pressed), shape(pressedBg))
            addState(intArrayOf(), shape(fill))
        }
        val textColors = ColorStateList(
            arrayOf(intArrayOf(android.R.attr.state_pressed), intArrayOf()),
            intArrayOf(pressedInk, ink),
        )
        return TextView(context).apply {
            text = label
            gravity = Gravity.CENTER
            includeFontPadding = false
            isClickable = true
            isFocusable = true
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
            setTextColor(textColors)
            background = bg
            layoutParams = LayoutParams(0, dp(52), weight).apply {
                val m = dp(2)
                setMargins(m, m, m, m)
            }
        }
    }

    /** Long-press starts auto-repeat (~12 Hz); any touch-up/cancel stops it.
     *  The plain click still delivers the single press. */
    private fun attachRepeat(v: View, action: () -> Unit) {
        val handler = Handler(Looper.getMainLooper())
        var repeater: Runnable? = null
        v.setOnLongClickListener {
            val r = object : Runnable {
                override fun run() {
                    action()
                    handler.postDelayed(this, 80)
                }
            }
            repeater = r
            handler.post(r) // first repeat immediately at long-press
            true
        }
        v.setOnTouchListener { _, ev ->
            if (ev.actionMasked == MotionEvent.ACTION_UP ||
                ev.actionMasked == MotionEvent.ACTION_CANCEL
            ) {
                repeater?.let { handler.removeCallbacks(it) }
                repeater = null
            }
            false // don't consume: click/long-click still fire normally
        }
    }

    private fun dp(v: Int): Int =
        (v * resources.displayMetrics.density).toInt()

    private fun color(id: Int): Int = Skin.color(context, id)
}
