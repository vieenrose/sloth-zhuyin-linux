package com.slothing.ime

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.StateListDrawable
import android.util.TypedValue
import android.view.Gravity
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
    private class BopoKey(val view: TextView, val ascii: Char, val glyph: String)
    private val bopoKeys = ArrayList<BopoKey>(37)
    private var englishKey: TextView? = null
    private var english = false

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
                val v = makeKey(glyph, weight = 1f, fnKey = false)
                v.setOnClickListener { listener?.onKey(k) }
                row.addView(v)
                bopoKeys.add(BopoKey(v, k, glyph))
            }
            // right-hand iOS function key for this row
            fnColumn.getOrNull(ri)?.let { fn ->
                val v = makeKey(fn.label, weight = 1f, fnKey = true)
                v.setOnClickListener { l -> listener?.let { fn.run(it) } }
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
            row.addView(it)
        }
        makeKey("。", weight = 1f, fnKey = true).also {
            it.setOnClickListener { listener?.onKey('>') }   // punctMap: '>' → 。
            row.addView(it)
        }
        // Wide space = 一聲 (tone 1). Long-press opens the ↓ candidate window.
        makeKey(context.getString(R.string.key_space), weight = 3f, fnKey = false).also {
            it.setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            it.setOnClickListener { listener?.onKey(' ') }
            it.setOnLongClickListener { listener?.onOpenChoosing(); true }
            row.addView(it)
        }
        makeKey(context.getString(R.string.key_choose), weight = 1f, fnKey = true).also {
            it.setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            it.setOnClickListener { listener?.onOpenChoosing() }
            row.addView(it)
        }
        addView(row)
    }

    /** Repaint the grid for English passthrough (uppercase latin) vs bopomofo. */
    fun setEnglish(on: Boolean) {
        if (english == on) return
        english = on
        for (b in bopoKeys) b.view.text = if (on) b.ascii.uppercaseChar().toString() else b.glyph
        englishKey?.text = if (on) "英" else "中"
    }

    // --- key + row factories ----------------------------------------------

    private fun newRow(): LinearLayout = LinearLayout(context).apply {
        orientation = HORIZONTAL
        layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
        gravity = Gravity.CENTER_HORIZONTAL
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

    private fun dp(v: Int): Int =
        (v * resources.displayMetrics.density).toInt()

    private fun color(id: Int): Int =
        resources.getColor(id, context.theme)
}
