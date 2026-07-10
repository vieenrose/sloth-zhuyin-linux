package com.slothing.ime

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.StateListDrawable
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Horizontal candidate strip shown above the keyboard while ChoosingCore is
 * open (after ↓). Renders the model-ranked 詞 (2-char phrase) chips first, then
 * the numbered 字 candidates for the focused segment — the same two rows the
 * web demo (#phrases / #cands) and the IBus engine (lookup table + 詞 aux row)
 * paint.
 *
 * A plain scrollable LinearLayout of flat chips (no RecyclerView, no animation)
 * keeps e-ink redraws to one clean pass. The currently selected 字 is shown
 * reverse-video (Candidates.cursor).
 */
class CandidateBar(context: Context) : HorizontalScrollView(context) {

    interface Listener {
        fun onPickCandidate(index: Int)
        fun onPickPhrase(phrase: Phrase)
    }

    var listener: Listener? = null

    private val strip = LinearLayout(context).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
    }

    init {
        isFillViewport = true
        isHorizontalScrollBarEnabled = false
        setBackgroundColor(color(R.color.eink_bg))
        setPadding(dp(4), dp(3), dp(4), dp(3))
        addView(
            strip,
            LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT),
        )
    }

    /**
     * Repaint from a Core snapshot. Returns true if anything is shown (the
     * service uses that to toggle the bar's visibility).
     */
    fun render(cands: Candidates?, phrases: Array<Phrase>): Boolean {
        strip.removeAllViews()
        scrollX = 0
        var shown = false

        if (phrases.isNotEmpty()) {
            strip.addView(labelChip("詞"))
            phrases.forEachIndexed { i, p ->
                val chip = chip("${supIndex(i + 1)} ${p.text}", phrase = true, selected = false)
                chip.setOnClickListener { listener?.onPickPhrase(p) }
                strip.addView(chip)
                shown = true
            }
        }

        val open = cands != null && cands.open && cands.items.isNotEmpty()
        if (open) {
            if (phrases.isNotEmpty()) strip.addView(divider())
            cands!!.items.forEachIndexed { i, c ->
                val label = if (i < 9) "${i + 1} $c" else c
                val chip = chip(label, phrase = false, selected = i == cands.cursor)
                chip.setOnClickListener { listener?.onPickCandidate(i) }
                strip.addView(chip)
                shown = true
            }
        }
        return shown
    }

    fun clear() {
        strip.removeAllViews()
    }

    // --- chip factories ----------------------------------------------------

    private fun chip(text: String, phrase: Boolean, selected: Boolean): TextView {
        val edge = color(R.color.eink_edge)
        val fill = color(if (selected) R.color.eink_focus_bg else R.color.eink_key)
        val ink = color(if (selected) R.color.eink_focus_ink else R.color.eink_ink)
        val pressedBg = color(R.color.eink_pressed_bg)
        val pressedInk = color(R.color.eink_pressed_ink)

        fun shape(bg: Int) = GradientDrawable().apply {
            setColor(bg)
            cornerRadius = dp(6).toFloat()
            setStroke(dp(1), edge)
        }
        val bg = StateListDrawable().apply {
            addState(intArrayOf(android.R.attr.state_pressed), shape(pressedBg))
            addState(intArrayOf(), shape(fill))
        }
        val colors = ColorStateList(
            arrayOf(intArrayOf(android.R.attr.state_pressed), intArrayOf()),
            intArrayOf(pressedInk, ink),
        )
        return TextView(context).apply {
            this.text = text
            gravity = Gravity.CENTER
            includeFontPadding = false
            isClickable = true
            setTextSize(TypedValue.COMPLEX_UNIT_SP, if (phrase) 18f else 20f)
            setTextColor(colors)
            background = bg
            minHeight = dp(44)
            minWidth = dp(44)
            setPadding(dp(12), dp(6), dp(12), dp(6))
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            )
            lp.setMargins(dp(3), dp(2), dp(3), dp(2))
            layoutParams = lp
        }
    }

    private fun labelChip(text: String): TextView = TextView(context).apply {
        this.text = text
        gravity = Gravity.CENTER
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
        setTextColor(color(R.color.eink_muted))
        setPadding(dp(4), 0, dp(6), 0)
    }

    private fun divider(): View = View(context).apply {
        setBackgroundColor(color(R.color.eink_edge))
        layoutParams = LinearLayout.LayoutParams(dp(1), dp(32)).apply {
            setMargins(dp(6), 0, dp(6), 0)
        }
    }

    private fun supIndex(n: Int): String = "⇧$n"

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
    private fun color(id: Int): Int = resources.getColor(id, context.theme)
}
