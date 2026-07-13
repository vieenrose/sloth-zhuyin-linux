package com.sloth.ime

import android.content.Context
import android.os.Build

/**
 * Palette picker (docs/THEME.md): ONYX e-ink devices keep the flat pure
 * black-on-white eink_* palette (16-level grayscale + full-refresh ghosting
 * punish anything subtler); everything else maps each eink_* color to its
 * sloth_* twin — 樹懶「樹梢」light in values/, 「夜森林」dark in
 * values-night/ — so the sloth-and-tree skin follows the system day/night
 * mode with no per-call-site changes.
 */
object Skin {
    val EINK: Boolean = Build.MANUFACTURER.equals("ONYX", ignoreCase = true)

    private val cache = HashMap<Int, Int>()

    fun color(ctx: Context, einkId: Int): Int {
        val id = if (EINK) einkId else cache.getOrPut(einkId) {
            val name = ctx.resources.getResourceEntryName(einkId)
                .replaceFirst("eink_", "sloth_")
            val sloth = ctx.resources.getIdentifier(name, "color", ctx.packageName)
            if (sloth != 0) sloth else einkId
        }
        return ctx.resources.getColor(id, ctx.theme)
    }
}
