package com.slothing.ime

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Tiny launcher/settings activity. Two taps get Slothing usable:
 *   1. "在系統設定啟用" -> Settings.ACTION_INPUT_METHOD_SETTINGS (enable the IME),
 *   2. "切換到 Slothing" -> InputMethodManager.showInputMethodPicker() (select it).
 * Also reachable as the IME's android:settingsActivity (method.xml).
 */
class EnableActivity : Activity() {

    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (16 * resources.displayMetrics.density).toInt()

        val title = TextView(this).apply {
            text = getString(R.string.enable_title)
            textSize = 22f
        }
        val blurb = TextView(this).apply {
            text = getString(R.string.enable_blurb)
            textSize = 14f
            setPadding(0, pad / 2, 0, pad)
        }
        status = TextView(this).apply {
            textSize = 14f
            setPadding(0, pad, 0, pad)
        }
        val enableBtn = Button(this).apply {
            text = getString(R.string.btn_enable)
            setOnClickListener {
                startActivity(
                    Intent(Settings.ACTION_INPUT_METHOD_SETTINGS)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                )
            }
        }
        val pickBtn = Button(this).apply {
            text = getString(R.string.btn_pick)
            setOnClickListener {
                (getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
                    .showInputMethodPicker()
            }
        }
        // A focusable field to try Slothing (and to drive on-device validation).
        val testField = EditText(this).apply {
            id = TEST_FIELD_ID
            hint = "在此測試輸入…"
            textSize = 20f
            setPadding(pad, pad, pad, pad)
        }

        setContentView(
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER_HORIZONTAL
                setPadding(pad, pad * 2, pad, pad * 2)
                addView(title)
                addView(blurb)
                addView(enableBtn, wide())
                addView(pickBtn, wide())
                addView(status)
                addView(testField, wide())
            },
        )
    }

    companion object { const val TEST_FIELD_ID = 0x510C }

    override fun onResume() {
        super.onResume()
        status.text = if (isEnabled()) getString(R.string.status_enabled)
        else getString(R.string.status_disabled)
    }

    /** True once the user has enabled Slothing in Settings. */
    private fun isEnabled(): Boolean {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        return imm.enabledInputMethodList.any { it.packageName == packageName }
    }

    private fun wide(): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT,
        )
}
