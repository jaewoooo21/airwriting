package com.airwriting.client

import android.os.Bundle
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import org.json.JSONArray

class ActionHistoryActivity : AppCompatActivity() {

    private lateinit var listContainer: LinearLayout
    private lateinit var txtHistoryCount: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_action_history)

        listContainer = findViewById(R.id.historyListContainer)
        txtHistoryCount = findViewById(R.id.txtHistoryCount)

        // Back
        findViewById<MaterialButton>(R.id.btnBack).setOnClickListener { finish() }

        // Clear
        findViewById<MaterialButton>(R.id.btnClearHistory).setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("히스토리 삭제")
                .setMessage("모든 액션 실행 기록을 삭제하시겠습니까?")
                .setPositiveButton("삭제") { _, _ ->
                    getSharedPreferences("airwriting_prefs", MODE_PRIVATE)
                        .edit().putString(WebSocketService.HISTORY_KEY, "[]").apply()
                    refreshList()
                    Toast.makeText(this, "히스토리 삭제 완료", Toast.LENGTH_SHORT).show()
                }
                .setNegativeButton("취소", null)
                .show()
        }

        refreshList()
    }

    private fun refreshList() {
        listContainer.removeAllViews()

        val prefs = getSharedPreferences("airwriting_prefs", MODE_PRIVATE)
        val json = prefs.getString(WebSocketService.HISTORY_KEY, "[]") ?: "[]"
        val arr = JSONArray(json)

        txtHistoryCount.text = "Total: ${arr.length()} actions"

        if (arr.length() == 0) {
            val empty = TextView(this).apply {
                text = "아직 실행된 액션이 없습니다.\n서버에 연결하여 제스처를 인식해보세요!"
                setTextColor(0xFF666666.toInt())
                textSize = 14f
                gravity = Gravity.CENTER
                setPadding(0, 60, 0, 60)
            }
            listContainer.addView(empty)
            return
        }

        // Show from newest first
        for (i in (arr.length() - 1) downTo 0) {
            val obj = arr.getJSONObject(i)
            val keyword = obj.optString("keyword", "?")
            val label = obj.optString("label", keyword)
            val time = obj.optString("time", "")

            val card = MaterialCardView(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = 8 }
                setCardBackgroundColor(0xFFFFFFFF.toInt())
                radius = 12f
                strokeColor = 0xFFE0E0E0.toInt()
                strokeWidth = 1
                cardElevation = 2f
            }

            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(16, 14, 16, 14)
            }

            // Keyword badge
            val badge = TextView(this).apply {
                text = keyword.take(1)
                setTextColor(0xFFFFFFFF.toInt())
                textSize = 16f
                setBackgroundColor(0xFF1A1A1A.toInt())
                gravity = Gravity.CENTER
                setPadding(14, 8, 14, 8)
                layoutParams = LinearLayout.LayoutParams(44, 44).apply { marginEnd = 12 }
            }
            row.addView(badge)

            // Info
            val info = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            info.addView(TextView(this).apply {
                text = "$label → $keyword"
                setTextColor(0xFF1A1A1A.toInt())
                textSize = 14f
            })
            info.addView(TextView(this).apply {
                text = time
                setTextColor(0xFF999999.toInt())
                textSize = 11f
            })
            row.addView(info)

            // Re-run button
            val rerunBtn = MaterialButton(this).apply {
                text = "▶"
                setTextColor(0xFF2E7D32.toInt())
                setBackgroundColor(0x00000000)
                minimumWidth = 0
                minWidth = 0
                setPadding(8, 0, 8, 0)
                setOnClickListener {
                    // Try custom mapping first, then fallback
                    try {
                        val letterMap = mapOf(
                            "CALL" to "C", "MESSAGE" to "M", "GEMINI" to "G",
                            "SEARCH" to "S", "WEATHER" to "W", "PHOTO" to "P",
                            "GALLERY" to "V", "TIMER" to "T", "ALARM" to "A",
                            "NOTE" to "N", "HOME" to "H", "BIXBY" to "B"
                        )
                        val letter = letterMap[keyword] ?: keyword
                        val custom = CustomMappingActivity.loadMappings(this@ActionHistoryActivity)[letter]
                        if (custom != null && CustomMappingActivity.executeCustomAction(this@ActionHistoryActivity, custom)) {
                            Toast.makeText(this@ActionHistoryActivity, "✅ $keyword re-executed", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(this@ActionHistoryActivity, "재실행은 서버 연결 상태에서만 가능합니다", Toast.LENGTH_SHORT).show()
                        }
                    } catch (_: Exception) {
                        Toast.makeText(this@ActionHistoryActivity, "❌ 실행 실패", Toast.LENGTH_SHORT).show()
                    }
                }
            }
            row.addView(rerunBtn)

            card.addView(row)
            listContainer.addView(card)
        }
    }
}
