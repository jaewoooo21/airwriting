package com.airwriting.client

import android.os.Bundle
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import org.json.JSONArray

class HistoryFragment : Fragment() {

    private lateinit var listContainer: LinearLayout
    private lateinit var txtHistoryCount: TextView

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.fragment_history, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val ctx = requireContext()
        listContainer = view.findViewById(R.id.historyListContainer)
        txtHistoryCount = view.findViewById(R.id.txtHistoryCount)

        view.findViewById<MaterialButton>(R.id.btnClearHistory).setOnClickListener {
            AlertDialog.Builder(ctx).setTitle("히스토리 삭제").setMessage("모든 기록을 삭제하시겠습니까?")
                .setPositiveButton("삭제") { _, _ ->
                    ctx.getSharedPreferences("airwriting_prefs", android.content.Context.MODE_PRIVATE)
                        .edit().putString(WebSocketService.HISTORY_KEY, "[]").apply()
                    refreshList(); Toast.makeText(ctx, "삭제 완료", Toast.LENGTH_SHORT).show()
                }.setNegativeButton("취소", null).show()
        }

        refreshList()
    }

    override fun onResume() {
        super.onResume()
        refreshList()
    }

    private fun refreshList() {
        listContainer.removeAllViews()
        val ctx = requireContext()
        val prefs = ctx.getSharedPreferences("airwriting_prefs", android.content.Context.MODE_PRIVATE)
        val json = prefs.getString(WebSocketService.HISTORY_KEY, "[]") ?: "[]"
        val arr = JSONArray(json)
        txtHistoryCount.text = "Total: ${arr.length()} actions"

        if (arr.length() == 0) {
            listContainer.addView(TextView(ctx).apply {
                text = "아직 실행된 액션이 없습니다.\n서버에 연결하여 제스처를 인식해보세요!"
                setTextColor(0xFF94A3B8.toInt()); textSize = 14f; gravity = Gravity.CENTER; setPadding(0, 60, 0, 60)
            }); return
        }

        for (i in (arr.length() - 1) downTo 0) {
            val obj = arr.getJSONObject(i)
            val keyword = obj.optString("keyword", "?"); val label = obj.optString("label", keyword)
            val time = obj.optString("time", "")

            val card = MaterialCardView(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { bottomMargin = 8 }
                setCardBackgroundColor(0xFFFFFFFF.toInt()); radius = 12f; strokeColor = 0xFFE2E8F0.toInt(); strokeWidth = 1; cardElevation = 2f
            }
            val row = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(16, 14, 16, 14) }
            row.addView(TextView(ctx).apply {
                text = keyword.take(1); setTextColor(0xFFFFFFFF.toInt()); textSize = 16f; setBackgroundColor(0xFF0EA5E9.toInt())
                gravity = Gravity.CENTER; setPadding(14, 8, 14, 8)
                layoutParams = LinearLayout.LayoutParams(44, 44).apply { marginEnd = 12 }
            })
            val info = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
            info.addView(TextView(ctx).apply { text = "$label → $keyword"; setTextColor(0xFF0F172A.toInt()); textSize = 14f })
            info.addView(TextView(ctx).apply { text = time; setTextColor(0xFF94A3B8.toInt()); textSize = 11f })
            row.addView(info)
            row.addView(MaterialButton(ctx).apply {
                text = "▶"; setTextColor(0xFF10B981.toInt()); setBackgroundColor(0x00000000); minimumWidth = 0; minWidth = 0; setPadding(8,0,8,0)
                setOnClickListener {
                    try {
                        val letterMap = mapOf("CALL" to "C","MESSAGE" to "M","GEMINI" to "G","SEARCH" to "S","WEATHER" to "W","PHOTO" to "P","GALLERY" to "V","TIMER" to "T","ALARM" to "A","NOTE" to "N","HOME" to "H","BIXBY" to "B")
                        val letter = letterMap[keyword] ?: keyword
                        val custom = ActionsFragment.loadMappings(ctx)[letter]
                        if (custom != null && ActionsFragment.executeCustomAction(ctx, custom))
                            Toast.makeText(ctx, "✅ Re-executed", Toast.LENGTH_SHORT).show()
                        else Toast.makeText(ctx, "재실행 불가", Toast.LENGTH_SHORT).show()
                    } catch (_: Exception) { Toast.makeText(ctx, "❌ 실패", Toast.LENGTH_SHORT).show() }
                }
            })
            card.addView(row); listContainer.addView(card)
        }
    }
}
