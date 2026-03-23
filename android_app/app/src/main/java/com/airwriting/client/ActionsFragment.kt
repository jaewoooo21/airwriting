package com.airwriting.client

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.textfield.TextInputEditText
import org.json.JSONArray
import org.json.JSONObject

class ActionsFragment : Fragment() {

    companion object {
        const val PREFS_KEY = "custom_mappings"
        val ACTION_TYPES = arrayOf("📞 전화 걸기", "💬 문자 보내기", "🌐 URL 열기", "🔍 검색하기", "📱 앱 실행")
        val ACTION_TYPE_KEYS = arrayOf("call", "sms", "url", "search", "app")

        fun loadMappings(context: Context): Map<String, JSONObject> {
            val prefs = context.getSharedPreferences("airwriting_prefs", Context.MODE_PRIVATE)
            val json = prefs.getString(PREFS_KEY, "[]") ?: "[]"
            val arr = JSONArray(json)
            val map = mutableMapOf<String, JSONObject>()
            for (i in 0 until arr.length()) { val obj = arr.getJSONObject(i); map[obj.getString("gesture")] = obj }
            return map
        }

        fun executeCustomAction(context: Context, mapping: JSONObject): Boolean {
            return try {
                val type = mapping.getString("actionType"); val value = mapping.getString("value")
                val intent = when (type) {
                    "call" -> Intent(Intent.ACTION_CALL, Uri.parse("tel:$value"))
                    "sms" -> Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$value"))
                    "url" -> Intent(Intent.ACTION_VIEW, Uri.parse(value))
                    "search" -> Intent(Intent.ACTION_WEB_SEARCH).apply { putExtra("query", value) }
                    "app" -> context.packageManager.getLaunchIntentForPackage(value)
                    else -> null
                }
                if (intent != null) { intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK); context.startActivity(intent); true } else false
            } catch (_: Exception) { false }
        }
    }

    data class Preset(val gesture: String, val actionType: String, val actionLabel: String, val value: String)
    private val PRESETS = mapOf(
        "kakao" to Preset("K", "app", "📱 앱 실행", "com.kakao.talk"),
        "youtube" to Preset("Y", "app", "📱 앱 실행", "com.google.android.youtube"),
        "naver" to Preset("N", "url", "🌐 URL 열기", "https://m.naver.com"),
        "chrome" to Preset("C", "app", "📱 앱 실행", "com.android.chrome"),
        "maps" to Preset("M", "app", "📱 앱 실행", "com.google.android.apps.maps")
    )

    private lateinit var inputGesture: TextInputEditText
    private lateinit var inputActionType: AutoCompleteTextView
    private lateinit var inputActionValue: TextInputEditText
    private lateinit var listContainer: LinearLayout
    private var selectedTypeIndex = 0

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.fragment_actions, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val ctx = requireContext()

        inputGesture = view.findViewById(R.id.inputGesture)
        inputActionType = view.findViewById(R.id.inputActionType)
        inputActionValue = view.findViewById(R.id.inputActionValue)
        listContainer = view.findViewById(R.id.mappingListContainer)

        val adapter = ArrayAdapter(ctx, android.R.layout.simple_dropdown_item_1line, ACTION_TYPES)
        inputActionType.setAdapter(adapter)
        inputActionType.setOnItemClickListener { _, _, pos, _ ->
            selectedTypeIndex = pos
            inputActionValue.hint = when (ACTION_TYPE_KEYS[pos]) {
                "call" -> "전화번호 (예: 01012345678)"; "sms" -> "전화번호"; "url" -> "URL (예: https://google.com)"
                "search" -> "검색어 (예: 오늘 날씨)"; "app" -> "패키지명 (예: com.kakao.talk)"; else -> "값 입력"
            }
        }

        view.findViewById<MaterialButton>(R.id.btnAddMapping).setOnClickListener {
            val gesture = inputGesture.text.toString().trim().uppercase()
            val value = inputActionValue.text.toString().trim()
            if (gesture.isEmpty() || gesture.length != 1) { Toast.makeText(ctx, "글자 1개를 입력해주세요", Toast.LENGTH_SHORT).show(); return@setOnClickListener }
            if (value.isEmpty()) { Toast.makeText(ctx, "값을 입력해주세요", Toast.LENGTH_SHORT).show(); return@setOnClickListener }
            val mapping = JSONObject().apply {
                put("gesture", gesture); put("actionType", ACTION_TYPE_KEYS[selectedTypeIndex])
                put("actionLabel", ACTION_TYPES[selectedTypeIndex]); put("value", value)
            }
            saveMappingToList(gesture, mapping)
            inputGesture.text?.clear(); inputActionValue.text?.clear()
            Toast.makeText(ctx, "✅ '$gesture' 매핑 저장됨", Toast.LENGTH_SHORT).show()
            refreshList()
        }

        view.findViewById<MaterialButton>(R.id.presetKakao).setOnClickListener { applyPreset("kakao") }
        view.findViewById<MaterialButton>(R.id.presetYoutube).setOnClickListener { applyPreset("youtube") }
        view.findViewById<MaterialButton>(R.id.presetNaver).setOnClickListener { applyPreset("naver") }
        view.findViewById<MaterialButton>(R.id.presetChrome).setOnClickListener { applyPreset("chrome") }
        view.findViewById<MaterialButton>(R.id.presetMaps).setOnClickListener { applyPreset("maps") }

        view.findViewById<MaterialButton>(R.id.btnShareAll).setOnClickListener {
            val prefs = ctx.getSharedPreferences("airwriting_prefs", Context.MODE_PRIVATE)
            val json = prefs.getString(PREFS_KEY, "[]") ?: "[]"
            startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"; putExtra(Intent.EXTRA_SUBJECT, "AirWriting Custom Mappings"); putExtra(Intent.EXTRA_TEXT, json)
            }, "Share mappings"))
        }

        view.findViewById<MaterialButton>(R.id.btnClearAll).setOnClickListener {
            AlertDialog.Builder(ctx).setTitle("전체 삭제").setMessage("모든 커스텀 매핑을 삭제하시겠습니까?")
                .setPositiveButton("삭제") { _, _ ->
                    ctx.getSharedPreferences("airwriting_prefs", Context.MODE_PRIVATE)
                        .edit().putString(PREFS_KEY, "[]").apply()
                    refreshList(); Toast.makeText(ctx, "전체 삭제 완료", Toast.LENGTH_SHORT).show()
                }.setNegativeButton("취소", null).show()
        }

        refreshList()
    }

    private fun applyPreset(key: String) {
        val preset = PRESETS[key] ?: return
        val mapping = JSONObject().apply {
            put("gesture", preset.gesture); put("actionType", preset.actionType)
            put("actionLabel", preset.actionLabel); put("value", preset.value)
        }
        saveMappingToList(preset.gesture, mapping)
        Toast.makeText(requireContext(), "✅ '${preset.gesture}' → ${preset.value}", Toast.LENGTH_SHORT).show()
        refreshList()
    }

    private fun saveMappingToList(gesture: String, mapping: JSONObject) {
        val prefs = requireContext().getSharedPreferences("airwriting_prefs", Context.MODE_PRIVATE)
        val json = prefs.getString(PREFS_KEY, "[]") ?: "[]"; val arr = JSONArray(json)
        val newArr = JSONArray()
        for (i in 0 until arr.length()) { val obj = arr.getJSONObject(i); if (obj.getString("gesture") != gesture) newArr.put(obj) }
        newArr.put(mapping); prefs.edit().putString(PREFS_KEY, newArr.toString()).apply()
    }

    private fun removeMappingFromList(gesture: String) {
        val prefs = requireContext().getSharedPreferences("airwriting_prefs", Context.MODE_PRIVATE)
        val json = prefs.getString(PREFS_KEY, "[]") ?: "[]"; val arr = JSONArray(json)
        val newArr = JSONArray()
        for (i in 0 until arr.length()) { val obj = arr.getJSONObject(i); if (obj.getString("gesture") != gesture) newArr.put(obj) }
        prefs.edit().putString(PREFS_KEY, newArr.toString()).apply()
    }

    private fun refreshList() {
        listContainer.removeAllViews()
        val ctx = requireContext()
        val mappings = loadMappings(ctx)
        if (mappings.isEmpty()) {
            listContainer.addView(TextView(ctx).apply {
                text = "커스텀 매핑이 없습니다.\n프리셋을 눌러 빠르게 추가해보세요!"; setTextColor(0xFF94A3B8.toInt())
                textSize = 14f; gravity = Gravity.CENTER; setPadding(0, 40, 0, 40)
            }); return
        }
        for ((gesture, obj) in mappings) {
            val card = MaterialCardView(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { bottomMargin = 8 }
                setCardBackgroundColor(0xFFFFFFFF.toInt()); radius = 12f; strokeColor = 0xFFE2E8F0.toInt(); strokeWidth = 1; cardElevation = 2f
            }
            val row = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL; setPadding(16, 14, 16, 14) }
            val badge = TextView(ctx).apply {
                text = gesture; setTextColor(0xFFFFFFFF.toInt()); textSize = 18f; setBackgroundColor(0xFF0EA5E9.toInt())
                gravity = Gravity.CENTER; setPadding(16, 8, 16, 8)
                layoutParams = LinearLayout.LayoutParams(48, 48).apply { marginEnd = 12 }
            }
            row.addView(badge)
            val info = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
            info.addView(TextView(ctx).apply { text = obj.optString("actionLabel", ""); setTextColor(0xFF0F172A.toInt()); textSize = 14f })
            info.addView(TextView(ctx).apply { text = obj.optString("value", ""); setTextColor(0xFF94A3B8.toInt()); textSize = 12f })
            row.addView(info)
            row.addView(MaterialButton(ctx).apply {
                text = "▶"; setTextColor(0xFF10B981.toInt()); setBackgroundColor(0x00000000); minimumWidth = 0; minWidth = 0; setPadding(8,0,8,0)
                setOnClickListener { if (executeCustomAction(ctx, obj)) Toast.makeText(ctx, "✅ 테스트 실행", Toast.LENGTH_SHORT).show() }
            })
            row.addView(MaterialButton(ctx).apply {
                text = "✕"; setTextColor(0xFFEF4444.toInt()); setBackgroundColor(0x00000000); minimumWidth = 0; minWidth = 0; setPadding(8,0,8,0)
                setOnClickListener { removeMappingFromList(gesture); refreshList(); Toast.makeText(ctx, "'$gesture' 삭제됨", Toast.LENGTH_SHORT).show() }
            })
            card.addView(row); listContainer.addView(card)
        }
    }
}
