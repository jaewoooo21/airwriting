package com.airwriting.client

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.textfield.TextInputEditText
import org.json.JSONArray
import org.json.JSONObject

class CustomMappingActivity : AppCompatActivity() {

    companion object {
        const val PREFS_KEY = "custom_mappings"

        // Action types the user can choose from
        val ACTION_TYPES = arrayOf(
            "📞 전화 걸기",
            "💬 문자 보내기",
            "🌐 URL 열기",
            "🔍 검색하기",
            "📱 앱 실행"
        )
        val ACTION_TYPE_KEYS = arrayOf("call", "sms", "url", "search", "app")

        /** Load custom mappings from SharedPreferences */
        fun loadMappings(context: Context): Map<String, JSONObject> {
            val prefs = context.getSharedPreferences("airwriting_prefs", Context.MODE_PRIVATE)
            val json = prefs.getString(PREFS_KEY, "[]") ?: "[]"
            val arr = JSONArray(json)
            val map = mutableMapOf<String, JSONObject>()
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                map[obj.getString("gesture")] = obj
            }
            return map
        }

        /** Execute a custom mapping */
        fun executeCustomAction(context: Context, mapping: JSONObject): Boolean {
            return try {
                val type = mapping.getString("actionType")
                val value = mapping.getString("value")
                val intent = when (type) {
                    "call" -> Intent(Intent.ACTION_DIAL, Uri.parse("tel:$value"))
                    "sms" -> Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$value"))
                    "url" -> Intent(Intent.ACTION_VIEW, Uri.parse(value))
                    "search" -> Intent(Intent.ACTION_WEB_SEARCH).apply {
                        putExtra("query", value)
                    }
                    "app" -> context.packageManager.getLaunchIntentForPackage(value)
                    else -> null
                }
                if (intent != null) {
                    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    context.startActivity(intent)
                    true
                } else false
            } catch (e: Exception) {
                false
            }
        }
    }

    private lateinit var inputGesture: TextInputEditText
    private lateinit var inputActionType: AutoCompleteTextView
    private lateinit var inputActionValue: TextInputEditText
    private lateinit var listContainer: LinearLayout

    private var selectedTypeIndex = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_custom_mapping)

        inputGesture = findViewById(R.id.inputGesture)
        inputActionType = findViewById(R.id.inputActionType)
        inputActionValue = findViewById(R.id.inputActionValue)
        listContainer = findViewById(R.id.mappingListContainer)

        // Back button
        findViewById<MaterialButton>(R.id.btnBack).setOnClickListener { finish() }

        // Action type dropdown
        val adapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, ACTION_TYPES)
        inputActionType.setAdapter(adapter)
        inputActionType.setOnItemClickListener { _, _, pos, _ ->
            selectedTypeIndex = pos
            // Update hint based on selected type
            inputActionValue.hint = when (ACTION_TYPE_KEYS[pos]) {
                "call" -> "전화번호 (예: 01012345678)"
                "sms" -> "전화번호 (예: 01012345678)"
                "url" -> "URL (예: https://google.com)"
                "search" -> "검색어 (예: 오늘 날씨)"
                "app" -> "패키지명 (예: com.kakao.talk)"
                else -> "값 입력"
            }
        }

        // Add button
        findViewById<MaterialButton>(R.id.btnAddMapping).setOnClickListener {
            val gesture = inputGesture.text.toString().trim().uppercase()
            val value = inputActionValue.text.toString().trim()

            if (gesture.isEmpty() || gesture.length != 1) {
                Toast.makeText(this, "글자 1개를 입력해주세요", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (value.isEmpty()) {
                Toast.makeText(this, "값을 입력해주세요", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val mapping = JSONObject().apply {
                put("gesture", gesture)
                put("actionType", ACTION_TYPE_KEYS[selectedTypeIndex])
                put("actionLabel", ACTION_TYPES[selectedTypeIndex])
                put("value", value)
            }

            saveMappingToList(gesture, mapping)
            inputGesture.text?.clear()
            inputActionValue.text?.clear()
            Toast.makeText(this, "✅ '$gesture' 매핑 저장됨", Toast.LENGTH_SHORT).show()
            refreshList()
        }

        refreshList()
    }

    private fun saveMappingToList(gesture: String, mapping: JSONObject) {
        val prefs = getSharedPreferences("airwriting_prefs", MODE_PRIVATE)
        val json = prefs.getString(PREFS_KEY, "[]") ?: "[]"
        val arr = JSONArray(json)

        // Remove existing mapping for the same gesture
        val newArr = JSONArray()
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            if (obj.getString("gesture") != gesture) newArr.put(obj)
        }
        newArr.put(mapping)
        prefs.edit().putString(PREFS_KEY, newArr.toString()).apply()
    }

    private fun removeMappingFromList(gesture: String) {
        val prefs = getSharedPreferences("airwriting_prefs", MODE_PRIVATE)
        val json = prefs.getString(PREFS_KEY, "[]") ?: "[]"
        val arr = JSONArray(json)
        val newArr = JSONArray()
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            if (obj.getString("gesture") != gesture) newArr.put(obj)
        }
        prefs.edit().putString(PREFS_KEY, newArr.toString()).apply()
    }

    private fun refreshList() {
        listContainer.removeAllViews()
        val mappings = loadMappings(this)

        if (mappings.isEmpty()) {
            val empty = TextView(this).apply {
                text = "커스텀 매핑이 없습니다.\n위에서 추가해보세요!"
                setTextColor(0xFF666666.toInt())
                textSize = 14f
                gravity = Gravity.CENTER
                setPadding(0, 40, 0, 40)
            }
            listContainer.addView(empty)
            return
        }

        for ((gesture, obj) in mappings) {
            val card = MaterialCardView(this).apply {
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = 8 }
                setCardBackgroundColor(0xFFFFFFFF.toInt())
                radius = 12f
                strokeColor = 0xFFE0E0E0.toInt()
                strokeWidth = 1
                cardElevation = 4f
            }

            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(16, 14, 16, 14)
            }

            // Gesture badge
            val badge = TextView(this).apply {
                text = gesture
                setTextColor(0xFFFFFFFF.toInt())
                textSize = 18f
                setBackgroundColor(0xFF1A1A1A.toInt())
                gravity = Gravity.CENTER
                setPadding(16, 8, 16, 8)
                layoutParams = LinearLayout.LayoutParams(48, 48).apply { marginEnd = 12 }
            }
            row.addView(badge)

            // Info
            val info = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            info.addView(TextView(this).apply {
                text = obj.optString("actionLabel", "")
                setTextColor(0xFF1A1A1A.toInt())
                textSize = 14f
            })
            info.addView(TextView(this).apply {
                text = obj.optString("value", "")
                setTextColor(0xFF777777.toInt())
                textSize = 12f
            })
            row.addView(info)

            // Delete button
            val btn = MaterialButton(this).apply {
                text = "✕"
                setTextColor(0xFFFF4444.toInt())
                setBackgroundColor(0x00000000)
                minimumWidth = 0
                minWidth = 0
                setPadding(8, 0, 8, 0)
                setOnClickListener {
                    removeMappingFromList(gesture)
                    refreshList()
                    Toast.makeText(this@CustomMappingActivity, "'$gesture' 삭제됨", Toast.LENGTH_SHORT).show()
                }
            }
            row.addView(btn)

            card.addView(row)
            listContainer.addView(card)
        }
    }
}
