package com.airwriting.client

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.View
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.textfield.TextInputEditText
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanIntentResult
import com.journeyapps.barcodescanner.ScanOptions
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    private lateinit var ipInput: TextInputEditText
    private lateinit var btnConnect: MaterialButton
    private lateinit var btnScanQr: MaterialButton
    private lateinit var btnToggleManual: MaterialButton
    private lateinit var btnDisconnect: MaterialButton
    private lateinit var manualInputContainer: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var statusDot: View
    private lateinit var statusContainer: LinearLayout
    private lateinit var logText: TextView
    private lateinit var logScroll: ScrollView
    // Recognition
    private lateinit var recognitionCard: MaterialCardView
    private lateinit var txtRecognizedLetter: TextView
    private lateinit var txtRecognitionDetails: TextView
    // Stats
    private lateinit var statsRow: LinearLayout
    private lateinit var statActionCount: TextView
    private lateinit var statUptime: TextView
    private lateinit var statLastAction: TextView

    private var actionCount = 0
    private var connectTime: Long = 0L
    private val uptimeHandler = Handler(Looper.getMainLooper())
    private val uptimeRunnable = object : Runnable {
        override fun run() {
            if (connectTime > 0) {
                val elapsed = (System.currentTimeMillis() - connectTime) / 1000
                val m = elapsed / 60
                val s = elapsed % 60
                statUptime.text = String.format("%d:%02d", m, s)
            }
            uptimeHandler.postDelayed(this, 1000)
        }
    }

    // QR Scanner
    private val barcodeLauncher = registerForActivityResult(ScanContract()) { result: ScanIntentResult ->
        if (result.contents != null) {
            appendLog("📷 QR: ${result.contents}")
            val ip = result.contents.removePrefix("ws://").removePrefix("wss://").substringBefore(":")
            if (ip.isNotEmpty()) {
                ipInput.setText(ip)
                saveIp(ip)
                startBackgroundService(ip)
            }
        }
    }

    private val notifPermLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) appendLog("✅ Notification permission granted")
        else appendLog("⚠️ Notification denied — background service may not show status")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ipInput = findViewById(R.id.ipInput)
        btnConnect = findViewById(R.id.btnConnect)
        btnScanQr = findViewById(R.id.btnScanQr)
        btnToggleManual = findViewById(R.id.btnToggleManual)
        btnDisconnect = findViewById(R.id.btnDisconnect)
        manualInputContainer = findViewById(R.id.manualInputContainer)
        statusText = findViewById(R.id.statusText)
        statusDot = findViewById(R.id.statusDot)
        statusContainer = findViewById(R.id.statusContainer)
        logText = findViewById(R.id.logText)
        logScroll = findViewById(R.id.logScroll)
        recognitionCard = findViewById(R.id.recognitionCard)
        txtRecognizedLetter = findViewById(R.id.txtRecognizedLetter)
        txtRecognitionDetails = findViewById(R.id.txtRecognitionDetails)
        statsRow = findViewById(R.id.statsRow)
        statActionCount = findViewById(R.id.statActionCount)
        statUptime = findViewById(R.id.statUptime)
        statLastAction = findViewById(R.id.statLastAction)

        val prefs = getSharedPreferences("airwriting_prefs", MODE_PRIVATE)
        ipInput.setText(prefs.getString("last_ip", ""))

        // Request notification permission (Android 13+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                notifPermLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        // Request "Draw over other apps" permission
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && !Settings.canDrawOverlays(this)) {
            appendLog("⚠️ '다른 앱 위에 표시' 권한이 필요합니다")
            val overlayIntent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")
            )
            startActivity(overlayIntent)
        }

        // ── QR Scan ──
        btnScanQr.setOnClickListener {
            barcodeLauncher.launch(ScanOptions().apply {
                setDesiredBarcodeFormats(ScanOptions.QR_CODE)
                setPrompt("Scan QR code on PC dashboard")
                setCameraId(0)
                setBeepEnabled(true)
                setOrientationLocked(true)
            })
        }

        // ── Toggle manual input ──
        var manualVisible = false
        btnToggleManual.setOnClickListener {
            manualVisible = !manualVisible
            manualInputContainer.visibility = if (manualVisible) View.VISIBLE else View.GONE
            btnToggleManual.text = if (manualVisible) "IP 입력 ▲" else "직접 IP 입력 ▼"
        }

        // ── Manual Connect ──
        btnConnect.setOnClickListener {
            val ip = ipInput.text.toString().trim()
            if (ip.isNotEmpty()) {
                saveIp(ip)
                startBackgroundService(ip)
            }
        }

        // ── Disconnect ──
        btnDisconnect.setOnClickListener {
            val intent = Intent(this, WebSocketService::class.java).apply {
                action = WebSocketService.ACTION_STOP
            }
            startService(intent)
            setStatus("disconnected")
            appendLog("🛑 Disconnected by user")
            showDashboard(false)
        }

        // ── Custom Mappings ──
        findViewById<MaterialButton>(R.id.btnCustomMapping).setOnClickListener {
            startActivity(Intent(this, CustomMappingActivity::class.java))
        }

        // ── Action History ──
        findViewById<MaterialButton>(R.id.btnHistory).setOnClickListener {
            startActivity(Intent(this, ActionHistoryActivity::class.java))
        }

        // ── Log controls ──
        var logVisible = true
        findViewById<MaterialButton>(R.id.btnToggleLog).setOnClickListener {
            logVisible = !logVisible
            logScroll.visibility = if (logVisible) View.VISIBLE else View.GONE
            (it as MaterialButton).text = if (logVisible) "▲" else "▼"
        }
        findViewById<MaterialButton>(R.id.btnClearLog).setOnClickListener {
            logText.text = ""
            appendLog("Log cleared")
        }

        // ── Service callbacks ──
        WebSocketService.onLog = { msg -> appendLog(msg) }
        WebSocketService.onStatusChange = { state -> setStatus(state) }
        WebSocketService.onRecognition = { label, details -> showRecognition(label, details) }
        WebSocketService.onActionDispatched = { keyword -> onActionDispatched(keyword) }

        if (WebSocketService.isRunning) {
            setStatus("connected")
            showDashboard(true)
            appendLog("Service running in background ✅")
        } else {
            appendLog("Ready. Scan QR or enter IP to connect.")
        }
    }

    override fun onResume() {
        super.onResume()
        WebSocketService.onLog = { msg -> appendLog(msg) }
        WebSocketService.onStatusChange = { state -> setStatus(state) }
        WebSocketService.onRecognition = { label, details -> showRecognition(label, details) }
        WebSocketService.onActionDispatched = { keyword -> onActionDispatched(keyword) }
    }

    private fun showDashboard(show: Boolean) {
        val vis = if (show) View.VISIBLE else View.GONE
        recognitionCard.visibility = vis
        statsRow.visibility = vis
        btnDisconnect.visibility = vis
        if (show) {
            connectTime = System.currentTimeMillis()
            actionCount = 0
            statActionCount.text = "0"
            statUptime.text = "0:00"
            statLastAction.text = "—"
            uptimeHandler.post(uptimeRunnable)
        } else {
            connectTime = 0
            uptimeHandler.removeCallbacks(uptimeRunnable)
        }
    }

    private fun showRecognition(label: String, details: String) {
        runOnUiThread {
            recognitionCard.visibility = View.VISIBLE
            txtRecognizedLetter.text = label
            txtRecognitionDetails.text = details
        }
    }

    private fun onActionDispatched(keyword: String) {
        runOnUiThread {
            actionCount++
            statActionCount.text = actionCount.toString()
            statLastAction.text = keyword
        }
    }

    private fun startBackgroundService(ip: String) {
        appendLog("Starting background service for $ip...")
        val intent = Intent(this, WebSocketService::class.java).apply {
            putExtra(WebSocketService.EXTRA_IP, ip)
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
            appendLog("🚀 Background service started!")
            showDashboard(true)
        } catch (e: Exception) {
            appendLog("❌ Failed to start service: ${e.message}")
        }
    }

    private fun appendLog(msg: String) {
        val t = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        runOnUiThread {
            logText.append("[$t] $msg\n")
            logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private fun setStatus(state: String) {
        runOnUiThread {
            when (state) {
                "connected" -> {
                    statusText.text = "Connected"
                    statusText.setTextColor(Color.parseColor("#2E7D32"))
                    statusContainer.setBackgroundResource(R.drawable.status_bg_connected)
                    statusDot.setBackgroundColor(Color.parseColor("#2E7D32"))
                    btnDisconnect.visibility = View.VISIBLE
                    showDashboard(true)
                }
                "connecting" -> {
                    statusText.text = "Connecting..."
                    statusText.setTextColor(Color.parseColor("#F57F17"))
                    statusContainer.setBackgroundResource(R.drawable.status_bg_connecting)
                    statusDot.setBackgroundColor(Color.parseColor("#F57F17"))
                }
                else -> {
                    statusText.text = "Not Connected"
                    statusText.setTextColor(Color.parseColor("#C62828"))
                    statusContainer.setBackgroundResource(R.drawable.status_bg_disconnected)
                    statusDot.setBackgroundColor(Color.parseColor("#C62828"))
                    btnDisconnect.visibility = View.GONE
                }
            }
        }
    }

    private fun saveIp(ip: String) {
        getSharedPreferences("airwriting_prefs", MODE_PRIVATE)
            .edit().putString("last_ip", ip).apply()
    }

    override fun onDestroy() {
        super.onDestroy()
        WebSocketService.onLog = null
        WebSocketService.onStatusChange = null
        WebSocketService.onRecognition = null
        WebSocketService.onActionDispatched = null
        uptimeHandler.removeCallbacks(uptimeRunnable)
    }
}
