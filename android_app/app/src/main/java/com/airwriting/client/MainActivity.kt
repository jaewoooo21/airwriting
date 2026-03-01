package com.airwriting.client

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.View
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
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
    private lateinit var manualInputContainer: LinearLayout
    private lateinit var statusText: TextView
    private lateinit var statusDot: View
    private lateinit var statusContainer: LinearLayout
    private lateinit var logText: TextView
    private lateinit var logScroll: ScrollView

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
        manualInputContainer = findViewById(R.id.manualInputContainer)
        statusText = findViewById(R.id.statusText)
        statusDot = findViewById(R.id.statusDot)
        statusContainer = findViewById(R.id.statusContainer)
        logText = findViewById(R.id.logText)
        logScroll = findViewById(R.id.logScroll)

        val prefs = getSharedPreferences("airwriting_prefs", MODE_PRIVATE)
        ipInput.setText(prefs.getString("last_ip", ""))

        // Request notification permission (Android 13+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                notifPermLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        // Request "Draw over other apps" permission (required to launch apps from background)
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

        // ── Custom Mappings ──
        findViewById<MaterialButton>(R.id.btnCustomMapping).setOnClickListener {
            startActivity(Intent(this, CustomMappingActivity::class.java))
        }

        // ── Service callbacks for UI updates ──
        WebSocketService.onLog = { msg -> appendLog(msg) }
        WebSocketService.onStatusChange = { state -> setStatus(state) }

        if (WebSocketService.isRunning) {
            setStatus("connected")
            appendLog("Service running in background ✅")
        } else {
            appendLog("Ready. Scan QR or enter IP to connect.")
        }
    }

    override fun onResume() {
        super.onResume()
        // Re-attach callbacks when returning to the app
        WebSocketService.onLog = { msg -> appendLog(msg) }
        WebSocketService.onStatusChange = { state -> setStatus(state) }
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
                }
                "connecting" -> {
                    statusText.text = "Connecting..."
                    statusText.setTextColor(Color.parseColor("#F57F17"))
                    statusDot.setBackgroundColor(Color.parseColor("#F57F17"))
                }
                else -> {
                    statusText.text = "Not Connected"
                    statusText.setTextColor(Color.parseColor("#C62828"))
                    statusContainer.setBackgroundResource(R.drawable.status_bg_disconnected)
                    statusDot.setBackgroundColor(Color.parseColor("#C62828"))
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
        // Don't stop the service! Just detach UI callbacks
        WebSocketService.onLog = null
        WebSocketService.onStatusChange = null
        // Service continues running in background
    }
}
