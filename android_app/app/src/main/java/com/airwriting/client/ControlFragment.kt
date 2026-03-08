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
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.textfield.TextInputEditText
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanIntentResult
import com.journeyapps.barcodescanner.ScanOptions
import java.text.SimpleDateFormat
import java.util.*

class ControlFragment : Fragment() {

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
    private lateinit var recognitionCard: MaterialCardView
    private lateinit var txtRecognizedLetter: TextView
    private lateinit var txtRecognitionDetails: TextView
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
        else appendLog("⚠️ Notification denied")
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.fragment_control, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val ctx = requireContext()

        ipInput = view.findViewById(R.id.ipInput)
        btnConnect = view.findViewById(R.id.btnConnect)
        btnScanQr = view.findViewById(R.id.btnScanQr)
        btnToggleManual = view.findViewById(R.id.btnToggleManual)
        btnDisconnect = view.findViewById(R.id.btnDisconnect)
        manualInputContainer = view.findViewById(R.id.manualInputContainer)
        statusText = view.findViewById(R.id.statusText)
        statusDot = view.findViewById(R.id.statusDot)
        statusContainer = view.findViewById(R.id.statusContainer)
        logText = view.findViewById(R.id.logText)
        logScroll = view.findViewById(R.id.logScroll)
        recognitionCard = view.findViewById(R.id.recognitionCard)
        txtRecognizedLetter = view.findViewById(R.id.txtRecognizedLetter)
        txtRecognitionDetails = view.findViewById(R.id.txtRecognitionDetails)
        statsRow = view.findViewById(R.id.statsRow)
        statActionCount = view.findViewById(R.id.statActionCount)
        statUptime = view.findViewById(R.id.statUptime)
        statLastAction = view.findViewById(R.id.statLastAction)

        val prefs = ctx.getSharedPreferences("airwriting_prefs", android.content.Context.MODE_PRIVATE)
        ipInput.setText(prefs.getString("last_ip", ""))

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                notifPermLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && !Settings.canDrawOverlays(ctx)) {
            appendLog("⚠️ '다른 앱 위에 표시' 권한이 필요합니다")
            startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:${ctx.packageName}")))
        }

        btnScanQr.setOnClickListener {
            barcodeLauncher.launch(ScanOptions().apply {
                setDesiredBarcodeFormats(ScanOptions.QR_CODE)
                setPrompt("Scan QR code on PC dashboard")
                setCameraId(0); setBeepEnabled(true); setOrientationLocked(true)
            })
        }

        var manualVisible = false
        btnToggleManual.setOnClickListener {
            manualVisible = !manualVisible
            manualInputContainer.visibility = if (manualVisible) View.VISIBLE else View.GONE
            btnToggleManual.text = if (manualVisible) "IP 입력 ▲" else "직접 IP 입력 ▼"
        }

        btnConnect.setOnClickListener {
            val ip = ipInput.text.toString().trim()
            if (ip.isNotEmpty()) { saveIp(ip); startBackgroundService(ip) }
        }

        btnDisconnect.setOnClickListener {
            val intent = Intent(ctx, WebSocketService::class.java).apply { action = WebSocketService.ACTION_STOP }
            ctx.startService(intent)
            setStatus("disconnected")
            appendLog("🛑 Disconnected by user")
            showDashboard(false)
        }

        var logVisible = true
        view.findViewById<MaterialButton>(R.id.btnToggleLog).setOnClickListener {
            logVisible = !logVisible
            logScroll.visibility = if (logVisible) View.VISIBLE else View.GONE
            (it as MaterialButton).text = if (logVisible) "▲" else "▼"
        }
        view.findViewById<MaterialButton>(R.id.btnClearLog).setOnClickListener {
            logText.text = ""; appendLog("Log cleared")
        }

        WebSocketService.onLog = { msg -> appendLog(msg) }
        WebSocketService.onStatusChange = { state -> setStatus(state) }
        WebSocketService.onRecognition = { label, details -> showRecognition(label, details) }
        WebSocketService.onActionDispatched = { keyword -> onActionDispatched(keyword) }

        if (WebSocketService.isRunning) {
            setStatus("connected"); showDashboard(true)
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
        recognitionCard.visibility = vis; statsRow.visibility = vis; btnDisconnect.visibility = vis
        if (show) {
            connectTime = System.currentTimeMillis(); actionCount = 0
            statActionCount.text = "0"; statUptime.text = "0:00"; statLastAction.text = "—"
            uptimeHandler.post(uptimeRunnable)
        } else { connectTime = 0; uptimeHandler.removeCallbacks(uptimeRunnable) }
    }

    private fun showRecognition(label: String, details: String) {
        activity?.runOnUiThread {
            recognitionCard.visibility = View.VISIBLE
            txtRecognizedLetter.text = label; txtRecognitionDetails.text = details
        }
    }

    private fun onActionDispatched(keyword: String) {
        activity?.runOnUiThread {
            actionCount++; statActionCount.text = actionCount.toString(); statLastAction.text = keyword
        }
    }

    private fun startBackgroundService(ip: String) {
        appendLog("Starting background service for $ip...")
        val intent = Intent(requireContext(), WebSocketService::class.java).apply {
            putExtra(WebSocketService.EXTRA_IP, ip)
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) requireContext().startForegroundService(intent)
            else requireContext().startService(intent)
            appendLog("🚀 Background service started!"); showDashboard(true)
        } catch (e: Exception) { appendLog("❌ Failed: ${e.message}") }
    }

    private fun appendLog(msg: String) {
        val t = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        activity?.runOnUiThread {
            logText.append("[$t] $msg\n")
            logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private fun setStatus(state: String) {
        activity?.runOnUiThread {
            when (state) {
                "connected" -> {
                    statusText.text = "Connected"; statusText.setTextColor(Color.parseColor("#10B981"))
                    statusContainer.setBackgroundResource(R.drawable.status_bg_connected)
                    statusDot.setBackgroundColor(Color.parseColor("#10B981"))
                    btnDisconnect.visibility = View.VISIBLE; showDashboard(true)
                }
                "connecting" -> {
                    statusText.text = "Connecting..."; statusText.setTextColor(Color.parseColor("#F59E0B"))
                    statusContainer.setBackgroundResource(R.drawable.status_bg_connecting)
                    statusDot.setBackgroundColor(Color.parseColor("#F59E0B"))
                }
                else -> {
                    statusText.text = "Not Connected"; statusText.setTextColor(Color.parseColor("#EF4444"))
                    statusContainer.setBackgroundResource(R.drawable.status_bg_disconnected)
                    statusDot.setBackgroundColor(Color.parseColor("#EF4444"))
                    btnDisconnect.visibility = View.GONE
                }
            }
        }
    }

    private fun saveIp(ip: String) {
        requireContext().getSharedPreferences("airwriting_prefs", android.content.Context.MODE_PRIVATE)
            .edit().putString("last_ip", ip).apply()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        uptimeHandler.removeCallbacks(uptimeRunnable)
    }
}
