package com.airwriting.client

import android.app.*
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.provider.AlarmClock
import android.provider.MediaStore
import android.util.Log
import androidx.core.app.NotificationCompat
import okhttp3.*
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class WebSocketService : Service() {

    companion object {
        const val TAG = "AirWriting"
        const val CHANNEL_ID = "airwriting_service"
        const val NOTIFICATION_ID = 1
        const val ACTION_STOP = "com.airwriting.STOP"
        const val EXTRA_IP = "server_ip"
        const val HISTORY_KEY = "action_history"
        const val MAX_HISTORY = 50

        var isRunning = false
        var onLog: ((String) -> Unit)? = null
        var onStatusChange: ((String) -> Unit)? = null
        var onRecognition: ((String, String) -> Unit)? = null
        var onActionDispatched: ((String) -> Unit)? = null
    }

    private var webSocket: WebSocket? = null
    private var serverIp = ""
    private var isConnecting = AtomicBoolean(false)
    private var shouldReconnect = true
    private val client = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            shouldReconnect = false
            webSocket?.close(1000, "User stopped")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        val ip = intent?.getStringExtra(EXTRA_IP)
        if (ip.isNullOrEmpty()) return START_NOT_STICKY

        if (isRunning && serverIp == ip && webSocket != null) {
            log("Already connected to $ip, skipping reconnect")
            return START_STICKY
        }

        serverIp = ip
        shouldReconnect = true

        try {
            val notification = buildNotification("Connecting to $serverIp...")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIFICATION_ID, notification,
                    android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
            isRunning = true
        } catch (e: Exception) {
            Log.e(TAG, "Foreground start failed: ${e.message}")
            return START_NOT_STICKY
        }

        webSocket?.close(1000, "New connection")
        webSocket = null
        connectWebSocket()

        return START_STICKY
    }

    private fun connectWebSocket() {
        if (!isConnecting.compareAndSet(false, true)) {
            Log.d(TAG, "Already connecting, skipping")
            return
        }

        val url = "ws://$serverIp:18800"
        log("Connecting → $url")
        onStatusChange?.invoke("connecting")

        val req = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                isConnecting.set(false)
                log("✅ Connected!")
                onStatusChange?.invoke("connected")
                updateNotification("Connected to $serverIp ✅")
            }

            override fun onMessage(ws: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    when (json.optString("type")) {
                        "action" -> {
                            val keyword = json.optString("keyword")
                            val label = json.optString("label")
                            log("🚀 $label → $keyword")

                            ws.send(JSONObject().apply {
                                put("type", "ack")
                                put("keyword", keyword)
                            }.toString())

                            vibrate()
                            dispatchIntent(keyword)
                            saveToHistory(keyword, label)
                            onActionDispatched?.invoke(keyword)
                        }
                        "config" -> log("⚙️ Config received, ready!")
                        "recognition" -> {
                            val label = json.optString("label", "?")
                            val preds = json.optJSONArray("predictions")
                            val details = if (preds != null && preds.length() > 0) {
                                val sb = StringBuilder()
                                for (i in 0 until minOf(preds.length(), 3)) {
                                    val p = preds.getJSONObject(i)
                                    val l = p.optString("label", "?")
                                    val c = (p.optDouble("confidence", 0.0) * 100).toInt()
                                    if (i > 0) sb.append("  ·  ")
                                    sb.append("$l ${c}%")
                                }
                                sb.toString()
                            } else {
                                label
                            }
                            log("👀 $label")
                            onRecognition?.invoke(label, details)
                        }
                    }
                } catch (e: Exception) {
                    log("❌ Parse: ${e.message}")
                }
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                isConnecting.set(false)
                log("Disconnected ($reason)")
                onStatusChange?.invoke("disconnected")
                updateNotification("Disconnected")
                scheduleReconnect()
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                isConnecting.set(false)
                log("Connection failed: ${t.message}")
                onStatusChange?.invoke("disconnected")
                updateNotification("Reconnecting...")
                scheduleReconnect()
            }
        })
    }

    private fun scheduleReconnect() {
        if (!shouldReconnect || !isRunning) return
        Thread {
            Thread.sleep(5000)
            if (shouldReconnect && isRunning && !isConnecting.get()) {
                connectWebSocket()
            }
        }.start()
    }

    private fun vibrate() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vm = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vm.defaultVibrator.vibrate(
                    VibrationEffect.createOneShot(80, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                val v = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    v.vibrate(VibrationEffect.createOneShot(80, VibrationEffect.DEFAULT_AMPLITUDE))
                } else {
                    @Suppress("DEPRECATION")
                    v.vibrate(80)
                }
            }
        } catch (_: Exception) {}
    }

    private fun saveToHistory(keyword: String, label: String) {
        try {
            val prefs = getSharedPreferences("airwriting_prefs", MODE_PRIVATE)
            val json = prefs.getString(HISTORY_KEY, "[]") ?: "[]"
            val arr = JSONArray(json)
            val entry = JSONObject().apply {
                put("keyword", keyword)
                put("label", label)
                put("time", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date()))
            }
            arr.put(entry)
            // Keep only last MAX_HISTORY entries
            val trimmed = if (arr.length() > MAX_HISTORY) {
                val newArr = JSONArray()
                for (i in (arr.length() - MAX_HISTORY) until arr.length()) {
                    newArr.put(arr.getJSONObject(i))
                }
                newArr
            } else arr
            prefs.edit().putString(HISTORY_KEY, trimmed.toString()).apply()
        } catch (_: Exception) {}
    }

    private fun dispatchIntent(keyword: String) {
        // Custom mappings first
        try {
            val letterMap = mapOf(
                "CALL" to "C", "MESSAGE" to "M", "GEMINI" to "G",
                "SEARCH" to "S", "WEATHER" to "W", "PHOTO" to "P",
                "GALLERY" to "V", "TIMER" to "T", "ALARM" to "A",
                "NOTE" to "N", "HOME" to "H", "BIXBY" to "B"
            )
            val letter = letterMap[keyword] ?: keyword
            val custom = ActionsFragment.loadMappings(this)[letter]
            if (custom != null && ActionsFragment.executeCustomAction(this, custom)) {
                log("🎯 Custom: ${custom.optString("value")}")
                return
            }
        } catch (_: Exception) {}

        // Default intents
        try {
            val intent = when (keyword) {
                "CALL" -> Intent(Intent.ACTION_CALL, Uri.parse("tel:"))
                "MESSAGE" -> Intent(Intent.ACTION_MAIN).apply { addCategory(Intent.CATEGORY_APP_MESSAGING) }
                "GEMINI" -> Intent(Intent.ACTION_ASSIST)
                "SEARCH" -> Intent(Intent.ACTION_WEB_SEARCH).apply { putExtra("query", "") }
                "WEATHER" -> Intent(Intent.ACTION_VIEW, Uri.parse("https://m.search.naver.com/search.naver?query=날씨"))
                "PHOTO" -> Intent(MediaStore.INTENT_ACTION_STILL_IMAGE_CAMERA).apply {
                    try { setPackage("com.sec.android.app.camera") } catch (_: Exception) {}
                }
                "GALLERY" -> Intent(Intent.ACTION_VIEW).apply { type = "image/*" }
                "TIMER" -> Intent(AlarmClock.ACTION_SET_TIMER).apply {
                    putExtra(AlarmClock.EXTRA_LENGTH, 60)
                    putExtra(AlarmClock.EXTRA_MESSAGE, "AirWriting")
                    putExtra(AlarmClock.EXTRA_SKIP_UI, false)
                }
                "ALARM" -> Intent(AlarmClock.ACTION_SET_ALARM)
                "NOTE" -> Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_TEXT, "AirWriting Memo")
                    try { setPackage("com.samsung.android.app.notes") } catch (_: Exception) {}
                }
                "HOME" -> Intent(Intent.ACTION_MAIN).apply { addCategory(Intent.CATEGORY_HOME) }
                "BIXBY" -> Intent().apply {
                    action = "android.intent.action.VOICE_COMMAND"
                    try { setPackage("com.samsung.android.bixby.agent") } catch (_: Exception) {}
                }
                else -> null
            }
            if (intent != null) {
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                startActivity(intent)
                log("✅ $keyword launched")
            }
        } catch (e: Exception) {
            log("❌ $keyword: ${e.message}")
        }
    }

    private fun log(msg: String) {
        Log.d(TAG, msg)
        onLog?.invoke(msg)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val chan = NotificationChannel(CHANNEL_ID, "AirWriting", NotificationManager.IMPORTANCE_LOW).apply {
                description = "Gesture connection active"
            }
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(chan)
        }
    }

    private fun buildNotification(text: String): Notification {
        val openPI = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val stopPI = PendingIntent.getService(
            this, 1,
            Intent(this, WebSocketService::class.java).apply { action = ACTION_STOP },
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("✍️ AirWriting")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_launcher)
            .setContentIntent(openPI)
            .addAction(0, "⏹ Stop", stopPI)
            .setOngoing(true)
            .setSilent(true)
            .build()
    }

    private fun updateNotification(text: String) {
        try {
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
                .notify(NOTIFICATION_ID, buildNotification(text))
        } catch (_: Exception) {}
    }

    override fun onDestroy() {
        shouldReconnect = false
        isRunning = false
        webSocket?.close(1000, "Service destroyed")
        onStatusChange?.invoke("disconnected")
        super.onDestroy()
    }
}
