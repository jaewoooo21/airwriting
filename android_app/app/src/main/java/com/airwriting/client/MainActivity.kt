package com.airwriting.client

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.android.material.bottomnavigation.BottomNavigationView

class MainActivity : AppCompatActivity() {

    private val PERMISSION_REQUEST_CODE = 100
    private val platformFragment = PlatformFragment()
    private val controlFragment = ControlFragment()
    private val actionsFragment = ActionsFragment()
    private val historyFragment = HistoryFragment()
    private var activeFragment: Fragment = controlFragment

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        checkPermissions()

        val bottomNav = findViewById<BottomNavigationView>(R.id.bottomNav)

        // Add all fragments, hide non-active ones
        supportFragmentManager.beginTransaction()
            .add(R.id.fragmentContainer, historyFragment, "history").hide(historyFragment)
            .add(R.id.fragmentContainer, actionsFragment, "actions").hide(actionsFragment)
            .add(R.id.fragmentContainer, platformFragment, "platform").hide(platformFragment)
            .add(R.id.fragmentContainer, controlFragment, "control")
            .commit()

        // Start on Control tab
        bottomNav.selectedItemId = R.id.nav_control

        bottomNav.setOnItemSelectedListener { item ->
            val target = when (item.itemId) {
                R.id.nav_platform -> platformFragment
                R.id.nav_control -> controlFragment
                R.id.nav_actions -> actionsFragment
                R.id.nav_history -> historyFragment
                else -> controlFragment
            }
            supportFragmentManager.beginTransaction()
                .hide(activeFragment)
                .show(target)
                .commit()
            activeFragment = target
            true
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        WebSocketService.onLog = null
        WebSocketService.onStatusChange = null
        WebSocketService.onRecognition = null
        WebSocketService.onActionDispatched = null
    }

    private fun checkPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.CALL_PHONE,
            Manifest.permission.CAMERA
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val toRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (toRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, toRequest.toTypedArray(), PERMISSION_REQUEST_CODE)
        }
    }
}
