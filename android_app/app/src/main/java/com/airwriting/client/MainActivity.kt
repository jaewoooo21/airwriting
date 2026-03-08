package com.airwriting.client

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.google.android.material.bottomnavigation.BottomNavigationView

class MainActivity : AppCompatActivity() {

    private val platformFragment = PlatformFragment()
    private val controlFragment = ControlFragment()
    private val actionsFragment = ActionsFragment()
    private val historyFragment = HistoryFragment()
    private var activeFragment: Fragment = controlFragment

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

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
}
