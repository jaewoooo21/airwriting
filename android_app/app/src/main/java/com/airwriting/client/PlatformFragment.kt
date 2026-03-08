package com.airwriting.client

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.webkit.*
import android.widget.LinearLayout
import androidx.fragment.app.Fragment

class PlatformFragment : Fragment() {

    private var webView: WebView? = null
    private var loadingContainer: LinearLayout? = null

    // Deployed platform URL — change to local if needed
    companion object {
        // Try deployed URL first, fallback to local server
        const val PLATFORM_URL = "https://airwriting.onrender.com"
        // const val PLATFORM_URL = "http://192.168.x.x:5050"
    }

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.fragment_platform, container, false)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        webView = view.findViewById(R.id.platformWebView)
        loadingContainer = view.findViewById(R.id.loadingContainer)

        webView?.apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = true
            settings.mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            settings.loadWithOverviewMode = true
            settings.useWideViewPort = true
            settings.builtInZoomControls = true
            settings.displayZoomControls = false
            settings.setSupportZoom(true)
            settings.cacheMode = WebSettings.LOAD_DEFAULT
            settings.mediaPlaybackRequiresUserGesture = false

            webViewClient = object : WebViewClient() {
                override fun onPageFinished(view: WebView?, url: String?) {
                    super.onPageFinished(view, url)
                    loadingContainer?.visibility = View.GONE
                }

                override fun onReceivedError(view: WebView?, request: WebResourceRequest?, error: WebResourceError?) {
                    super.onReceivedError(view, request, error)
                    // If main frame failed, show error
                    if (request?.isForMainFrame == true) {
                        loadingContainer?.visibility = View.GONE
                        view?.loadData(
                            "<html><body style='display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;background:#FAFBFC;color:#334155;text-align:center;'>" +
                            "<div><h2>⚠️ 연결 실패</h2><p>플랫폼 서버에 접속할 수 없습니다.<br>PC에서 서버를 실행하거나<br>인터넷 연결을 확인해주세요.</p>" +
                            "<p style='color:#94A3B8;font-size:12px;margin-top:20px;'>URL: $PLATFORM_URL</p></div></body></html>",
                            "text/html", "UTF-8"
                        )
                    }
                }

                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest?): Boolean {
                    // Keep navigation inside WebView
                    return false
                }
            }

            webChromeClient = WebChromeClient()

            // Load the platform
            loadUrl(PLATFORM_URL)
        }
    }

    override fun onPause() {
        super.onPause()
        webView?.onPause()
    }

    override fun onResume() {
        super.onResume()
        webView?.onResume()
    }

    override fun onDestroyView() {
        webView?.destroy()
        webView = null
        super.onDestroyView()
    }
}
