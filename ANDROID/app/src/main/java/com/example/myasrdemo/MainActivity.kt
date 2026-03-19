package com.example.myasrdemo

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.myasrdemo.ui.theme.MyasrdemoTheme
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {
    private val TAG = "StreamActivity"
    private val PERMISSIONS_REQUEST_CODE = 10
    private val PERMISSIONS_REQUIRED = arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)

    private var webSocket: WebSocket? = null
    private var isStreaming by mutableStateOf(false)
    private var connectionStatus by mutableStateOf("未连接")
    private var asrResult by mutableStateOf("")
    private var serverIp by mutableStateOf("10.0.2.2") // Default emulator IP

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var audioRecord: AudioRecord? = null
    private var audioThread: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(this, PERMISSIONS_REQUIRED, PERMISSIONS_REQUEST_CODE)
        }

        setContent {
            MyasrdemoTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    StreamScreen(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    private fun hasPermissions(context: Context) = PERMISSIONS_REQUIRED.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    @Composable
    fun StreamScreen(modifier: Modifier = Modifier) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current

        Column(modifier = modifier.fillMaxSize()) {
            // Camera Preview Area (Weight 1 to fill available space)
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .background(Color.Black)
            ) {
                if (hasPermissions(context)) {
                    AndroidView(
                        factory = { ctx ->
                            PreviewView(ctx).apply {
                                layoutParams = ViewGroup.LayoutParams(
                                    ViewGroup.LayoutParams.MATCH_PARENT,
                                    ViewGroup.LayoutParams.MATCH_PARENT
                                )
                                scaleType = PreviewView.ScaleType.FILL_CENTER
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                        update = { previewView ->
                            if (isStreaming) {
                                startCamera(context, lifecycleOwner, previewView)
                            } else {
                                // Stop camera logic if needed, but keeping it warm is fine
                                startCamera(context, lifecycleOwner, previewView) 
                            }
                        }
                    )
                } else {
                    Text("需要相机权限", color = Color.White, modifier = Modifier.align(Alignment.Center))
                }

                // Overlay ASR Results
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .height(200.dp)
                        .background(Color(0xAA000000))
                        .padding(16.dp)
                ) {
                    val scrollState = rememberScrollState()
                    // 始终滚动到顶部 (因为最新的内容被插入到了最前面)
                    LaunchedEffect(asrResult) {
                        scrollState.animateScrollTo(0)
                    }
                    Column(modifier = Modifier.verticalScroll(scrollState)) {
                        Text(
                            text = asrResult,
                            color = Color.White,
                            style = MaterialTheme.typography.bodyLarge
                        )
                    }
                }
            }

            // Controls Area
            Column(
                modifier = Modifier
                    .padding(16.dp)
                    .fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                OutlinedTextField(
                    value = serverIp,
                    onValueChange = { serverIp = it },
                    label = { Text("服务器 IP (如 192.168.1.5)") },
                    modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp)
                )

                Text("状态: $connectionStatus", modifier = Modifier.padding(bottom = 8.dp))

                Button(
                    onClick = {
                        if (isStreaming) {
                            stopStreaming()
                        } else {
                            startStreaming()
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isStreaming) Color.Red else MaterialTheme.colorScheme.primary
                    )
                ) {
                    Text(if (isStreaming) "停止直播" else "开始直播")
                }
            }
        }
    }

    private fun startCamera(context: Context, lifecycleOwner: androidx.lifecycle.LifecycleOwner, previewView: PreviewView) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // Preview
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Image Analysis (Video Stream)
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                if (isStreaming && webSocket != null) {
                    // Convert YUV to JPEG
                    val yBuffer = imageProxy.planes[0].buffer
                    val uBuffer = imageProxy.planes[1].buffer
                    val vBuffer = imageProxy.planes[2].buffer

                    val ySize = yBuffer.remaining()
                    val uSize = uBuffer.remaining()
                    val vSize = vBuffer.remaining()

                    val nv21 = ByteArray(ySize + uSize + vSize)

                    // U and V are swapped
                    yBuffer.get(nv21, 0, ySize)
                    vBuffer.get(nv21, ySize, vSize)
                    uBuffer.get(nv21, ySize + vSize, uSize)

                    val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
                    val out = ByteArrayOutputStream()
                    yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 50, out)
                    val jpegBytes = out.toByteArray()

                    // Send Header (0x01) + JPEG
                    val packet = ByteArray(1 + jpegBytes.size)
                    packet[0] = 0x01
                    System.arraycopy(jpegBytes, 0, packet, 1, jpegBytes.size)
                    
                    try {
                        webSocket?.send(packet.toByteString())
                    } catch (e: Exception) {
                        Log.e(TAG, "Video send failed", e)
                    }
                }
                imageProxy.close()
            }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner, 
                    CameraSelector.DEFAULT_BACK_CAMERA, 
                    preview, 
                    imageAnalysis
                )
            } catch(exc: Exception) {
                Log.e(TAG, "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    private fun startStreaming() {
        if (serverIp.isBlank()) {
            Toast.makeText(this, "请输入服务器 IP", Toast.LENGTH_SHORT).show()
            return
        }

        connectionStatus = "正在连接..."
        val request = Request.Builder().url("ws://$serverIp:8000/ws").build()
        
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                isStreaming = true
                runOnUiThread { connectionStatus = "已连接 (直播中)" }
                startAudioStreaming()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    if (json.optString("type") == "result") {
                        val text = json.optString("text")
                        runOnUiThread { 
                            if (asrResult.isNotEmpty()) {
                                asrResult = text + "\n" + asrResult
                            } else {
                                asrResult = text
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "JSON parse error", e)
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                stopStreaming()
                runOnUiThread { connectionStatus = "连接关闭: $reason" }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                stopStreaming()
                val errorMsg = t.message ?: t.javaClass.simpleName
                runOnUiThread { connectionStatus = "连接失败: $errorMsg" }
                Log.e(TAG, "WS Failure", t)
            }
        })
    }

    private fun stopStreaming() {
        isStreaming = false
        webSocket?.close(1000, "User stopped")
        webSocket = null
        stopAudioStreaming()
        runOnUiThread { connectionStatus = "已停止" }
    }

    private fun startAudioStreaming() {
        audioThread = thread {
            val sampleRate = 16000
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
            val bufferSize = maxOf(minBufferSize, 4096)

            try {
                if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    return@thread
                }

                audioRecord = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    channelConfig,
                    audioFormat,
                    bufferSize
                )

                audioRecord?.startRecording()
                // Silero VAD requires specific chunk sizes: 512 samples for 16000Hz (1024 bytes)
                val buffer = ByteArray(1024) 

                while (isStreaming && webSocket != null) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                    if (read > 0) {
                        // Header (0x00) + Audio
                        val packet = ByteArray(1 + read)
                        packet[0] = 0x00
                        System.arraycopy(buffer, 0, packet, 1, read)
                        
                        try {
                            webSocket?.send(packet.toByteString())
                        } catch (e: Exception) {
                            Log.e(TAG, "Audio send failed", e)
                            break
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Audio streaming error", e)
            } finally {
                audioRecord?.stop()
                audioRecord?.release()
            }
        }
    }
    
    private fun stopAudioStreaming() {
        // Flag isStreaming handles loop exit
        try {
            audioThread?.join(1000)
        } catch (e: Exception) {}
    }
}
