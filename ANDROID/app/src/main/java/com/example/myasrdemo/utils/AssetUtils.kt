package com.example.myasrdemo.utils

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream

object AssetUtils {
    private const val TAG = "AssetUtils"

    fun copyAssets(context: Context) {
        val assets = arrayOf("model.int8.onnx", "tokens.txt", "silero_vad.onnx")
        for (filename in assets) {
            copyAssetFile(context, filename)
        }
    }

    private fun copyAssetFile(context: Context, filename: String) {
        val file = File(context.filesDir, filename)
        if (file.exists()) {
            Log.d(TAG, "File already exists: $filename")
            return
        }

        try {
            context.assets.open(filename).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    copyFile(inputStream, outputStream)
                }
            }
            Log.d(TAG, "Copied asset: $filename to ${file.absolutePath}")
        } catch (e: IOException) {
            Log.e(TAG, "Failed to copy asset: $filename", e)
        }
    }

    private fun copyFile(inputStream: InputStream, outputStream: OutputStream) {
        val buffer = ByteArray(8192) // 8KB buffer
        var read: Int
        while (inputStream.read(buffer).also { read = it } != -1) {
            outputStream.write(buffer, 0, read)
        }
    }
    
    fun getModelPath(context: Context): String {
        return File(context.filesDir, "model.int8.onnx").absolutePath
    }
    
    fun getTokensPath(context: Context): String {
        return File(context.filesDir, "tokens.txt").absolutePath
    }
    
    fun getVadPath(context: Context): String {
        return File(context.filesDir, "silero_vad.onnx").absolutePath
    }
}
