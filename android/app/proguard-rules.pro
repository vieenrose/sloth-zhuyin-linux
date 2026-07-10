# Keep the JNI bridge (native methods resolved by exact name).
-keep class com.slothing.ime.NativeBridge { *; }
# ONNX Runtime Java API.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
