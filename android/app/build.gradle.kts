import org.gradle.api.tasks.Copy

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.slothing.ime"
    compileSdk = 34            // android-34 installed; bump to 35 (android-35 also installed) by editing this line only

    defaultConfig {
        applicationId = "com.slothing.ime"
        minSdk = 30            // Android 11, BOOX Tab Mini C
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"

        // arm64-v8a ONLY — the only ABI the Tab Mini C needs.
        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
                // ORT prebuilt is built against the shared STL.
                arguments += "-DANDROID_STL=c++_shared"
                // Point app/CMakeLists.txt at the vendored ONNX Runtime (headers + arm64 .so).
                arguments += "-DORT_ROOT=${projectDir}/ort"
            }
        }
    }

    // Pin the NDK + CMake that are actually installed under ~/Android/Sdk.
    ndkVersion = "27.2.12479018"

    externalNativeBuild {
        cmake {
            // Canonical native build owned by the decode-port/native agent. Target "slothing"
            // -> libslothing.so, matching Core.kt's System.loadLibrary("slothing"). It compiles
            // app/cpp/{jni_slothing,onnx_decoder}.cpp + engine/common headers, and links the
            // vendored ONNX Runtime at -DORT_ROOT.
            path = file("CMakeLists.txt")
            version = "3.22.1"   // the ONLY cmake installed; without this AGP tries to fetch another
        }
    }

    // Curated asset staging happens via the copyModelAssets task below,
    // but keep the default assets dir too (holds the staged files + any static assets).
    sourceSets["main"].assets.srcDirs("src/main/assets")

    // Ship the vendored libonnxruntime.so (app/ort/lib/arm64-v8a/) inside the APK.
    // Layout <srcDir>/<abi>/lib*.so, so "ort/lib" resolves ort/lib/arm64-v8a/libonnxruntime.so.
    sourceSets["main"].jniLibs.srcDirs("ort/lib")

    buildTypes {
        getByName("debug") {
            isMinifyEnabled = false
            isJniDebuggable = true
        }
        getByName("release") {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        // Run the build on JDK 21, emit Java 17 bytecode (the Android-standard target).
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // Do NOT compress the model; ORT/mmap wants it uncompressed in the APK.
    androidResources {
        noCompress += listOf("onnx", "tsv", "json")
    }

    packaging {
        // Two arm64 libc++_shared.so can collide (ORT ships one); keep the first.
        jniLibs.pickFirsts += "**/libc++_shared.so"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")

    // Core.kt hops heavy ONNX calls to Dispatchers.Default via withContext.
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // ONNX Runtime is VENDORED (app/ort/, extracted from the official 1.27.0 AAR) because
    // app/CMakeLists.txt links it directly via -DORT_ROOT and the C++ API. This is the
    // offline/local path and needs no network at build time.
    //
    // ONLINE ALTERNATIVE (AAR + Prefab): if you switch app/CMakeLists.txt to
    // `find_package(onnxruntime REQUIRED CONFIG)` + `onnxruntime::onnxruntime`, then delete
    // app/ort + the jniLibs srcDir above, add `buildFeatures { prefab = true }`, and enable:
    //   implementation("com.microsoft.onnxruntime:onnxruntime-android:1.27.0")
}

/*
 * Stage ONLY the assets the IME actually loads, out of ../../model, into the APK.
 * Avoids bundling all six .onnx variants (~38 MB) — we ship one 4.9 MB quantized model.
 * Source of truth: /home/luigi/sloth-zhuyin-linux/model/
 */
val modelDir = rootProject.file("../model")
val encDir   = rootProject.file("../model/slothe_4m_onnx")

val copyModelAssets by tasks.registering(Copy::class) {
    description = "Stage the .onnx + vocab + phonetic table into src/main/assets/slothing"
    into(layout.projectDirectory.dir("src/main/assets/slothing"))

    from(encDir) {
        include("model_quantized.onnx")   // 4.9 MB single-file quantized model (self-contained)
        include("syl_vocab.json")
        include("char2id.json")
    }
    from(modelDir) {
        include("phonetic_table.tsv")
        include("assoc_tc.tsv")   // 聯想 dictionary (built by model/build_assoc.py)
    }
    // on-device accuracy benchmark (230-case 免選字 reference set)
    from(rootProject.file("../eval")) {
        include("reference_mspy.tsv")
    }
    // NB: no doFirst {} existence check here — an action lambda in a .kts script
    // captures the script object, which the configuration cache cannot serialize.
    // If encDir is missing the task simply copies nothing (the .onnx is absent and
    // Core.init throws at runtime), which is the loud failure we want anyway.
}

// Make every APK build stage the assets first.
tasks.named("preBuild").configure { dependsOn(copyModelAssets) }
