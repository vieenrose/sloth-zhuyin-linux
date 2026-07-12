import org.gradle.api.tasks.Copy
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Release signing — credentials live OUTSIDE the repo in ~/.slothing/
// keystore.properties (storeFile/storePassword/keyAlias/keyPassword).
// Absent file => release builds stay unsigned (CI / other machines).
val signingProps = Properties().apply {
    val f = File(System.getProperty("user.home"), ".slothing/keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
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
                // The vendored ggml static libs (app/ggml/) were cross-compiled
                // with the shared STL; match it so libc++_shared.so is bundled.
                arguments += "-DANDROID_STL=c++_shared"
            }
        }
    }

    // Pin the NDK + CMake that are actually installed under ~/Android/Sdk.
    ndkVersion = "27.2.12479018"

    externalNativeBuild {
        cmake {
            // Canonical native build owned by the decode-port/native agent. Target "slothing"
            // -> libslothing.so, matching Core.kt's System.loadLibrary("slothing"). It compiles
            // app/cpp/{jni_slothing,onnx_decoder,slothe}.cpp + engine/common headers, and links
            // the vendored ggml static libs (app/ggml/lib/arm64-v8a/) — the ternary GGUF encoder.
            path = file("CMakeLists.txt")
            version = "3.22.1"   // the ONLY cmake installed; without this AGP tries to fetch another
        }
    }

    // Curated asset staging happens via the copyModelAssets task below,
    // but keep the default assets dir too (holds the staged files + any static assets).
    sourceSets["main"].assets.srcDirs("src/main/assets")

    // ggml is linked STATICALLY into libslothing.so (app/ggml/lib/arm64-v8a/*.a),
    // so there is no external inference .so to bundle here anymore. libc++_shared.so
    // is added automatically by the NDK from the c++_shared STL.

    signingConfigs {
        if (signingProps.isNotEmpty()) {
            create("release") {
                storeFile = file(signingProps.getProperty("storeFile"))
                storePassword = signingProps.getProperty("storePassword")
                keyAlias = signingProps.getProperty("keyAlias")
                keyPassword = signingProps.getProperty("keyPassword")
            }
        }
    }

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
            if (signingProps.isNotEmpty()) {
                signingConfig = signingConfigs.getByName("release")
            }
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

    // Do NOT compress the model; slothe_load reads the GGUF from a real file
    // (copied to cache) and the .tsv/.json assets stay uncompressed too.
    androidResources {
        noCompress += listOf("gguf", "onnx", "tsv", "json")
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

    // No ONNX Runtime dependency: inference is now libslothe/ggml, cross-compiled
    // for arm64-v8a and vendored as static libs under app/ggml/ (linked directly
    // by app/CMakeLists.txt into libslothing.so). Fully offline, no AAR/Prefab.
}

/*
 * Stage ONLY the assets the IME actually loads, out of ../../model, into the APK.
 * Avoids bundling all six .onnx variants (~38 MB) — we ship one 4.9 MB quantized model.
 * Source of truth: /home/luigi/sloth-zhuyin-linux/model/
 */
val modelDir = rootProject.file("../model")
val encDir   = rootProject.file("../model/slothe_10m_onnx")

val copyModelAssets by tasks.registering(Copy::class) {
    description = "Stage vocab + phonetic table + 聯想 dict into src/main/assets/slothing"
    into(layout.projectDirectory.dir("src/main/assets/slothing"))

    // NB: the model itself is now the ternary GGUF (slothe-t-25m.gguf), committed
    // directly under src/main/assets/slothing/ (18 MB, built by the slothe port).
    // The old model_quantized.onnx is no longer staged or loaded.
    from(encDir) {
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
