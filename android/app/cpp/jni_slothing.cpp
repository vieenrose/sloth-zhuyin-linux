// JNI boundary for com.slothing.ime.Core. One SlothingSession per IME instance
// lives behind an opaque jlong handle (reinterpret_cast of the pointer); Kotlin
// stores it and threads it through every call. Nothing here holds JNI state
// across calls except cached jclass/jmethodID (resolved once in JNI_OnLoad).
//
// Threading: these functions run on whatever thread Kotlin calls them from.
// The cheap ones are called on the UI thread; the heavy ones (nativeDecodeLive,
// nativeBeginConvert, nativePickSegment/Phrase, nativeEnsurePhrases,
// nativeConfirmChoosing) are called from a coroutine on Dispatchers.Default.
// SlothingSession's own mutex + generation counter make that race-free — see
// session.h. JNIEnv is per-thread and never shared, so each call uses the env
// it was handed; only the cached global refs are shared (immutable).
#include "onnx_decoder.h"
#include "session.h"

#include <jni.h>
#include <string>
#include <vector>

using namespace slothing;

// ---- cached Kotlin classes/ctors (see Core.kt) -----------------------------
namespace {
JavaVM *g_vm = nullptr;
jclass g_preedit = nullptr;    // com.slothing.ime.Preedit
jmethodID g_preeditCtor = nullptr;
jclass g_candidates = nullptr; // com.slothing.ime.Candidates
jmethodID g_candidatesCtor = nullptr;
jclass g_phrase = nullptr;     // com.slothing.ime.Phrase
jmethodID g_phraseCtor = nullptr;
jclass g_string = nullptr;

SlothingSession *sess(jlong h) { return reinterpret_cast<SlothingSession *>(h); }

// ---- string codecs (correct for supplementary CJK, unlike GetStringUTFChars,
// which emits modified/CESU-8) --------------------------------------------
std::string jstrToUtf8(JNIEnv *env, jstring s) {
    if (!s) return {};
    const jsize n = env->GetStringLength(s);
    const jchar *u = env->GetStringChars(s, nullptr);
    std::string out;
    out.reserve(n * 3);
    for (jsize i = 0; i < n; i++) {
        unsigned int cp = u[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < n) { // high surrogate
            unsigned int lo = u[i + 1];
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                i++;
            }
        }
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    env->ReleaseStringChars(s, u);
    return out;
}

jstring utf8ToJstr(JNIEnv *env, const std::string &s) {
    std::vector<jchar> u;
    u.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        unsigned char c = s[i];
        unsigned int cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < s.size(); k++)
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        i += len;
        if (cp < 0x10000) {
            u.push_back(static_cast<jchar>(cp));
        } else {
            cp -= 0x10000;
            u.push_back(static_cast<jchar>(0xD800 + (cp >> 10)));
            u.push_back(static_cast<jchar>(0xDC00 + (cp & 0x3FF)));
        }
    }
    return env->NewString(u.data(), static_cast<jsize>(u.size()));
}

std::string byteArrayToString(JNIEnv *env, jbyteArray a) {
    if (!a) return {};
    const jsize n = env->GetArrayLength(a);
    std::string out(static_cast<size_t>(n), '\0');
    env->GetByteArrayRegion(a, 0, n, reinterpret_cast<jbyte *>(out.data()));
    return out;
}

jobjectArray toStringArray(JNIEnv *env, const std::vector<std::string> &v) {
    jobjectArray arr =
        env->NewObjectArray(static_cast<jsize>(v.size()), g_string, nullptr);
    for (jsize i = 0; i < static_cast<jsize>(v.size()); i++) {
        jstring s = utf8ToJstr(env, v[i]);
        env->SetObjectArrayElement(arr, i, s);
        env->DeleteLocalRef(s);
    }
    return arr;
}
} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *) {
    g_vm = vm;
    JNIEnv *env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK)
        return JNI_ERR;
    auto pin = [&](const char *name) -> jclass {
        jclass local = env->FindClass(name);
        if (!local) return nullptr;
        auto g = static_cast<jclass>(env->NewGlobalRef(local));
        env->DeleteLocalRef(local);
        return g;
    };
    g_string = pin("java/lang/String");
    g_preedit = pin("com/slothing/ime/Preedit");
    g_candidates = pin("com/slothing/ime/Candidates");
    g_phrase = pin("com/slothing/ime/Phrase");
    if (!g_string || !g_preedit || !g_candidates || !g_phrase) return JNI_ERR;
    g_preeditCtor =
        env->GetMethodID(g_preedit, "<init>", "(Ljava/lang/String;III)V");
    g_candidatesCtor = env->GetMethodID(g_candidates, "<init>",
                                        "(IIZ[Ljava/lang/String;)V");
    g_phraseCtor =
        env->GetMethodID(g_phrase, "<init>", "(ILjava/lang/String;)V");
    if (!g_preeditCtor || !g_candidatesCtor || !g_phraseCtor) return JNI_ERR;
    return JNI_VERSION_1_6;
}

// The extern "C" JNIEXPORT surface. Names follow the Java_<pkg>_<Class>_<method>
// mangling so Kotlin's `external fun` binds them with no RegisterNatives.
extern "C" {

#define JNI(ret, name) \
    JNIEXPORT ret JNICALL Java_com_slothing_ime_Core_##name

// ---- lifecycle -------------------------------------------------------------

JNI(jlong, nativeInit)(JNIEnv *env, jobject, jbyteArray model,
                       jbyteArray sylVocab, jbyteArray char2id,
                       jbyteArray table, jint threads, jstring learnPath,
                       jbyteArray assocTsv, jstring assocUserPath) {
    std::string modelBytes = byteArrayToString(env, model);
    std::string sylBytes = byteArrayToString(env, sylVocab);
    std::string charBytes = byteArrayToString(env, char2id);
    std::string tableBytes = byteArrayToString(env, table);
    auto dec = std::make_unique<OnnxDecoder>(
        modelBytes.data(), modelBytes.size(), sylBytes, charBytes, tableBytes,
        threads, jstrToUtf8(env, learnPath));
    auto *s = new SlothingSession(std::move(dec), tableBytes,
                                  byteArrayToString(env, assocTsv),
                                  jstrToUtf8(env, assocUserPath));
    return reinterpret_cast<jlong>(s);
}

JNI(jboolean, nativeReady)(JNIEnv *, jobject, jlong h) {
    return h && sess(h)->ready() ? JNI_TRUE : JNI_FALSE;
}

// Debug/benchmark: space-separated bopomofo syllables -> best decoded sentence.
JNI(jstring, nativeDecodeBest)(JNIEnv *env, jobject, jlong h, jstring sylsSp) {
    std::string sp = jstrToUtf8(env, sylsSp);
    std::vector<std::string> syls;
    size_t i = 0;
    while (i < sp.size()) {
        size_t j = sp.find(' ', i);
        std::string t = sp.substr(i, j == std::string::npos ? j : j - i);
        if (!t.empty()) syls.push_back(t);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    return utf8ToJstr(env, sess(h)->decodeBest(syls));
}

JNI(void, nativeDestroy)(JNIEnv *, jobject, jlong h) {
    delete sess(h);
}

JNI(void, nativeSetContext)(JNIEnv *env, jobject, jlong h, jstring ctx) {
    sess(h)->setContext(jstrToUtf8(env, ctx));
}

JNI(void, nativeSetEnglishMode)(JNIEnv *, jobject, jlong h, jboolean on) {
    sess(h)->setEnglishMode(on == JNI_TRUE);
}

JNI(void, nativeSetFullWidth)(JNIEnv *, jobject, jlong h, jboolean on) {
    sess(h)->setFullWidth(on == JNI_TRUE);
}

// ---- composing input (cheap) ----------------------------------------------

JNI(jint, nativeFeedKey)(JNIEnv *, jobject, jlong h, jint cp) {
    return static_cast<jint>(sess(h)->feedKey(static_cast<uint32_t>(cp)));
}
JNI(jint, nativeToneOrSpace)(JNIEnv *, jobject, jlong h, jint cp) {
    return static_cast<jint>(sess(h)->toneOrSpace(static_cast<uint32_t>(cp)));
}
JNI(jint, nativeBackspace)(JNIEnv *, jobject, jlong h) {
    return static_cast<jint>(sess(h)->backspace());
}
JNI(jint, nativeMoveCursor)(JNIEnv *, jobject, jlong h, jint dir) {
    return static_cast<jint>(sess(h)->moveCursor(dir));
}
JNI(void, nativeReset)(JNIEnv *, jobject, jlong h) { sess(h)->reset(); }
JNI(void, nativeFlush)(JNIEnv *, jobject, jlong h) { sess(h)->flush(); }

// ---- live conversion -------------------------------------------------------

JNI(jboolean, nativeRefreshLiveFast)(JNIEnv *, jobject, jlong h) {
    return sess(h)->refreshLiveFast() ? JNI_TRUE : JNI_FALSE;
}
// HEAVY — call on Dispatchers.Default.
JNI(jboolean, nativeDecodeLive)(JNIEnv *, jobject, jlong h) {
    return sess(h)->decodeLive() ? JNI_TRUE : JNI_FALSE;
}
JNI(jstring, nativeGetLive)(JNIEnv *env, jobject, jlong h) {
    return utf8ToJstr(env, sess(h)->getLive());
}
JNI(jobjectArray, nativeGetLiveSuggestions)(JNIEnv *env, jobject, jlong h) {
    return toStringArray(env, sess(h)->getLiveSuggestions());
}
JNI(jobjectArray, nativeGetLastWordCands)(JNIEnv *env, jobject, jlong h) {
    return toStringArray(env, sess(h)->getLastWordCands());
}
JNI(jstring, nativeGetLastWordCurrent)(JNIEnv *env, jobject, jlong h) {
    return utf8ToJstr(env, sess(h)->getLastWordCurrent());
}
JNI(jboolean, nativePickLastWord)(JNIEnv *env, jobject, jlong h, jstring ch) {
    return sess(h)->pickLastWord(jstrToUtf8(env, ch)) ? JNI_TRUE : JNI_FALSE;
}
JNI(jobjectArray, nativeGetPredictions)(JNIEnv *env, jobject, jlong h) {
    return toStringArray(env, sess(h)->getPredictions());
}
JNI(void, nativePredicted)(JNIEnv *env, jobject, jlong h, jstring s) {
    sess(h)->predicted(jstrToUtf8(env, s));
}
JNI(void, nativeClearPredictions)(JNIEnv *, jobject, jlong h) {
    sess(h)->clearPredictions();
}
JNI(void, nativeCommitSentence)(JNIEnv *env, jobject, jlong h, jstring s) {
    sess(h)->commitSentence(jstrToUtf8(env, s));
}
JNI(jint, nativeInsertSymbol)(JNIEnv *env, jobject, jlong h, jstring s) {
    return static_cast<jint>(sess(h)->insertSymbol(jstrToUtf8(env, s)));
}

// ---- convert / choose (HEAVY — Dispatchers.Default) -----------------------

JNI(jboolean, nativeBeginConvert)(JNIEnv *, jobject, jlong h, jint focus,
                                  jboolean commitDirect) {
    return sess(h)->beginConvert(focus, commitDirect == JNI_TRUE) ? JNI_TRUE
                                                                  : JNI_FALSE;
}
JNI(jboolean, nativeCommitLive)(JNIEnv *, jobject, jlong h) {
    return sess(h)->commitLive() ? JNI_TRUE : JNI_FALSE;
}
JNI(void, nativeEnsurePhrases)(JNIEnv *, jobject, jlong h) {
    sess(h)->ensurePhrases();
}
JNI(void, nativePickSegment)(JNIEnv *, jobject, jlong h, jint idx) {
    sess(h)->pickSegment(idx);
}
JNI(void, nativePickPhrase)(JNIEnv *env, jobject, jlong h, jint start,
                            jstring phrase) {
    sess(h)->pickPhrase(start, jstrToUtf8(env, phrase));
}
JNI(jboolean, nativeConfirmChoosing)(JNIEnv *, jobject, jlong h) {
    return sess(h)->confirmChoosing() ? JNI_TRUE : JNI_FALSE;
}

// ---- choosing navigation (cheap) ------------------------------------------

JNI(jboolean, nativeEscapeChoosing)(JNIEnv *, jobject, jlong h) {
    return sess(h)->escapeChoosing() ? JNI_TRUE : JNI_FALSE;
}
JNI(void, nativeMoveHighlight)(JNIEnv *, jobject, jlong h, jint dir) {
    sess(h)->moveHighlight(dir);
}
JNI(void, nativeMoveFocus)(JNIEnv *, jobject, jlong h, jint dir) {
    sess(h)->moveFocus(dir);
}
JNI(void, nativeReopen)(JNIEnv *, jobject, jlong h) { sess(h)->reopenCandList(); }
JNI(void, nativeCloseCandList)(JNIEnv *, jobject, jlong h) {
    sess(h)->closeCandList();
}

// ---- read-only views (cheap) ----------------------------------------------

JNI(jobject, nativeGetPreedit)(JNIEnv *env, jobject, jlong h) {
    PreeditView v = sess(h)->getPreedit();
    jstring text = utf8ToJstr(env, v.text);
    jobject o = env->NewObject(g_preedit, g_preeditCtor, text, v.cursor,
                               v.hlStart, v.hlEnd);
    env->DeleteLocalRef(text);
    return o;
}

JNI(jobject, nativeGetCandidates)(JNIEnv *env, jobject, jlong h) {
    CandidateView v = sess(h)->getCandidates();
    jobjectArray items = toStringArray(env, v.items);
    jobject o = env->NewObject(g_candidates, g_candidatesCtor, v.focus,
                               v.cursor, v.open ? JNI_TRUE : JNI_FALSE, items);
    env->DeleteLocalRef(items);
    return o;
}

JNI(jobjectArray, nativeGetPhrases)(JNIEnv *env, jobject, jlong h) {
    std::vector<PhraseView> ph = sess(h)->getPhrases();
    jobjectArray arr =
        env->NewObjectArray(static_cast<jsize>(ph.size()), g_phrase, nullptr);
    for (jsize i = 0; i < static_cast<jsize>(ph.size()); i++) {
        jstring t = utf8ToJstr(env, ph[i].text);
        jobject o = env->NewObject(g_phrase, g_phraseCtor, ph[i].start, t);
        env->SetObjectArrayElement(arr, i, o);
        env->DeleteLocalRef(t);
        env->DeleteLocalRef(o);
    }
    return arr;
}

JNI(jstring, nativeGetCommit)(JNIEnv *env, jobject, jlong h) {
    return utf8ToJstr(env, sess(h)->getCommit());
}
JNI(jstring, nativeGetNotice)(JNIEnv *env, jobject, jlong h) {
    return utf8ToJstr(env, sess(h)->getNotice());
}
JNI(jint, nativeState)(JNIEnv *, jobject, jlong h) { return sess(h)->state(); }
JNI(jboolean, nativeSymbolMode)(JNIEnv *, jobject, jlong h) {
    return sess(h)->symbolMode() ? JNI_TRUE : JNI_FALSE;
}

#undef JNI
} // extern "C"
