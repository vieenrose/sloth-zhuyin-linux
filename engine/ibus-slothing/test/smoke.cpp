// Headless end-to-end smoke test for ibus-engine-slothing: connects to a
// (private) ibus-daemon as a client, focuses an input context on the
// "slothing" engine, types dachen keys, and asserts what gets committed.
// Run via run-smoke.sh (dbus-run-session + ibus-daemon + engine + this).
// Requires a running slothingd (real decode) — the same bar as the web
// demo's parity suite.
#include <cstdio>
#include <cstring>
#include <ibus.h>
#include <string>

static std::string committed;
static GMainLoop *loop = nullptr;

static void on_commit(IBusInputContext *, IBusText *text, gpointer) {
    committed += ibus_text_get_text(text);
    g_main_loop_quit(loop);
}

static gboolean timeoutFired = FALSE;
static gboolean on_timeout(gpointer) {
    timeoutFired = TRUE;
    g_main_loop_quit(loop);
    return G_SOURCE_REMOVE;
}

// Send one key press+release and let the engine process it.
static void key(IBusInputContext *ctx, guint keyval) {
    ibus_input_context_process_key_event(ctx, keyval, 0, 0);
    ibus_input_context_process_key_event(ctx, keyval, 0, IBUS_RELEASE_MASK);
    // drain pending dbus traffic
    while (g_main_context_pending(nullptr)) {
        g_main_context_iteration(nullptr, FALSE);
    }
}

static void spin_ms(int ms) {
    timeoutFired = FALSE;
    guint id = g_timeout_add(ms, on_timeout, nullptr);
    g_main_loop_run(loop);
    if (!timeoutFired) {
        g_source_remove(id); // loop quit by a commit before the timeout
    }
}

int main() {
    ibus_init();
    IBusBus *bus = ibus_bus_new();
    if (!ibus_bus_is_connected(bus)) {
        fprintf(stderr, "SKIP: no ibus daemon\n");
        return 2;
    }
    loop = g_main_loop_new(nullptr, FALSE);

    IBusInputContext *ctx = ibus_bus_create_input_context(bus, "smoke");
    if (!ctx) {
        fprintf(stderr, "FAIL: cannot create input context\n");
        return 1;
    }
    ibus_input_context_set_capabilities(
        ctx, IBUS_CAP_PREEDIT_TEXT | IBUS_CAP_AUXILIARY_TEXT |
                 IBUS_CAP_LOOKUP_TABLE | IBUS_CAP_FOCUS);
    g_signal_connect(ctx, "commit-text", G_CALLBACK(on_commit), nullptr);
    ibus_input_context_focus_in(ctx);
    ibus_input_context_set_engine(ctx, "slothing");
    spin_ms(500); // let the engine attach

    int failures = 0;
    auto check = [&](bool ok, const char *desc) {
        if (!ok) failures++;
        printf("%s  %s\n", ok ? "PASS" : "FAIL", desc);
    };

    // ---- zh: ji3 y94 (ㄨㄛˇ ㄗㄞˋ) + Enter -> 我在 --------------------------
    committed.clear();
    for (char c : std::string("ji3y94")) {
        key(ctx, (guint)c);
    }
    spin_ms(600); // live decode
    key(ctx, IBUS_KEY_Return);
    // wait for the commit (decode may still be in flight)
    for (int i = 0; i < 20 && committed.empty(); i++) {
        spin_ms(250);
    }
    check(committed == "我在", "ji3y94 + Enter commits 我在");

    // ---- auto zh/en: python + 5k4 ek7 -> python 這個 ------------------------
    committed.clear();
    for (char c : std::string("python5k4ek7")) {
        key(ctx, (guint)c);
    }
    spin_ms(600);
    key(ctx, IBUS_KEY_Return);
    for (int i = 0; i < 20 && committed.empty(); i++) {
        spin_ms(250);
    }
    check(committed == "python 這個",
          "python5k4ek7 + Enter commits 'python 這個'");

    // ---- Esc clears only pending bopomofo, never converted text -----------
    committed.clear();
    for (char c : std::string("ji3")) {
        key(ctx, (guint)c);
    }
    for (char c : std::string("5k")) { // pending, no tone yet
        key(ctx, (guint)c);
    }
    key(ctx, IBUS_KEY_Escape); // clears ㄓㄜ only
    spin_ms(600);
    key(ctx, IBUS_KEY_Return);
    for (int i = 0; i < 20 && committed.empty(); i++) {
        spin_ms(250);
    }
    check(committed == "我", "Esc clears pending bopomofo only; 我 survives");

    // ---- punctuation: Shift+, -> ， direct commit --------------------------
    committed.clear();
    key(ctx, (guint)'<');
    for (int i = 0; i < 8 && committed.empty(); i++) {
        spin_ms(250);
    }
    check(committed == "，", "Shift+, commits ， directly");

    // ---- English words with a manual space keep the space on commit -------
    // "web app" (last word has no trailing tone/space, so it commits via the
    // decode path) must stay "web app", not collapse to "webapp".
    committed.clear();
    for (char c : std::string("web app")) {
        key(ctx, (guint)c);
    }
    spin_ms(400);
    key(ctx, IBUS_KEY_Return);
    for (int i = 0; i < 20 && committed.empty(); i++) {
        spin_ms(250);
    }
    check(committed == "web app",
          "manual space between English words survives commit");

    // ---- 聯想: after committing 電, ⇧1 commits the top prediction (腦) ----
    committed.clear();
    for (char c : std::string("2u04")) { // ㄉㄧㄢˋ
        key(ctx, (guint)c);
    }
    spin_ms(600);
    key(ctx, IBUS_KEY_Return);
    for (int i = 0; i < 20 && committed.empty(); i++) {
        spin_ms(250);
    }
    check(committed == "電", "2u04 + Enter commits 電");
    committed.clear();
    // ⇧1 = '!' with the shift modifier
    ibus_input_context_process_key_event(ctx, (guint)'!', 0, IBUS_SHIFT_MASK);
    ibus_input_context_process_key_event(ctx, (guint)'!', 0,
                                         IBUS_SHIFT_MASK | IBUS_RELEASE_MASK);
    for (int i = 0; i < 8 && committed.empty(); i++) {
        spin_ms(250);
    }
    check(committed == "腦",
          "聯想: ⇧1 after committing 電 commits the top prediction 腦");

    printf("\n%s\n", failures ? "FAILURES" : "all passed");
    return failures ? 1 : 0;
}
