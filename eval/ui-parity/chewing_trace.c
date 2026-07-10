/*
 * chewing_trace: feed a key sequence to REAL libchewing and dump the
 * observable UI state after every keystroke, as JSON lines. The reference
 * trace for the differential UI-parity suite (compare_traces.py).
 *
 * We record STRUCTURE, not characters (Slothing's model is supposed to pick
 * different/better chars):
 *   {"key":"j","zh":2,"bopo":1,"cand":0,"cursor":2,"commit":0}
 *     zh     = converted (Chinese) chars in the preedit
 *     bopo   = 1 if an in-progress bopomofo syllable is visible
 *     cand   = candidate window open (total choices > 0)
 *     cursor = preedit cursor position (chars)
 *     commit = chars committed BY THIS KEY (0 if none)
 *
 * Key encoding in the input string: printable ASCII = itself, plus
 *   <D>=Down <U>=Up <L>=Left <R>=Right <E>=Enter <ESC>=Esc <B>=Backspace
 *   <T>=Tab <H>=Home <N>=End  (space = space)
 *
 * Build:
 *   gcc eval/ui-parity/chewing_trace.c -o /tmp/chewing_trace \
 *       -I ~/.local/include -L ~/.local/lib -lchewing -Wl,-rpath,$HOME/.local/lib
 * Run:
 *   CHEWING_PATH=~/.local/share/libchewing /tmp/chewing_trace "su3cl3<D>2<E>"
 */
#include <chewing.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t u8len(const char *s) {
    size_t n = 0;
    for (; *s; s++)
        if ((*s & 0xC0) != 0x80)
            n++;
    return n;
}

static void emit(ChewingContext *ctx, const char *key) {
    char *buf = chewing_buffer_String(ctx);
    const char *bopo = chewing_bopomofo_String_static(ctx);
    int cand = chewing_cand_TotalChoice(ctx) > 0;
    int cursor = chewing_cursor_Current(ctx);
    size_t commit = 0;
    if (chewing_commit_Check(ctx)) {
        char *c = chewing_commit_String(ctx);
        commit = u8len(c);
        chewing_free(c);
    }
    printf("{\"key\":\"%s\",\"zh\":%zu,\"bopo\":%d,\"cand\":%d,"
           "\"cursor\":%d,\"commit\":%zu}\n",
           key, u8len(buf), bopo && bopo[0] ? 1 : 0, cand, cursor, commit);
    chewing_free(buf);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <keys>\n", argv[0]);
        return 2;
    }
    ChewingContext *ctx = chewing_new();
    chewing_set_maxChiSymbolLen(ctx, 30);
    chewing_set_candPerPage(ctx, 9);
    chewing_set_spaceAsSelection(ctx, 0);

    const char *p = argv[1];
    char label[8];
    while (*p) {
        if (*p == '<') {
            const char *e = strchr(p, '>');
            if (!e) break;
            size_t n = (size_t)(e - p - 1);
            snprintf(label, sizeof label, "%.*s", (int)n, p + 1);
            if (!strcmp(label, "D")) chewing_handle_Down(ctx);
            else if (!strcmp(label, "U")) chewing_handle_Up(ctx);
            else if (!strcmp(label, "L")) chewing_handle_Left(ctx);
            else if (!strcmp(label, "R")) chewing_handle_Right(ctx);
            else if (!strcmp(label, "E")) chewing_handle_Enter(ctx);
            else if (!strcmp(label, "ESC")) chewing_handle_Esc(ctx);
            else if (!strcmp(label, "B")) chewing_handle_Backspace(ctx);
            else if (!strcmp(label, "T")) chewing_handle_Tab(ctx);
            else if (!strcmp(label, "H")) chewing_handle_Home(ctx);
            else if (!strcmp(label, "N")) chewing_handle_End(ctx);
            p = e + 1;
            emit(ctx, label);
            continue;
        }
        char k = *p++;
        if (k == ' ')
            chewing_handle_Space(ctx);
        else
            chewing_handle_Default(ctx, k);
        snprintf(label, sizeof label, "%c", k);
        emit(ctx, label);
    }
    chewing_delete(ctx);
    return 0;
}
