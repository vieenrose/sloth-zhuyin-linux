/*
 * harvest: replicate the Slothing engine's candidate harvest against
 * libchewing, for the evaluation harness.
 *
 * Usage: ./harvest "su3cl3"   (default-layout key sequence; ' ' = tone 1)
 *
 * Types the keys into a fresh ChewingContext, then walks the interval
 * segmentation exactly like eim.cpp's collectCandidatePositions(): per
 * interval, cycle every candidate list (cand_list_first/next), keep
 * candidates whose UTF-8 char length equals the interval span, and fall back
 * to the user's own slice for a dry interval.
 *
 * Output (stdout, one JSON object):
 *   {"buffer":"你好","positions":[["你","妳",...],["好","號",...]]}
 */
#include <chewing.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* UTF-8 char count */
static size_t u8len(const char *s) {
    size_t n = 0;
    for (; *s; s++) {
        if ((*s & 0xC0) != 0x80) {
            n++;
        }
    }
    return n;
}

/* byte offset of char index `idx` */
static size_t u8off(const char *s, size_t idx) {
    size_t n = 0, i = 0;
    for (; s[i]; i++) {
        if ((s[i] & 0xC0) != 0x80) {
            if (n == idx) {
                return i;
            }
            n++;
        }
    }
    return i;
}

static void json_escape(const char *s, FILE *out) {
    for (; *s; s++) {
        unsigned char c = (unsigned char) *s;
        if (c == '"' || c == '\\') {
            fputc('\\', out);
            fputc(c, out);
        } else if (c < 0x20) {
            fprintf(out, "\\u%04x", c);
        } else {
            fputc(c, out);
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <key-sequence>\n", argv[0]);
        return 2;
    }
    ChewingContext *ctx = chewing_new();
    if (!ctx) {
        fprintf(stderr, "chewing_new failed\n");
        return 1;
    }
    chewing_set_maxChiSymbolLen(ctx, 18);
    chewing_set_spaceAsSelection(ctx, 0);
    chewing_set_phraseChoiceRearward(ctx, 0);

    for (const char *k = argv[1]; *k; k++) {
        if (*k == ' ') {
            chewing_handle_Space(ctx);
        } else {
            chewing_handle_Default(ctx, *k);
        }
    }

    char *buf = chewing_buffer_String(ctx);
    if (!buf || !buf[0]) {
        fprintf(stderr, "empty buffer for keys '%s'\n", argv[1]);
        chewing_free(buf);
        chewing_delete(ctx);
        return 1;
    }

    /* collect intervals */
    int from[64], to[64], nIv = 0;
    chewing_handle_Home(ctx);
    chewing_interval_Enumerate(ctx);
    while (chewing_interval_hasNext(ctx) && nIv < 64) {
        IntervalType iv;
        chewing_interval_Get(ctx, &iv);
        from[nIv] = iv.from;
        to[nIv] = iv.to;
        nIv++;
    }

    printf("{\"buffer\":\"");
    json_escape(buf, stdout);
    printf("\",\"positions\":[");

    for (int i = 0; i < nIv; i++) {
        if (i) {
            printf(",");
        }
        /* advance cursor with progress guard */
        int guard = 0;
        while (chewing_cursor_Current(ctx) < from[i]) {
            int before = chewing_cursor_Current(ctx);
            chewing_handle_Right(ctx);
            if (chewing_cursor_Current(ctx) == before || ++guard > 20) {
                break;
            }
        }
        size_t span = (size_t) (to[i] - from[i]);
        printf("[");
        int emitted = 0;
        if (chewing_cand_open(ctx) == 0) {
            chewing_cand_list_first(ctx);
            int lg = 0;
            while (1) {
                chewing_cand_Enumerate(ctx);
                while (chewing_cand_hasNext(ctx)) {
                    char *cand = chewing_cand_String(ctx);
                    if (cand && u8len(cand) == span) {
                        if (emitted++) {
                            printf(",");
                        }
                        printf("\"");
                        json_escape(cand, stdout);
                        printf("\"");
                    }
                    chewing_free(cand);
                }
                if (emitted || !chewing_cand_list_has_next(ctx) || ++lg > 20) {
                    break;
                }
                chewing_cand_list_next(ctx);
            }
            chewing_cand_close(ctx);
        }
        if (!emitted) {
            /* pin dry interval to the user's own slice */
            size_t b0 = u8off(buf, (size_t) from[i]);
            size_t b1 = u8off(buf, (size_t) to[i]);
            char slice[256];
            size_t len = b1 - b0 < sizeof(slice) - 1 ? b1 - b0 : sizeof(slice) - 1;
            memcpy(slice, buf + b0, len);
            slice[len] = 0;
            printf("\"");
            json_escape(slice, stdout);
            printf("\"");
        }
        printf("]");
    }
    printf("]}\n");

    chewing_free(buf);
    chewing_delete(ctx);
    return 0;
}
