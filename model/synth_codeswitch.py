#!/usr/bin/env python3
"""Synthesize zh/en code-switch sentences for SlothLM training.

The Traditional-Chinese corpus contains no English, so the model never learns
to leave English tokens alone and decode the Chinese around them. This builds
natural Taiwanese-style code-switch lines two ways:

  1. REPLACE: take a real corpus sentence containing a common loan-noun and
     swap that noun for its English form (會議 -> meeting).
  2. TEMPLATE: fill sentence templates with always-English tech terms
     (Python, Excel, LINE ...).

Output is one sentence per line, appended to the training corpus; prepare_data
.py's mixed_input() then renders each as a zhuyin->text code-switch example
(Han chars -> bopomofo, English kept verbatim).

Usage:
  python3 model/synth_codeswitch.py --corpus corpus_big.txt \
      --out codeswitch.txt [--replace-limit N] [--templates-per-term K]
"""
import argparse
import hashlib
import re

# Common loan-nouns Taiwanese speakers routinely say in English. Chinese term
# (as it appears in text) -> English form. Kept to terms whose English is the
# natural spoken choice, so a replacement reads authentically.
REPLACE = {
    "會議": "meeting", "程式碼": "code", "程式": "code", "電子郵件": "email",
    "郵件": "email", "專案": "project", "檔案": "file", "訊息": "message",
    "團隊": "team", "報告": "report", "簡報": "slides", "資料": "data",
    "帳號": "account", "密碼": "password", "應用程式": "app", "網站": "website",
    "伺服器": "server", "資料庫": "database", "使用者": "user", "系統": "system",
    "功能": "feature", "版本": "version", "測試": "test", "部署": "deploy",
    "產品": "product", "客戶": "client", "訂單": "order", "預算": "budget",
    "行事曆": "calendar", "筆記": "note", "影片": "video", "照片": "photo",
    "連結": "link", "通知": "notification", "分享": "share", "留言": "comment",
    "貼文": "post", "頻道": "channel", "直播": "live", "簡訊": "SMS",
    "會員": "member", "介面": "interface", "流程": "workflow", "期限": "deadline",
    "團購": "group buy", "折扣": "discount", "優惠": "coupon", "更新": "update",
    "下載": "download", "上傳": "upload", "螢幕": "screen", "鍵盤": "keyboard",
}

# Terms almost always written in English; used to fill templates.
EN_TERMS = [
    "Python", "JavaScript", "React", "Java", "Excel", "Word", "PowerPoint",
    "Google", "Facebook", "Instagram", "YouTube", "LINE", "Zoom", "GitHub",
    "iPhone", "Android", "Windows", "Mac", "Wi-Fi", "PDF", "AI", "API",
    "ChatGPT", "Notion", "Slack", "Figma", "Docker", "SQL", "Linux", "VS Code",
]

TEMPLATES = [
    "我用 {} 寫程式", "今天要學 {}", "這個功能用 {} 做的", "我在研究 {}",
    "團隊都在用 {}", "我把檔案傳到 {}", "用 {} 開會", "我在 {} 上留言",
    "這個專案需要 {}", "老師教我們 {}", "我下載了 {}", "公司規定要用 {}",
    "我用 {} 做簡報", "他傳訊息到 {}", "我在 {} 追蹤你", "這題要用 {} 解",
]


def bucket(s, salt):
    return hashlib.sha1((salt + s).encode()).digest()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replace-limit", type=int, default=300000)
    ap.add_argument("--templates-per-term", type=int, default=400)
    args = ap.parse_args()

    # longest terms first so 電子郵件 wins over 郵件
    terms = sorted(REPLACE, key=len, reverse=True)
    out = open(args.out, "w", encoding="utf-8")
    n_repl = 0

    for line in open(args.corpus, encoding="utf-8"):
        s = line.strip()
        if not s or not (2 <= len(s) <= 26):
            continue
        if n_repl >= args.replace_limit:
            break
        for t in terms:
            if t in s:
                # replace only the first occurrence, ~50% of eligible lines
                if bucket(s, t) < 128:
                    mixed = s.replace(t, " " + REPLACE[t] + " ", 1)
                    mixed = re.sub(r"\s+", " ", mixed).strip()
                    out.write(mixed + "\n")
                    n_repl += 1
                break  # one replacement per sentence

    n_tmpl = 0
    for term in EN_TERMS:
        for i in range(args.templates_per_term):
            out.write(TEMPLATES[i % len(TEMPLATES)].format(term) + "\n")
            n_tmpl += 1

    out.close()
    print(f"wrote {n_repl} replacement + {n_tmpl} template code-switch lines "
          f"to {args.out}")


if __name__ == "__main__":
    main()
