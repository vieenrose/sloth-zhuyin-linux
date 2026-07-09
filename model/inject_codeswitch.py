#!/usr/bin/env python3
"""Inject English into REAL Chinese sentences -> contextual zh/en code-switch.

Solution (B): instead of a few synthetic templates (~3% of training), replace
common loan-nouns in real corpus sentences with the English word Taiwanese
actually type ("我在寫程式" -> "我在寫 code"). The Chinese context is genuine;
only the term flips to English. Produces code-switch at scale (~15-20% of the
corpus) with realistic context -- the data K/E needed to learn segmentation.

  python3 model/inject_codeswitch.py --corpus model/corpus_huge.txt \
      --out model/corpus_cs_big.txt --frac 0.18
"""
import argparse

# zh -> en loan-nouns Taiwanese commonly code-switch (extends synth_codeswitch).
REPLACE = {
    "會議":"meeting","開會":"meeting","程式碼":"code","程式":"code","程序":"code",
    "電子郵件":"email","郵件":"email","信件":"email","專案":"project","計畫":"project",
    "檔案":"file","文件":"file","訊息":"message","簡訊":"message","團隊":"team",
    "小組":"team","報告":"report","簡報":"slides","投影片":"slides","資料":"data",
    "數據":"data","帳號":"account","帳戶":"account","密碼":"password","應用程式":"app",
    "程式集":"app","網站":"website","網頁":"website","伺服器":"server","主機":"server",
    "資料庫":"database","使用者":"user","用戶":"user","系統":"system","功能":"feature",
    "版本":"version","測試":"test","部署":"deploy","上線":"deploy","產品":"product",
    "客戶":"client","顧客":"customer","訂單":"order","預算":"budget","行事曆":"calendar",
    "行程":"schedule","筆記":"note","影片":"video","照片":"photo","相片":"photo",
    "連結":"link","網址":"link","通知":"notification","分享":"share","留言":"comment",
    "評論":"comment","貼文":"post","文章":"post","頻道":"channel","直播":"live",
    "會員":"member","介面":"interface","流程":"workflow","期限":"deadline","截止":"deadline",
    "團購":"group buy","折扣":"discount","優惠券":"coupon","更新":"update","下載":"download",
    "上傳":"upload","螢幕":"screen","畫面":"screen","鍵盤":"keyboard","滑鼠":"mouse",
    "問題":"issue","錯誤":"bug","漏洞":"bug","需求":"requirement","規格":"spec",
    "介紹":"demo","示範":"demo","面試":"interview","履歷":"resume","offer":"offer",
    "預約":"booking","訂位":"booking","付款":"payment","發票":"invoice","收據":"receipt",
    "行銷":"marketing","業績":"KPI","目標":"goal","策略":"strategy","流量":"traffic",
    "點擊":"click","按鈕":"button","選單":"menu","標籤":"tag","關鍵字":"keyword",
    "搜尋":"search","篩選":"filter","排序":"sort","備份":"backup","還原":"restore",
    "設定":"setting","權限":"permission","登入":"login","登出":"logout","註冊":"register",
    "群組":"group","社團":"group","朋友":"friend","粉絲":"follower","追蹤":"follow",
    "編輯":"edit","刪除":"delete","儲存":"save","複製":"copy","貼上":"paste",
    "咖啡":"coffee","蛋糕":"cake","披薩":"pizza","起司":"cheese","巧克力":"chocolate",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-sentence", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=2,
                    help="write each code-switch line this many times (oversample: "
                         "real TW code-switch is concentrated in tech/social text)")
    args = ap.parse_args()

    # longest keys first so 程式碼 matches before 程式
    keys = sorted(REPLACE, key=len, reverse=True)
    n = cs = 0
    with open(args.corpus, encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            s = line.rstrip("\n")
            if not s:
                continue
            n += 1
            mixed, done = s, 0
            for k in keys:
                if k in mixed and done < args.max_per_sentence:
                    mixed = mixed.replace(k, " " + REPLACE[k] + " ", 1)
                    done += 1
            if done:                                    # a replaceable line -> code-switch
                cs += 1
                out.write((" ".join(mixed.split()) + "\n") * args.repeat)
            else:
                out.write(s + "\n")
    print(f"{n} lines -> {cs} unique code-switch (x{args.repeat}); "
          f"~{100*cs*args.repeat/(n + cs*(args.repeat-1)):.1f}% of output; "
          f"dict {len(REPLACE)} terms, wrote {args.out}")


if __name__ == "__main__":
    main()
