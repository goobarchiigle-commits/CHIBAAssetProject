"""
tools/pdf_ocr.py
スキャンPDF → テキスト変換ツール（ocr.space 無料API使用）

【使い方】
  python tools/pdf_ocr.py <PDFパス> <APIキー>

【例】
  python tools/pdf_ocr.py "C:/Users/owner/Downloads/book.pdf/50万円を50億円_compressed.pdf" K12345678

【出力】
  PDFと同じフォルダに .txt ファイルを保存
  例: 50万円を50億円_compressed.txt

【無料APIキー取得】
  https://ocr.space/ocrapi → FREE API KEY
  月25,000リクエスト無料（319ページなら余裕）
"""

import sys
import os
import time
import json
import requests
import fitz  # PyMuPDF

sys.stdout.reconfigure(encoding="utf-8")


OCR_API_URL = "https://api.ocr.space/parse/image"
LANGUAGE    = "jpn"     # 日本語
DPI         = 150       # 解像度（高いほど精度UP、低いほど速い）
JPEG_QUALITY = 85       # JPEG圧縮品質


def ocr_page(image_bytes: bytes, api_key: str) -> str:
    """1ページをocr.space APIに送信してテキストを返す。失敗時は1回リトライ。"""
    for attempt in range(2):
        try:
            response = requests.post(
                OCR_API_URL,
                files={"file": ("page.jpg", image_bytes, "image/jpeg")},
                data={
                    "apikey":   api_key,
                    "language": LANGUAGE,
                    "isOverlayRequired": False,
                    "detectOrientation": True,
                    "scale":    True,
                    "OCREngine": 2,
                },
                timeout=60,
            )

            # レスポンスが JSON でない場合（サーバーエラー等）
            try:
                result = response.json()
            except Exception:
                if attempt == 0:
                    time.sleep(5)
                    continue
                return f"[JSONパースエラー: HTTP {response.status_code}]\n"

            # result が dict でない場合（文字列エラーメッセージ等）
            if not isinstance(result, dict):
                if attempt == 0:
                    time.sleep(5)
                    continue
                return f"[APIレスポンス異常: {str(result)[:80]}]\n"

            # エラーフラグチェック
            if result.get("IsErroredOnProcessing"):
                msg = result.get("ErrorMessage", ["不明なエラー"])
                if attempt == 0:
                    time.sleep(5)
                    continue
                return f"[OCRエラー: {msg}]\n"

            # ParsedResults が存在しない
            parsed = result.get("ParsedResults", [])
            if not parsed:
                return "[テキストなし]\n"

            # parsed[0] が dict でない場合
            first = parsed[0]
            if not isinstance(first, dict):
                return f"[ParsedResults形式異常: {str(first)[:80]}]\n"

            return first.get("ParsedText", "") + "\n"

        except requests.RequestException as e:
            if attempt == 0:
                time.sleep(5)
                continue
            return f"[通信エラー: {e}]\n"

    return "[リトライ失敗]\n"


def main():
    if len(sys.argv) < 3:
        print("使い方: python tools/pdf_ocr.py <PDFパス> <APIキー> [--pages 1,2,3]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    api_key  = sys.argv[2]

    # --pages オプション（特定ページのみ再処理）
    target_pages = None
    if "--pages" in sys.argv:
        idx = sys.argv.index("--pages")
        target_pages = [int(p) for p in sys.argv[idx + 1].split(",")]
        print(f"対象ページ: {target_pages}")

    if not os.path.exists(pdf_path):
        print(f"エラー: ファイルが見つかりません: {pdf_path}")
        sys.exit(1)

    # 出力ファイルパス
    base = os.path.splitext(pdf_path)[0]
    out_path = base + ".txt"

    print(f"入力: {pdf_path}")
    print(f"出力: {out_path}")

    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"総ページ数: {total}")

    # 既存ファイル読み込み
    import re
    if os.path.exists(out_path):
        existing = open(out_path, encoding="utf-8").read()
    else:
        existing = ""

    # --pages モード: 特定ページだけ再処理して上書き
    if target_pages:
        print(f"\n{len(target_pages)}ページを再処理します...\n")
        text = existing
        errors = 0
        for page_num in target_pages:
            i = page_num - 1
            page = doc[i]
            pix  = page.get_pixmap(dpi=DPI)
            jpeg = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)
            if len(jpeg) > 900_000:
                pix  = page.get_pixmap(dpi=100)
                jpeg = pix.tobytes("jpeg", jpg_quality=75)

            new_text = ocr_page(jpeg, api_key)
            status = "❌" if any(e in new_text for e in ["エラー", "異常", "失敗"]) else "✓"
            if status == "❌":
                errors += 1
            print(f"  {status} Page {page_num:3d}  ({len(jpeg)//1024} KB)  {new_text[:40].strip()!r}")

            # 既存テキストの該当ページを置換
            pattern = rf"(--- Page {page_num} ---\n).*?(?=\n\n--- Page |\Z)"
            replacement = rf"\g<1>{new_text.strip()}"
            text = re.sub(pattern, replacement, text, flags=re.DOTALL)
            time.sleep(1.5)

        doc.close()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n完了: {len(target_pages)}ページ再処理、エラー{errors}件")
        print(f"保存: {out_path}")
        return

    # 通常モード: 続きから処理
    all_text = []
    start_page = 0
    if existing:
        pages_done = re.findall(r"--- Page (\d+) ---", existing)
        if pages_done:
            start_page = int(pages_done[-1])
            all_text   = [existing]
            print(f"再開: {start_page}ページまで完了済み → {start_page+1}ページから継続")
        else:
            print(f"開始します（目安: {total * 3 // 60}〜{total * 5 // 60}分）")
    else:
        print(f"開始します（目安: {total * 3 // 60}〜{total * 5 // 60}分）")
    print()

    errors = 0

    for i in range(start_page, total):
        page = doc[i]

        # ページ → JPEG バイト列
        pix  = page.get_pixmap(dpi=DPI)
        jpeg = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)

        # 1MB超の場合はDPIを下げて再変換
        if len(jpeg) > 900_000:
            pix  = page.get_pixmap(dpi=100)
            jpeg = pix.tobytes("jpeg", jpg_quality=75)

        # OCR
        text = ocr_page(jpeg, api_key)
        all_text.append(f"\n\n--- Page {i+1} ---\n{text}")

        if "[OCRエラー" in text:
            errors += 1
            status = "❌"
        else:
            status = "✓"

        print(f"  {status} Page {i+1:3d}/{total}  ({len(jpeg)//1024} KB)  {text[:40].strip()!r}")

        # API レート制限対策（1秒待機）
        time.sleep(1.0)

        # 10ページごとに中間保存
        if (i + 1) % 10 == 0:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("".join(all_text))
            print(f"  → 中間保存: {i+1}/{total} ページ完了\n")

    doc.close()

    # 最終保存
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(all_text))

    print(f"\n完了！")
    print(f"  総ページ: {total}")
    print(f"  エラー  : {errors}")
    print(f"  保存先  : {out_path}")


if __name__ == "__main__":
    main()
