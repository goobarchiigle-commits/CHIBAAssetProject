"""
test_token_debug.py
fetch_token() の送信内容と応答を完全出力するデバッグスクリプト。

実行: python test_token_debug.py
     python test_token_debug.py --session   # Session有り（本番コードと同等）
     python test_token_debug.py --no-session # Session無し（手動テストと同等）
"""
import sys
import os
import json
import argparse

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "C:/ai-trading")

from dotenv import load_dotenv
load_dotenv(dotenv_path="C:/ai-trading/src/.env", override=False)

import requests

# ─── 設定 ───────────────────────────────────────────────────────────────────
API_PORT = int(os.getenv("KABU_API_PORT", "18080"))
BASE_URL  = f"http://localhost:{API_PORT}/kabusapi"
TOKEN_URL = f"{BASE_URL}/token"

pw = os.getenv("KABU_API_PASSWORD", "")

# ─── パスワード検証 ──────────────────────────────────────────────────────────
import hashlib
pw_bytes = pw.encode("utf-8")
pw_sha256_8 = hashlib.sha256(pw_bytes).hexdigest()[:8]
pw_repr = f"len={len(pw)} sha256_8={pw_sha256_8}"

# ─── 引数 ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--session",    dest="use_session", action="store_true",  default=True)
parser.add_argument("--no-session", dest="use_session", action="store_false")
args = parser.parse_args()

print("=" * 60)
print(f"[CONFIG] url        = {TOKEN_URL}")
print(f"[CONFIG] password   = {pw_repr}")
print(f"[CONFIG] use_session= {args.use_session}")
print("=" * 60)

payload = {"APIPassword": pw}

if args.use_session:
    # 本番コード (KabuClient.__init__) と同一
    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})
    print("[SESSION] Session headers before POST:")
    for k, v in sess.headers.items():
        print(f"  {k}: {v}")

    # PreparedRequest を作成して送信前ヘッダーを確認
    req = requests.Request("POST", TOKEN_URL, json=payload)
    prepped = sess.prepare_request(req)
    print("\n[PREPARED] Headers that will be sent:")
    for k, v in prepped.headers.items():
        print(f"  {k}: {v}")
    print(f"[PREPARED] Body: {prepped.body}")
    print(f"[PREPARED] Body length: {len(prepped.body) if prepped.body else 0}")

    print("\n[SEND] Sending via session.send()...")
    try:
        resp = sess.send(prepped, timeout=5)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)
else:
    # 手動テストと同一
    req = requests.Request("POST", TOKEN_URL, json=payload)
    prepped = req.prepare()
    print("[PREPARED] Headers that will be sent (no session):")
    for k, v in prepped.headers.items():
        print(f"  {k}: {v}")
    print(f"[PREPARED] Body: {prepped.body}")
    print(f"[PREPARED] Body length: {len(prepped.body) if prepped.body else 0}")

    print("\n[SEND] Sending via requests.Session().send()...")
    try:
        resp = requests.Session().send(prepped, timeout=5)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        sys.exit(1)

# ─── 応答出力 ────────────────────────────────────────────────────────────────
print("\n[RESPONSE]")
print(f"  status_code   = {resp.status_code}")
print(f"  reason        = {resp.reason}")
print(f"  headers       = {dict(resp.headers)}")
print(f"  body          = {resp.text}")
print(f"  elapsed_ms    = {resp.elapsed.total_seconds() * 1000:.1f}")

try:
    body = resp.json()
    print(f"  Code          = {body.get('Code')}")
    print(f"  Message       = {body.get('Message')}")
except Exception:
    print("  (body is not JSON)")

# ─── 追加: apisoftlimit 疎通確認 ─────────────────────────────────────────────
print("\n[EXTRA] GET /kabusapi/apisoftlimit (auth不要エンドポイント確認)")
try:
    r2 = requests.get(f"{BASE_URL}/apisoftlimit", timeout=3)
    print(f"  status_code = {r2.status_code}")
    print(f"  body        = {r2.text[:200]}")
except Exception as e:
    print(f"  error       = {e}")

print("\n[DONE]")
