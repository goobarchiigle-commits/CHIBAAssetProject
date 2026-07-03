"""
src/scripts/diagnose_api_auth.py
kabu Station API 認証診断スクリプト

401 Unauthorized のトラブルシューティング用。
コードを変更せずに実行できる診断専用スクリプト。

実行方法:
    cd C:\ai-trading
    python src/scripts/diagnose_api_auth.py

出力:
    A. .env 読み込み経路と優先順位
    B. パスワード詳細（値は非表示、長さ/repr/文字コードのみ）
    C. kabu Station 到達確認
    D. /token POST 実際の応答
    E. 診断サマリー
"""
from __future__ import annotations

import os
import sys
import socket
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ─────────────────────────────────────────────────────────────────────────────
# 診断ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _mask(val: str) -> str:
    """パスワードを非表示化。長さ・先頭1文字・末尾1文字のみ表示。"""
    if not val:
        return "(empty)"
    n = len(val)
    return f"{val[0]}{'*' * (n - 2)}{val[-1]}  [len={n}]"


def _pw_detail(val: str, label: str) -> None:
    if not val:
        print(f"  {label}: EMPTY")
        return
    print(f"  {label}:")
    print(f"    masked  : {_mask(val)}")
    print(f"    len     : {len(val)}")
    print(f"    first   : ord={ord(val[0])} char='{val[0]}'")
    print(f"    last    : ord={ord(val[-1])} char='{val[-1]}'")
    print(f"    repr[:4]: {repr(val[:4])}")
    print(f"    repr[-4:]: {repr(val[-4:])}")
    has_space  = " " in val
    has_nl     = "\n" in val or "\r" in val
    has_tab    = "\t" in val
    has_quote  = '"' in val or "'" in val
    print(f"    space={has_space}  nl={has_nl}  tab={has_tab}  quote={has_quote}")


# ─────────────────────────────────────────────────────────────────────────────
# A. .env 読み込み経路
# ─────────────────────────────────────────────────────────────────────────────

def check_env_files() -> dict[str, str | None]:
    ts = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
    print(f"\n{'='*60}")
    print(f"[A] .env 読み込み経路  ({ts})")
    print(f"{'='*60}")

    project_root = Path(__file__).resolve().parents[2]   # C:\ai-trading
    src_dir      = Path(__file__).resolve().parents[1]   # C:\ai-trading\src

    candidates = {
        "ROOT .env (C:\\ai-trading\\.env)": project_root / ".env",
        "SRC  .env (src\\.env)           ": src_dir / ".env",
        "paths._env_file (src/.env)      ": src_dir / ".env",   # PROJECT_ROOT=src
        "client._ENV_PATH (src/../.env)  ": (src_dir / "kabusapi" / "client.py").parents[1] / ".env",
    }

    results: dict[str, str | None] = {}
    for label, path in candidates.items():
        exists = path.exists()
        has_pw = False
        pw_len = 0
        if exists:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("KABU_API_PASSWORD="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        has_pw = bool(val)
                        pw_len = len(val)
                        break
            except Exception:
                pass
        print(f"  {label} exists={exists}  has_KABU_PW={has_pw}  pw_len={pw_len}")
        results[str(path)] = str(path) if (exists and has_pw) else None

    print()
    print("  Import 優先順序:")
    print("    1. run_live_signal.py → import src.paths")
    print("       → load_dotenv(src/.env, override=False)  [最初にセット]")
    print("    2. capital block (lazy) → from src.kabusapi.client import KabuClient")
    print("       → load_dotenv(src/.env, override=False)  [同ファイル, no-op]")
    print("  結論: 両者とも src/.env のみを使用。競合なし。")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# B. パスワード詳細（環境変数・KabuClientインスタンス）
# ─────────────────────────────────────────────────────────────────────────────

def check_password_details() -> tuple[str, str]:
    print(f"\n{'='*60}")
    print("[B] パスワード詳細")
    print(f"{'='*60}")

    # 1. pathsインポート前の環境変数
    pw_before = os.environ.get("KABU_API_PASSWORD", "")
    print(f"\n  [B-1] paths.py import前 (os.environ):")
    _pw_detail(pw_before, "KABU_API_PASSWORD")

    # 2. paths import → load_dotenv(src/.env)
    import src.paths  # noqa: F401
    pw_after_paths = os.environ.get("KABU_API_PASSWORD", "")
    print(f"\n  [B-2] paths.py import後 (dotenv loaded):")
    _pw_detail(pw_after_paths, "KABU_API_PASSWORD")

    # 3. KabuClient instantiation
    from src.kabusapi.client import KabuClient
    client = KabuClient()
    pw_client = client._api_password
    print(f"\n  [B-3] KabuClient()._api_password:")
    _pw_detail(pw_client, "_api_password")

    changed = pw_before != pw_after_paths
    match   = pw_after_paths == pw_client
    print(f"\n  [B-summary] dotenv changed env={changed}  client matches env={match}")

    return pw_after_paths, pw_client


# ─────────────────────────────────────────────────────────────────────────────
# C. kabu Station 到達確認
# ─────────────────────────────────────────────────────────────────────────────

def check_port(host: str = "localhost", port: int = 18080) -> bool:
    print(f"\n{'='*60}")
    print("[C] kabu Station 到達確認")
    print(f"{'='*60}")
    try:
        with socket.create_connection((host, port), timeout=3):
            print(f"  port {port}: REACHABLE")
            return True
    except OSError as e:
        print(f"  port {port}: UNREACHABLE — {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# D. /token POST 実際の応答
# ─────────────────────────────────────────────────────────────────────────────

def check_token_request(pw: str, port: int = 18080) -> bool:
    import requests as _requests

    print(f"\n{'='*60}")
    print("[D] POST /kabusapi/token 実応答")
    print(f"{'='*60}")
    print(f"  送信payload: APIPassword=<masked len={len(pw)}>")

    try:
        resp = _requests.post(
            f"http://localhost:{port}/kabusapi/token",
            json={"APIPassword": pw},
            timeout=5,
        )
        print(f"  HTTP status : {resp.status_code}")
        print(f"  response    : {resp.text[:200]}")
        print(f"  content-type: {resp.headers.get('Content-Type','')}")
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("Token", "")
            print(f"  Token       : {token[:8]}...{token[-4:]}  [len={len(token)}]")
            return True
        elif resp.status_code == 401:
            print()
            print("  [401 診断]")
            print("  kabu Station は応答しているが、パスワードを拒否した。")
            print("  考えられる原因:")
            print("    1. kabu Station の API パスワードが src/.env と不一致")
            print("    2. kabu Station が起動中だがブローカーログイン未完了")
            print("    3. kabu Station の API 機能が無効化されている")
            print("  確認手順:")
            print("    kabu Station > ツール設定 > Web API > API パスワード")
            return False
        else:
            print(f"  unexpected status: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  request failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# E. Windows 環境変数レベル確認
# ─────────────────────────────────────────────────────────────────────────────

def check_windows_env() -> None:
    print(f"\n{'='*60}")
    print("[E] Windows 環境変数レベル確認")
    print(f"{'='*60}")
    import ctypes

    for scope_name, scope_id in [("Machine", "MACHINE"), ("User", "USER"), ("Process", "PROCESS")]:
        try:
            import winreg
            if scope_id == "MACHINE":
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                    r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
            elif scope_id == "USER":
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
            else:
                val = os.environ.get("KABU_API_PASSWORD", "")
                print(f"  PROCESS  KABU_API_PASSWORD: {'SET len=' + str(len(val)) if val else 'NOT_SET'}")
                continue
            try:
                val, _ = winreg.QueryValueEx(key, "KABU_API_PASSWORD")
                print(f"  {scope_id:8} KABU_API_PASSWORD: SET len={len(val)}")
            except FileNotFoundError:
                print(f"  {scope_id:8} KABU_API_PASSWORD: NOT_SET")
            winreg.CloseKey(key)
        except Exception as e:
            print(f"  {scope_id:8} check failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "="*60)
    print("  kabu Station API 認証診断")
    print("  run: " + datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST"))
    print("="*60)

    env_files = check_env_files()
    pw_env, pw_client = check_password_details()
    check_windows_env()
    port_ok = check_port()
    token_ok = False
    if port_ok:
        token_ok = check_token_request(pw_client)

    print(f"\n{'='*60}")
    print("[SUMMARY]")
    print(f"{'='*60}")
    src_env_has_pw = any(v is not None for v in env_files.values())
    print(f"  src/.env has KABU_API_PASSWORD : {src_env_has_pw}")
    print(f"  KabuClient._api_password len   : {len(pw_client)}")
    print(f"  port 18080 reachable           : {port_ok}")
    print(f"  /token POST success            : {token_ok}")

    if not port_ok:
        print("\n  => kabu Station が起動していません。アプリを起動してください。")
    elif not token_ok:
        print("\n  => kabu Station は応答しているが認証失敗。")
        print("     src/.env の KABU_API_PASSWORD を kabu Station の API パスワードと一致させてください。")
    else:
        print("\n  => 認証 OK。明日の LIVE run は通過するはずです。")

    print()


if __name__ == "__main__":
    main()
