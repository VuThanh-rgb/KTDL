"""Tài khoản, mật khẩu (băm), giới hạn đăng nhập sai, lưu gu phim. Dùng SQLite trong data/app.db."""
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

DB_PATH = Path(__file__).resolve().parent / "data" / "app.db"
USER_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")

# Tài khoản quản trị: đặt bằng biến môi trường ADMIN_USER / ADMIN_PASSWORD
ADMIN_USER = (os.environ.get("ADMIN_USER") or "admin").strip() or "admin"
_ADMIN_PW = os.environ.get("ADMIN_PASSWORD")
USING_DEFAULT_ADMIN_PW = not _ADMIN_PW
ADMIN_HASH = generate_password_hash(_ADMIN_PW or "admin123")
_DUMMY_HASH = generate_password_hash("khong-ai-dung-mat-khau-nay")


@contextmanager
def db():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY COLLATE NOCASE,"
                  " pw_hash TEXT NOT NULL, created TEXT NOT NULL)")
        c.execute("CREATE TABLE IF NOT EXISTS prefs(key TEXT PRIMARY KEY,"
                  " liked TEXT NOT NULL DEFAULT '[]', disliked TEXT NOT NULL DEFAULT '[]')")


def secret_key():
    """Khoá ký phiên đăng nhập: biến SECRET_KEY hoặc file data/.secret_key (tạo một lần)."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    p = DB_PATH.parent / ".secret_key"
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        p.write_text(secrets.token_hex(32))
    return p.read_text().strip()


# ---- người dùng ----
def create_user(username, password):
    """Trả về (tên chuẩn, None) nếu thành công, hoặc (None, thông báo lỗi)."""
    username = (username or "").strip()
    if not USER_RE.match(username):
        return None, "Tên đăng nhập gồm 3–20 ký tự: chữ cái, số hoặc dấu gạch dưới"
    if username.lower() == ADMIN_USER.lower():
        return None, "Tên đăng nhập này không được phép dùng"
    if not 6 <= len(password or "") <= 128:
        return None, "Mật khẩu phải từ 6 đến 128 ký tự"
    try:
        with db() as c:
            c.execute("INSERT INTO users(username, pw_hash, created) VALUES(?,?,?)",
                      (username, generate_password_hash(password), datetime.now().strftime("%Y-%m-%d %H:%M")))
    except sqlite3.IntegrityError:
        return None, "Tên đăng nhập đã tồn tại"
    return username, None


def verify_user(username, password):
    """Trả về tên chuẩn nếu đúng, ngược lại None. Luôn băm một lần để thời gian phản hồi giống nhau."""
    with db() as c:
        row = c.execute("SELECT username, pw_hash FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
    ok = check_password_hash(row["pw_hash"] if row else _DUMMY_HASH, password or "")
    return row["username"] if (row and ok) else None


def verify_admin(username, password):
    ok_u = hmac.compare_digest((username or "").encode(), ADMIN_USER.encode())
    ok_p = check_password_hash(ADMIN_HASH, password or "")
    return ok_u and ok_p


def list_users():
    with db() as c:
        rows = c.execute("SELECT u.username, u.created, p.liked FROM users u "
                         "LEFT JOIN prefs p ON p.key = u.username ORDER BY u.created DESC").fetchall()
    return [dict(username=r["username"], created=r["created"],
                 n_liked=len(json.loads(r["liked"])) if r["liked"] else 0) for r in rows]


def delete_user(username):
    with db() as c:
        n = c.execute("DELETE FROM users WHERE username=?", (username,)).rowcount
        c.execute("DELETE FROM prefs WHERE key=?", (username,))
    return n > 0


# ---- gu phim ----
def get_prefs(key):
    with db() as c:
        r = c.execute("SELECT liked, disliked FROM prefs WHERE key=?", (key,)).fetchone()
    return (json.loads(r["liked"]), json.loads(r["disliked"])) if r else ([], [])


def set_prefs(key, liked, disliked):
    with db() as c:
        c.execute("INSERT INTO prefs(key, liked, disliked) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET "
                  "liked=excluded.liked, disliked=excluded.disliked", (key, json.dumps(liked), json.dumps(disliked)))


# ---- giới hạn đăng nhập sai: 5 lần sai -> khoá 60 giây ----
MAX_FAILS, LOCK_SEC = 5, 60
_fails, _lock = {}, threading.Lock()


def locked_for(key):
    with _lock:
        rec = _fails.get(key)
        return int(rec[1] - time.time()) + 1 if rec and rec[1] > time.time() else 0


def register_fail(key):
    with _lock:
        n = _fails.get(key, (0, 0))[0] + 1
        _fails[key] = (0, time.time() + LOCK_SEC) if n >= MAX_FAILS else (n, 0)


def register_success(key):
    with _lock:
        _fails.pop(key, None)
