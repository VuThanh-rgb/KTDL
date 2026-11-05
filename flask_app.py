"""Giao diện Flask kiểu Netflix, dùng chung recsys.py.
Chạy:  python flask_app.py   ->  mở http://127.0.0.1:5000
Tuỳ chọn:  MOVIE_DATASET=ml-1m  (chọn bộ dữ liệu)  |  TMDB_API_KEY=...  (hiện poster thật)
"""
import os
import re
import threading
import urllib.parse
import urllib.request
import json
from datetime import timedelta
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, session

import auth
import recsys

try:                                   # dùng kho chứng chỉ của Windows nếu có (tránh lỗi SSL)
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

K, SHRINK = 50, 10
TMDB_KEY = os.environ.get("TMDB_API_KEY", "").strip()
app = Flask(__name__)
app.json.ensure_ascii = False
auth.init_db()
app.config.update(SECRET_KEY=auth.secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(days=7), MAX_CONTENT_LENGTH=64 * 1024)
PUBLIC_API = {"/api/auth/login", "/api/auth/register"}

# ----------------------------------------------------------------------------
# Nạp dữ liệu + huấn luyện một lần khi khởi động
# ----------------------------------------------------------------------------
ratings, movies = recsys.load_movielens(os.environ.get("MOVIE_DATASET", "auto"))
N_U, N_I = int(ratings.user.max()) + 1, len(movies)
R = recsys.to_matrix(ratings, N_U, N_I)
print("Đang huấn luyện Item-KNN ...")
item_model = recsys.KNNRecommender(K, "item", SHRINK).fit(R)
cnt = item_model.M.sum(0)
avg = np.where(cnt > 0, R.sum(0) / np.maximum(cnt, 1), 0)
m_ = float(np.percentile(cnt[cnt > 0], 60))                       # điểm "Bayesian" cho bảng xếp hạng
bayes = cnt / (cnt + m_) * avg + m_ / (cnt + m_) * avg[cnt > 0].mean()


def split_title(t):
    m = re.search(r"\((\d{4})\)\s*$", t)
    year, name = (m.group(1), t[:m.start()].strip()) if m else ("", t)
    m2 = re.match(r"^(.*), (The|A|An)$", name)
    return (f"{m2.group(2)} {m2.group(1)}" if m2 else name), year


META = []
for i, (t, g) in enumerate(zip(movies.title, movies.genres)):
    name, year = split_title(t)
    META.append(dict(id=i, title=name, year=year, n=int(cnt[i]), avg=round(float(avg[i]), 2),
                     genres=[x for x in str(g).split("|") if x and not x.startswith("(")]))


def card(i, **extra):
    return {**META[int(i)], **extra}


def build_rows(n=16):
    rows = [{"title": "🔥 Được xem nhiều nhất", "items": [card(i) for i in np.argsort(-cnt)[:n]]},
            {"title": "⭐ Được yêu thích nhất", "items": [card(i) for i in np.argsort(-bayes)[:n]]}]
    counts = {}
    for m in META:
        for g in m["genres"]:
            counts[g] = counts.get(g, 0) + 1
    for g in sorted(counts, key=counts.get, reverse=True)[:5]:
        mask = np.array([g in m["genres"] for m in META])
        idx = np.where(mask)[0]
        top = idx[np.argsort(-bayes[idx])][:n]
        rows.append({"title": f"Thể loại: {g}", "items": [card(i) for i in top]})
    return rows


ROWS = build_rows()
_want = os.environ.get("MOVIE_DATASET", "auto")
_found = recsys.list_datasets()
DATASET_LABEL = next((k for k in (_want, f"ml-{_want}") if k in _found), None) or (
    max(_found, key=lambda k: _found[k].stat().st_ctime) if _want == "auto" and _found else Path(_want).name)
UID_IDX = dict(zip(ratings.user_id.tolist(), ratings.user.tolist()))
_user_models, _lock = {}, threading.Lock()


def get_model(mode, k):
    if mode == "item" and k == K:
        return item_model
    with _lock:
        if (mode, k) not in _user_models:
            _user_models[(mode, k)] = recsys.KNNRecommender(k, mode, SHRINK).fit(R)
        return _user_models[(mode, k)]


def bad(msg, code=400):
    return jsonify(error=msg), code


def admin_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if session.get("role") != "admin":
            return bad("Chỉ quản trị viên được phép", 403)
        return f(*a, **k)
    return wrapper


def pref_key():
    return ("admin:" + session["u"]) if session["role"] == "admin" else session["u"]


def clean_ids(v):
    if not isinstance(v, list) or len(v) > 200:
        raise ValueError
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < N_I:
            raise ValueError
        if x not in out:
            out.append(x)
    return out


@app.before_request
def guard():
    p = request.path
    if not p.startswith("/api/"):
        return None
    if p not in PUBLIC_API and "role" not in session:
        return bad("Cần đăng nhập", 401)
    if request.method in ("POST", "PUT") and not request.is_json:
        return bad("Yêu cầu phải là JSON", 415)       # chặn form chéo trang (CSRF)
    return None


@app.after_request
def secure_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if resp.mimetype == "text/html" or request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    if "role" not in session:
        return redirect("/login")
    return render_template("index.html")


@app.get("/login")
def login_page():
    if "role" in session:
        return redirect("/")
    return render_template("login.html")


def start_session(name, role):
    session.clear()
    session.permanent = True
    session["u"], session["role"] = name, role


@app.post("/api/auth/login")
def api_login():
    b = request.get_json(silent=True) or {}
    role = b.get("role")
    username, password = str(b.get("username", "")).strip(), str(b.get("password", ""))
    if role not in ("user", "admin") or len(username) > 64 or len(password) > 128:
        return bad("Thông tin đăng nhập không hợp lệ")
    key = (request.remote_addr, role, username.lower())
    wait = auth.locked_for(key)
    if wait:
        return bad(f"Sai quá nhiều lần. Thử lại sau {wait} giây", 429)
    name = (auth.ADMIN_USER if auth.verify_admin(username, password) else None) if role == "admin" \
        else auth.verify_user(username, password)
    if not name:
        auth.register_fail(key)
        return bad("Sai tên đăng nhập hoặc mật khẩu", 401)
    auth.register_success(key)
    start_session(name, role)
    return jsonify(ok=True, role=role, username=name)


@app.post("/api/auth/register")
def api_register():
    b = request.get_json(silent=True) or {}
    name, err = auth.create_user(str(b.get("username", "")), str(b.get("password", "")))
    if err:
        return bad(err)
    start_session(name, "user")
    return jsonify(ok=True, role="user", username=name)


@app.post("/api/auth/logout")
def api_logout():
    session.clear()
    return jsonify(ok=True)


@app.get("/api/auth/me")
def api_me():
    return jsonify(username=session["u"], role=session["role"])


@app.get("/api/prefs")
def api_get_prefs():
    liked, disliked = auth.get_prefs(pref_key())
    return jsonify(liked=[card(i) for i in liked if 0 <= i < N_I], disliked=[card(i) for i in disliked if 0 <= i < N_I])


@app.put("/api/prefs")
def api_put_prefs():
    b = request.get_json(silent=True) or {}
    try:
        liked, disliked = clean_ids(b.get("liked", [])), clean_ids(b.get("disliked", []))
    except ValueError:
        return bad("Danh sách phim không hợp lệ")
    auth.set_prefs(pref_key(), liked, [i for i in disliked if i not in liked])
    return jsonify(ok=True)


@app.get("/api/admin/stats")
@admin_required
def api_admin_stats():
    return jsonify(dataset=DATASET_LABEL, movies=N_I, users=N_U, ratings=len(ratings), k=K, shrink=SHRINK,
                   sparsity=round(1 - len(ratings) / (N_U * N_I), 4),
                   est_mem_gb=round(recsys.estimate_memory_gb(N_U, N_I), 2), accounts=len(auth.list_users()))


@app.get("/api/admin/users")
@admin_required
def api_admin_users():
    return jsonify(auth.list_users())


@app.delete("/api/admin/users/<username>")
@admin_required
def api_admin_delete(username):
    if not auth.USER_RE.match(username) or not auth.delete_user(username):
        return bad("Không tìm thấy tài khoản", 404)
    return jsonify(ok=True)


@app.get("/api/info")
def api_info():
    return jsonify(movies=N_I, users=N_U, ratings=len(ratings), posters=bool(TMDB_KEY), k=K,
                   user_id_min=int(ratings.user_id.min()), user_id_max=int(ratings.user_id.max()))


@app.get("/api/rows")
def api_rows():
    return jsonify(ROWS)


@app.get("/api/movies")
def api_movies():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])
    hits = [i for i, m in enumerate(META) if q in m["title"].lower()]
    hits.sort(key=lambda i: -cnt[i])
    return jsonify([card(i) for i in hits[:12]])


@app.get("/api/similar")
def api_similar():
    try:
        item, n = int(request.args["id"]), min(int(request.args.get("n", 10)), 30)
        assert 0 <= item < N_I
    except Exception:
        return bad("Tham số id không hợp lệ")
    idx, sim = item_model.similar_items(item, n, min_ratings=20)
    return jsonify([card(i, badge=f"Độ giống {s:.2f}") for i, s in zip(idx, sim) if np.isfinite(s)])


@app.post("/api/profile")
def api_profile():
    body = request.get_json(silent=True) or {}
    try:
        liked = [int(x) for x in body.get("liked", [])]
        disliked = [int(x) for x in body.get("disliked", [])]
        assert all(0 <= x < N_I for x in liked + disliked)
    except Exception:
        return bad("Danh sách phim không hợp lệ")
    if not liked:
        return bad("Hãy chọn ít nhất một phim bạn thích")
    prof = {i: 1 for i in disliked} | {i: 5 for i in liked}
    idx, pred, _ = item_model.recommend_for_ratings(prof, 12)
    S = item_model.item_sim_
    out = []
    for i, p in zip(idx, pred):
        why = META[liked[int(S[i, liked].argmax())]]["title"]
        out.append(card(i, badge=f"⭐ {p:.1f} dự đoán", note=f"Vì bạn thích {why}"))
    return jsonify(out)


@app.get("/api/user")
@admin_required
def api_user():
    try:
        uid, k = int(request.args["uid"]), int(request.args.get("k", K))
        mode = request.args.get("mode", "item")
        assert mode in ("item", "user") and 1 <= k <= 200
    except Exception:
        return bad("Tham số không hợp lệ")
    if uid not in UID_IDX:
        return bad(f"Không có người dùng {uid} trong dữ liệu", 404)
    u = UID_IDX[uid]
    model = get_model(mode, k)
    idx, pred = model.recommend(u, 12)
    hist = ratings[ratings.user == u].sort_values("rating", ascending=False).head(8)
    return jsonify(recs=[card(i, badge=f"⭐ {p:.1f} dự đoán") for i, p in zip(idx, pred)],
                   history=[card(i, badge=f"Đã chấm {int(r)}★") for i, r in zip(hist.item, hist.rating)])


_poster_cache = {}


@app.get("/api/poster")
def api_poster():
    if not TMDB_KEY:
        return jsonify(url=None)
    try:
        i = int(request.args["id"])
        m = META[i]
    except Exception:
        return bad("id không hợp lệ")
    if i not in _poster_cache:
        url = None
        try:
            q = {"api_key": TMDB_KEY, "query": m["title"]}
            if m["year"]:
                q["year"] = m["year"]
            with urllib.request.urlopen("https://api.themoviedb.org/3/search/movie?" + urllib.parse.urlencode(q),
                                        timeout=6) as r:
                res = json.load(r).get("results", [])
            if res and res[0].get("poster_path"):
                url = "https://image.tmdb.org/t/p/w342" + res[0]["poster_path"]
        except Exception:
            pass
        _poster_cache[i] = url
    return jsonify(url=_poster_cache[i])


if __name__ == "__main__":
    if auth.USING_DEFAULT_ADMIN_PW:
        print(f"\n⚠  Đang dùng tài khoản quản trị mặc định: {auth.ADMIN_USER} / admin123."
              "\n   Nên đổi:  $env:ADMIN_PASSWORD=\"mat_khau_cua_ban\"  (PowerShell) trước khi chạy.\n")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
