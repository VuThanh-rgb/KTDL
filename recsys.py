"""Hệ thống gợi ý phim KNN trên MovieLens (dùng chung cho notebook và Streamlit)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "data"
URLS = {  # chỉ để in hướng dẫn tải tay
    "100k": "https://grouplens.org/datasets/movielens/100k/",
    "1m": "https://grouplens.org/datasets/movielens/1m/",
}
GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]


# ----------------------------------------------------------------------------
# 1. Dữ liệu
# ----------------------------------------------------------------------------
def _kind(folder):
    """Đoán định dạng MovieLens từ tên file trong thư mục."""
    f = lambda n: (folder / n).exists()
    if f("u.data") and f("u.item"):
        return "100k"
    if f("ratings.dat") and f("movies.dat"):
        return "dat"          # 1M, 10M
    if f("ratings.csv") and f("movies.csv"):
        return "csv"          # latest-small, 25M...
    return None


def list_datasets():
    """Quét máy (thư mục data/, thư mục dự án, thư mục cha, thư mục đang chạy)
    và trả về {tên thư mục: đường dẫn} của mọi bộ MovieLens tìm thấy."""
    here = Path(__file__).resolve().parent
    found = {}
    for base in (DATA_DIR, here, here.parent, Path.cwd()):
        if not base.is_dir():
            continue
        cands = [base]
        for d in sorted(base.iterdir()):
            if d.is_dir() and not d.name.startswith((".", "_")):
                cands.append(d)
                try:
                    cands += [x for x in sorted(d.iterdir()) if x.is_dir() and not x.name.startswith(".")]
                except OSError:
                    pass
        for c in cands:
            if _kind(c) and c.name not in found:
                found[c.name] = c
    return found


def load_movielens(name="auto", folder=None):
    """Tự nhận dữ liệu có sẵn. name: "auto" | "100k" | "1m" | tên thư mục | đường dẫn thư mục.
    Chỉ tải 100k/1m khi không thấy trên máy.
    ratings: user, item (chỉ số 0..n-1), rating, user_id, movie_id
    movies : item, movie_id, title, genres
    """
    if folder is None and Path(str(name)).is_dir():
        folder = name
    if folder is None:
        found = list_datasets()
        key = next((k for k in (name, f"ml-{name}") if k in found), None)
        if key is None and name == "auto" and found:
            # nhiều bộ cùng tồn tại -> lấy bộ mới thêm vào máy gần nhất
            key = max(found, key=lambda k: found[k].stat().st_ctime)
        if key is None:
            raise FileNotFoundError(
                f"Không tìm thấy dữ liệu MovieLens ('{name}').\n"
                "Hãy tải tay (ví dụ: " + ", ".join(URLS.values()) + "), giải nén,\n"
                "rồi đặt thư mục (ml-100k, ml-1m, ml-latest-small...) cạnh recsys.py hoặc vào thư mục data/.")
        folder = found[key]
        if len(found) > 1:
            print("Các bộ dữ liệu tìm thấy:", list(found), "| đang dùng:", key,
                  "(muốn chọn bộ khác: DATASET = tên thư mục)")
    folder = Path(folder)
    kind = _kind(folder)
    if kind is None:
        raise FileNotFoundError(f"Thư mục {folder} không có u.data / ratings.dat / ratings.csv")
    print("Dùng dữ liệu tại:", folder, f"(định dạng {kind})")

    cols = ["user_id", "movie_id", "rating", "ts"]
    if kind == "100k":
        ratings = pd.read_csv(folder / "u.data", sep="\t", names=cols)
        raw = pd.read_csv(folder / "u.item", sep="|", header=None, encoding="latin-1")
        flags = raw.iloc[:, 5:24].values
        movies = pd.DataFrame({
            "movie_id": raw[0], "title": raw[1],
            "genres": ["|".join(GENRES[i] for i in np.flatnonzero(f)) for f in flags]})
    elif kind == "dat":
        ratings = pd.read_csv(folder / "ratings.dat", sep="::", engine="python", names=cols)
        movies = pd.read_csv(folder / "movies.dat", sep="::", engine="python",
                             names=["movie_id", "title", "genres"], encoding="latin-1")
    else:
        ratings = pd.read_csv(folder / "ratings.csv").rename(
            columns={"userId": "user_id", "movieId": "movie_id", "timestamp": "ts"})
        movies = pd.read_csv(folder / "movies.csv").rename(columns={"movieId": "movie_id"})

    movies = movies.sort_values("movie_id").reset_index(drop=True)
    movies["item"] = movies.index
    ratings["item"] = ratings["movie_id"].map(dict(zip(movies.movie_id, movies.item)))
    ratings["user"] = ratings["user_id"].astype("category").cat.codes.astype(int)
    return ratings.dropna(subset=["item"]).astype({"item": int}), movies


def to_matrix(df, n_users=None, n_items=None):
    """DataFrame -> ma trận user x item (0 = chưa đánh giá)."""
    n_users = n_users or int(df.user.max()) + 1
    n_items = n_items or int(df.item.max()) + 1
    R = np.zeros((n_users, n_items), dtype=np.float32)
    R[df.user.values, df.item.values] = df.rating.values
    return R


def split(ratings, test_size=0.2, seed=42):
    return train_test_split(ratings, test_size=test_size, random_state=seed)


# ----------------------------------------------------------------------------
# 2. KNN
# ----------------------------------------------------------------------------
MAX_GB = 4.0   # giới hạn RAM ước tính cho KNN dạng dense


def estimate_memory_gb(n_users, n_items):
    """Ước tính RAM (GB, float32): ~8 ma trận user x item + ~3 ma trận tương đồng."""
    return 4 * (8 * n_users * n_items + 3 * max(n_users, n_items) ** 2) / 1e9


def _topk(S, k):
    """Giữ k giá trị lớn nhất mỗi hàng, còn lại = 0."""
    k = min(k, S.shape[1] - 1)          # K không vượt số láng giềng khả dụng
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    out = np.zeros_like(S)
    np.put_along_axis(out, idx, np.take_along_axis(S, idx, 1), 1)
    return out


class KNNRecommender:
    """KNN collaborative filtering.
    mode="item": phim giống phim (adjusted cosine) | mode="user": người giống người.
    shrink: giảm độ tin cậy của similarity khi có ít điểm đồng đánh giá (0 = tắt).
    """

    def __init__(self, k=20, mode="item", shrink=10):
        assert mode in ("item", "user")
        self.k, self.mode, self.shrink = k, mode, shrink

    def fit(self, R):
        R = R.astype(np.float32)
        est = estimate_memory_gb(*R.shape)
        if est > MAX_GB:
            raise MemoryError(f"Ước tính cần ~{est:.1f} GB RAM (giới hạn {MAX_GB} GB). "
                              "KNN dạng dense chỉ hợp với ml-100k, ml-1m, ml-latest-small.")
        lo, hi = float(R[R > 0].min()), float(R.max())
        M = (R > 0).astype(np.float32)
        um = R.sum(1) / np.maximum(M.sum(1), 1)          # điểm TB mỗi user
        Rc = (R - um[:, None]) * M                       # chuẩn hoá theo user
        X, XM = (Rc, M) if self.mode == "user" else (Rc.T, M.T)

        S = cosine_similarity(X).astype(np.float32)
        if self.shrink:
            co = XM @ XM.T                               # số điểm đồng đánh giá
            S *= co / (co + self.shrink)
        np.fill_diagonal(S, 0)
        self.item_sim_ = S if self.mode == "item" else None
        Sk = _topk(S, self.k)

        if self.mode == "user":
            base = um[:, None]
            num, den = Sk @ Rc, np.abs(Sk) @ M
        else:
            cnt = M.sum(0)
            base = np.where(cnt > 0, R.sum(0) / np.maximum(cnt, 1), R.sum() / M.sum())[None, :]
            D = (R - base) * M
            num, den = D @ Sk.T, M @ np.abs(Sk).T
        delta = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
        self.P = np.clip(base + delta, lo, hi)             # ma trận điểm dự đoán
        self.base_, self.lo_, self.hi_ = base, lo, hi
        L = (R >= 4).astype(np.float32)                  # phim user đã thích
        self.score = L @ Sk.T if self.mode == "item" else Sk @ L   # điểm xếp hạng top-N
        self.R, self.Rc, self.M = R, Rc, M
        return self

    def recommend(self, user, n=10, by="score"):
        """Top-n phim chưa xem của user: (chỉ số item, điểm dự đoán).
        by="score": xếp theo độ giống phim đã thích (khuyên dùng, tránh phim quá ít người xem)
        by="rating": xếp theo điểm dự đoán thuần."""
        key = self.score[user] if by == "score" else self.P[user]
        s = np.where(self.R[user] > 0, -np.inf, key)
        top = np.argsort(-s)[:n]
        return top, self.P[user, top]

    def similar_items(self, item, n=10, min_ratings=20):
        """Top-n phim giống `item` nhất (adjusted cosine), lọc phim quá ít lượt đánh giá."""
        if self.item_sim_ is None:
            S = cosine_similarity(self.Rc.T)
            np.fill_diagonal(S, 0)
            self.item_sim_ = S
        s = self.item_sim_[item].copy()
        s[self.M.sum(0) < min_ratings] = -np.inf
        s[item] = -np.inf
        top = np.argsort(-s)[:n]
        return top, s[top]

    def recommend_for_ratings(self, ratings, n=10):
        """Gợi ý cho người dùng MỚI từ vài phim họ chấm: ratings = {chỉ số item: điểm}.
        Chỉ dùng cho item-based. Trả về (item, điểm dự đoán, điểm xếp hạng)."""
        if self.mode != "item" or not ratings:
            raise ValueError("Cần mode='item' và ít nhất một phim đã chấm.")
        S = self.item_sim_
        r = np.zeros(S.shape[0], dtype=np.float32)
        for i, v in ratings.items():
            r[int(i)] = v
        m = (r > 0).astype(np.float32)
        D = (r - self.base_[0]) * m
        num, den = S @ D, np.abs(S) @ m
        pred = np.clip(self.base_[0] + np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9),
                       self.lo_, self.hi_)
        score = np.maximum(S, 0) @ (r >= 4).astype(np.float32)
        score[m > 0] = -np.inf
        top = np.argsort(-score)[:n]
        return top, pred[top], score[top]


# ----------------------------------------------------------------------------
# 3. Đánh giá
# ----------------------------------------------------------------------------
def rating_metrics(P, test):
    err = P[test.user.values, test.item.values] - test.rating.values
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def ranking_metrics(P, R_train, test, k=10, thr=4.0):
    """Precision@K, Recall@K. Phim 'liên quan' = phim trong tập test có rating >= thr."""
    S = np.where(R_train > 0, -np.inf, P)
    top = np.argpartition(-S, k, axis=1)[:, :k]
    rel = np.zeros(R_train.shape, dtype=bool)
    t = test[test.rating >= thr]
    rel[t.user.values, t.item.values] = True
    hits = np.take_along_axis(rel, top, 1).sum(1)
    n_rel = rel.sum(1)
    ok = n_rel > 0
    return float((hits[ok] / k).mean()), float((hits[ok] / n_rel[ok]).mean())


# ----------------------------------------------------------------------------
# 4. Quy trình đánh giá: chia theo từng user, nhiều seed, chọn K theo mục tiêu
# ----------------------------------------------------------------------------
_LOAD_CACHE = {}


def load_cached(name):
    if str(name) not in _LOAD_CACHE:
        _LOAD_CACHE[str(name)] = load_movielens(name)
    return _LOAD_CACHE[str(name)]


def split_per_user(ratings, test_frac=0.2, seed=42, min_train=5):
    """Mỗi user giữ ~test_frac số phim làm test (luôn để lại >= min_train phim cho train)."""
    rank = ratings.assign(_r=np.random.default_rng(seed).random(len(ratings))) \
                  .groupby("user")["_r"].rank(method="first")
    n = ratings.groupby("user")["rating"].transform("size")
    n_te = np.minimum(np.round(n * test_frac), np.maximum(n - min_train, 0))
    is_test = (rank <= n_te).values
    return ratings[~is_test], ratings[is_test]


def split3(ratings, val_size=0.1, test_size=0.2, seed=42, per_user=True):
    """70% train / 10% validation / 20% test. per_user=True: chia riêng từng user."""
    if per_user:
        train_val, test = split_per_user(ratings, test_size, seed)
        train, val = split_per_user(train_val, val_size / (1 - test_size), seed + 1)
    else:
        train_val, test = train_test_split(ratings, test_size=test_size, random_state=seed)
        train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), random_state=seed)
    return train, val, test


def _resolve(datasets):
    if datasets in (None, "all"):
        datasets = list(list_datasets())
    if isinstance(datasets, (str, Path)):
        datasets = [datasets]
    if not datasets:
        raise FileNotFoundError("Không tìm thấy bộ dữ liệu nào (xem hướng dẫn trong load_movielens).")
    return datasets


def select_and_evaluate(datasets="all", ks=(5, 10, 20, 50), modes=("item", "user"), shrink=10,
                        topn=10, seeds=(0,), per_user=True, quiet=False):
    """Với mỗi bộ dữ liệu, seed và mode:
      1) huấn luyện trên TRAIN với từng K, đo trên VALIDATION;
      2) chọn K theo HAI mục tiêu: goal="ranking" (Precision@topn cao nhất - gợi ý danh sách)
         và goal="rating" (RMSE thấp nhất - dự đoán điểm);
      3) huấn luyện lại trên TRAIN+VALIDATION, đánh giá TEST đúng một lần cho mỗi mục tiêu.
    Trả về (kết quả test, đường cong validation theo K, thông tin dữ liệu)."""
    final, curves, info = [], [], []
    P_ = f"P@{topn}"
    for name in _resolve(datasets):
        label = Path(str(name)).name
        ratings, movies = load_cached(name)
        n_u, n_i = int(ratings.user.max()) + 1, len(movies)
        est = estimate_memory_gb(n_u, n_i)
        info.append(dict(dataset=label, users=n_u, movies=n_i, ratings=len(ratings),
                         sparsity=round(1 - len(ratings) / (n_u * n_i), 4),
                         ratings_per_user=round(len(ratings) / n_u, 1),
                         ratings_per_movie=round(len(ratings) / n_i, 1), est_mem_gb=round(est, 2)))
        if est > MAX_GB:
            print(f"[{label}] ước tính {est:.1f} GB > giới hạn {MAX_GB} GB -> bỏ qua")
            continue
        for seed in seeds:
            train, val, test = split3(ratings, seed=seed, per_user=per_user)
            R_tr = to_matrix(train, n_u, n_i)
            R_full = to_matrix(pd.concat([train, val]), n_u, n_i)

            cnt = (R_full > 0).sum(0)                       # baseline (đánh giá trên test)
            mean = np.where(cnt > 0, R_full.sum(0) / np.maximum(cnt, 1), R_full.sum() / (R_full > 0).sum())
            mae, rmse = rating_metrics(np.tile(mean, (n_u, 1)), test)
            p, r = ranking_metrics(np.tile(cnt.astype(float), (n_u, 1)), R_full, test, k=topn)
            final.append(dict(dataset=label, seed=seed, mode="baseline", goal="baseline", best_K=0,
                              val_RMSE=np.nan, test_MAE=mae, test_RMSE=rmse, **{P_: p, f"R@{topn}": r}, fit_sec=0.0))
            for mode in modes:
                cur = []
                for k in ks:
                    t0 = time.time()
                    m = KNNRecommender(k, mode, shrink).fit(R_tr)
                    ft = time.time() - t0
                    mae, rmse = rating_metrics(m.P, val)
                    p, r = ranking_metrics(m.score, R_tr, val, k=topn)
                    cur.append(dict(dataset=label, seed=seed, mode=mode, K=k, val_MAE=mae, val_RMSE=rmse,
                                    **{f"val_{P_}": p, f"val_R@{topn}": r}, fit_sec=ft))
                    if not quiet:
                        print(f"[{label}|seed {seed}] {mode:4s} K={k:<3d} val RMSE={rmse:.4f} {P_}={p:.4f}")
                curves += cur
                picks = {"ranking": max(cur, key=lambda d: d[f"val_{P_}"]),
                         "rating": min(cur, key=lambda d: d["val_RMSE"])}
                fitted = {}
                for goal, best in picks.items():
                    k = best["K"]
                    if k not in fitted:
                        t0 = time.time()
                        fitted[k] = (KNNRecommender(k, mode, shrink).fit(R_full), time.time() - t0)
                    m, ft = fitted[k]
                    mae, rmse = rating_metrics(m.P, test)
                    p, r = ranking_metrics(m.score, R_full, test, k=topn)
                    final.append(dict(dataset=label, seed=seed, mode=mode, goal=goal, best_K=k,
                                      val_RMSE=best["val_RMSE"], test_MAE=mae, test_RMSE=rmse,
                                      **{P_: p, f"R@{topn}": r}, fit_sec=ft))
    return pd.DataFrame(final), pd.DataFrame(curves), pd.DataFrame(info)


def shrink_ablation(datasets="all", shrinks=(0, 10, 25, 50, 100), k=20, mode="item",
                    seeds=(0,), per_user=True, topn=10):
    """Ảnh hưởng của shrinkage, đo trên VALIDATION (không đụng tới test)."""
    rows = []
    for name in _resolve(datasets):
        label = Path(str(name)).name
        ratings, movies = load_cached(name)
        n_u, n_i = int(ratings.user.max()) + 1, len(movies)
        if estimate_memory_gb(n_u, n_i) > MAX_GB:
            continue
        for seed in seeds:
            train, val, _ = split3(ratings, seed=seed, per_user=per_user)
            R_tr = to_matrix(train, n_u, n_i)
            for sh in shrinks:
                m = KNNRecommender(k, mode, sh).fit(R_tr)
                rows.append(dict(dataset=label, seed=seed, shrink=sh,
                                 val_RMSE=rating_metrics(m.P, val)[1],
                                 **{f"val_P@{topn}": ranking_metrics(m.score, R_tr, val, k=topn)[0]}))
    return pd.DataFrame(rows)


def summarize(df, keys, cols, digits=4):
    """Gộp nhiều seed thành 'trung bình ± độ lệch chuẩn'."""
    g = df.groupby(keys, sort=False)
    out = g[cols].mean().round(digits).astype(str) + " ± " + g[cols].std().fillna(0).round(digits).astype(str)
    if "best_K" in df:
        out.insert(0, "best_K (mỗi seed)", g["best_K"].agg(lambda x: ", ".join(str(int(v)) for v in x)))
    out.insert(0, "n_seeds", g["seed"].nunique() if "seed" in df else 1)
    return out.reset_index()