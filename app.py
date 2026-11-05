"""Demo Streamlit:  streamlit run app.py"""
import html
import re

import pandas as pd
import streamlit as st
import recsys

st.set_page_config(page_title="KNN Movie Recommender", page_icon="🎬", layout="wide")
EXPLORE_K, EXPLORE_SHRINK = 50, 10          # cấu hình cố định cho khu vực người xem

GENRE_STYLE = {   # genre: (emoji, màu trên, màu dưới)
    "Action": ("💥", "#ef4444", "#7f1d1d"), "Adventure": ("🧭", "#f59e0b", "#78350f"),
    "Animation": ("🎨", "#ec4899", "#831843"), "Children's": ("🧸", "#f472b6", "#9d174d"),
    "Children": ("🧸", "#f472b6", "#9d174d"), "Comedy": ("😂", "#ca8a04", "#713f12"),
    "Crime": ("🕵️", "#64748b", "#0f172a"), "Documentary": ("🎥", "#14b8a6", "#134e4a"),
    "Drama": ("🎭", "#8b5cf6", "#3b0764"), "Fantasy": ("🧙", "#a855f7", "#4c1d95"),
    "Film-Noir": ("🎞️", "#475569", "#020617"), "Horror": ("👻", "#991b1b", "#1c1917"),
    "Musical": ("🎵", "#06b6d4", "#164e63"), "Mystery": ("🔍", "#6366f1", "#1e1b4b"),
    "Romance": ("💘", "#fb7185", "#9f1239"), "Sci-Fi": ("🚀", "#3b82f6", "#1e3a8a"),
    "Thriller": ("🔪", "#dc2626", "#450a0a"), "War": ("🪖", "#78716c", "#292524"),
    "Western": ("🤠", "#d97706", "#451a03"), "IMAX": ("📽️", "#0ea5e9", "#0c4a6e"),
}
DEFAULT_STYLE = ("🎬", "#6b7280", "#1f2937")

st.markdown("""<style>
.hero{padding:1.4rem 1.6rem;border-radius:16px;margin-bottom:1rem;color:#fff;
background:linear-gradient(120deg,#4f46e5,#9333ea 55%,#ec4899)}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:.3rem 0 0;opacity:.92}
.mcard{border-radius:14px;padding:.9rem .8rem;color:#fff;min-height:215px;margin-bottom:.8rem;
display:flex;flex-direction:column;box-shadow:0 4px 14px rgba(0,0,0,.25)}
.mcard .em{font-size:2.3rem;line-height:1}.mcard .yr{font-size:.75rem;opacity:.8;margin-top:.2rem}
.mcard .tt{font-weight:700;font-size:1rem;line-height:1.2;margin:.3rem 0;flex:1}
.mcard .ch{display:inline-block;background:rgba(255,255,255,.2);border-radius:999px;
padding:1px 8px;font-size:.68rem;margin:0 3px 3px 0}
.mcard .bd{font-size:.8rem;font-weight:600;margin-top:.3rem}
.mcard .nt{font-size:.7rem;opacity:.85;margin-top:.15rem}
</style>""", unsafe_allow_html=True)


def split_title(t):
    m = re.search(r"\((\d{4})\)\s*$", t)
    year, name = (m.group(1), t[:m.start()].strip()) if m else ("", t)
    m2 = re.match(r"^(.*), (The|A|An)$", name)
    return (f"{m2.group(2)} {m2.group(1)}" if m2 else name), year


def card_html(title, genres, badge, note=""):
    gl = [g for g in str(genres).split("|") if g]
    emoji, c1, c2 = next((GENRE_STYLE[g] for g in gl if g in GENRE_STYLE), DEFAULT_STYLE)
    name, year = split_title(title)
    chips = "".join(f'<span class="ch">{html.escape(g)}</span>' for g in gl[:3])
    return (f'<div class="mcard" style="background:linear-gradient(160deg,{c1},{c2})">'
            f'<div class="em">{emoji}</div><div class="yr">{html.escape(year)}</div>'
            f'<div class="tt">{html.escape(name)}</div><div>{chips}</div>'
            f'<div class="bd">{html.escape(badge)}</div><div class="nt">{html.escape(note)}</div></div>')


def render_cards(movies_df, idx, badges, notes=None, per_row=5):
    for start in range(0, len(idx), per_row):
        cols = st.columns(per_row)
        for c, pos in zip(cols, range(start, min(start + per_row, len(idx)))):
            row = movies_df.loc[idx[pos]]
            c.markdown(card_html(row.title, row.genres, badges[pos], notes[pos] if notes else ""),
                       unsafe_allow_html=True)


@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load(ds):
    ratings, movies = recsys.load_movielens(ds)
    movies = movies.copy()
    dup = movies.title.duplicated(keep=False)
    movies["label"] = movies.title.where(~dup, movies.title + " [id " + movies.movie_id.astype(str) + "]")
    return ratings, movies


@st.cache_resource(show_spinner="Đang huấn luyện KNN...")
def train(ds, mode, k, shrink):
    ratings, movies = load(ds)
    R = recsys.to_matrix(ratings, int(ratings.user.max()) + 1, len(movies))
    return recsys.KNNRecommender(k, mode, shrink).fit(R)


names = list(recsys.list_datasets())
if not names:
    st.error("Không thấy dữ liệu MovieLens. Hãy tải tay, giải nén thư mục (vd ml-100k) cạnh app.py "
             "hoặc vào data/, rồi tải lại trang.")
    st.stop()

st.markdown('<div class="hero"><h1>🎬 Gợi ý phim bằng KNN</h1>'
            '<p>Chọn vài bộ phim bạn thích, hệ thống tìm những phim có người xem giống bạn cũng thích.</p></div>',
            unsafe_allow_html=True)
explore, lab = st.tabs(["🍿 Khám phá phim", "🔬 Thử nghiệm mô hình"])

# ----------------------------------------------------------------------------
# Khu vực người xem phim: không có tham số kỹ thuật, hiển thị bằng thẻ phim
# ----------------------------------------------------------------------------
with explore:
    ratings, movies = load("auto")
    model = train("auto", "item", EXPLORE_K, EXPLORE_SHRINK)
    item_of = dict(zip(movies.label, movies.item))
    labels = movies.label.tolist()

    st.subheader("Bạn thích phim nào?")
    pick = st.selectbox("Chọn (hoặc gõ để tìm) một bộ phim", labels, index=None, placeholder="Gõ tên phim...")
    if pick:
        idx, sim = model.similar_items(item_of[pick], 10, min_ratings=20)
        st.caption(f"Nếu bạn thích **{pick}**, có thể bạn cũng sẽ thích:")
        render_cards(movies, idx, [f"Độ giống {v:.2f}" for v in sim], [f"Giống {pick.split(' (')[0]}"] * len(idx))

    st.divider()
    st.subheader("Hoặc cho mình biết gu của bạn")
    liked = st.multiselect("Những phim bạn thích (chọn vài phim)", labels)
    disliked = st.multiselect("Những phim bạn không thích (không bắt buộc)", [l for l in labels if l not in liked])
    if st.button("Gợi ý cho tôi", type="primary"):
        if not liked:
            st.warning("Hãy chọn ít nhất một phim bạn thích.")
        else:
            prof = {item_of[l]: 5 for l in liked} | {item_of[l]: 1 for l in disliked}
            idx, pred, _ = model.recommend_for_ratings(prof, 10)
            lk = [item_of[l] for l in liked]
            why = [split_title(movies.title[lk[int(model.item_sim_[i, lk].argmax())]])[0] for i in idx]
            st.caption("Dành cho bạn:")
            render_cards(movies, idx, [f"⭐ {p:.1f} dự đoán" for p in pred], [f"Vì bạn thích {w}" for w in why])

# ----------------------------------------------------------------------------
# Khu vực thử nghiệm: tham số, user_id có sẵn, chỉ số đánh giá
# ----------------------------------------------------------------------------
with lab:
    st.caption("Dành cho thử nghiệm và báo cáo: mã người dùng có sẵn của MovieLens, chỉnh K/shrinkage, xem chỉ số đánh giá.")
    t_user, t_eval = st.tabs(["👤 Gợi ý theo mã người dùng", "📊 Đánh giá & so sánh cấu hình"])

    with t_user:
        c1, c2, c3, c4 = st.columns(4)
        ds = c1.selectbox("Bộ dữ liệu", names, key="u_ds")
        mode = c2.radio("Thuật toán", ["item", "user"], horizontal=True,
                        format_func=lambda m: "Item-based" if m == "item" else "User-based")
        k = c3.select_slider("K", [5, 10, 20, 50, 100, 200], value=50)
        shrink = c4.slider("Shrinkage", 0, 100, 10)
        r_, m_ = load(ds)
        uid = st.number_input("Mã người dùng (user_id)", int(r_.user_id.min()), int(r_.user_id.max()),
                              int(r_.user_id.min()))
        if st.button("Gợi ý cho người dùng này", type="primary"):
            model_u = train(ds, mode, k, shrink)
            sel = r_.loc[r_.user_id == uid, "user"]
            if sel.empty:
                st.warning("Không có người dùng này trong dữ liệu.")
            else:
                u = int(sel.iloc[0])
                idx, pred = model_u.recommend(u, 10)
                out = m_.loc[idx, ["title", "genres"]].copy()
                out["điểm dự đoán (1-5)"] = pred.round(2)
                st.subheader("Phim được gợi ý")
                st.dataframe(out.reset_index(drop=True), hide_index=True)
                hist = r_[r_.user == u].sort_values("rating", ascending=False).head(10)
                st.subheader("Phim người này đã chấm cao (để đối chiếu)")
                st.dataframe(hist.merge(m_, on="item")[["title", "genres", "rating"]],
                             hide_index=True)

    with t_eval:
        sel_ds = st.multiselect("Bộ dữ liệu", names, default=names[:1])
        e1, e2, e3 = st.columns(3)
        ks = e1.multiselect("Các giá trị K", [5, 10, 20, 50, 100, 200], default=[5, 20, 50])
        n_seeds = e2.slider("Số seed (lần chia dữ liệu)", 1, 5, 2)
        e_shrink = e3.slider("Shrinkage", 0, 100, 10, key="e_shrink")
        per_user = st.checkbox("Chia train/val/test riêng cho từng người dùng", value=True)
        st.caption("Chọn K trên validation theo 2 mục tiêu (ranking = Precision@10, rating = RMSE), "
                   "đánh giá test một lần. Bộ lớn × nhiều seed có thể mất vài phút.")
        if st.button("Chạy đánh giá", type="primary") and sel_ds and ks:
            with st.spinner("Đang chạy..."):
                final, curves, info = recsys.select_and_evaluate(
                    sel_ds, ks=sorted(ks), shrink=e_shrink, seeds=list(range(n_seeds)),
                    per_user=per_user, quiet=True)
            st.subheader("Thông tin dữ liệu")
            st.dataframe(info, hide_index=True)
            st.subheader("Kết quả trên test (trung bình ± độ lệch chuẩn)")
            st.dataframe(recsys.summarize(final, ["dataset", "mode", "goal"],
                                          ["P@10", "R@10", "test_RMSE", "test_MAE", "fit_sec"]),
                         hide_index=True)
            for col, title in [("val_RMSE", "RMSE trên validation theo K"),
                               ("val_P@10", "Precision@10 trên validation theo K")]:
                st.caption(title)
                g = curves.assign(series=curves.dataset + " · " + curves["mode"])
                st.line_chart(g.groupby(["K", "series"])[col].mean().unstack())

st.caption("Dữ liệu: MovieLens (GroupLens Research, University of Minnesota). Thuật toán: KNN collaborative filtering.")