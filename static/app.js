const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const h = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
};
const JSON_H = { 'Content-Type': 'application/json' };
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (r.status === 401) { window.location.href = '/login'; throw new Error('Cần đăng nhập'); }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
};
const GENRE = {
  'Action': ['💥', '#ef4444', '#b91c1c'], 'Adventure': ['🧭', '#f59e0b', '#c2410c'], 'Animation': ['🎨', '#ec4899', '#be185d'],
  "Children's": ['🧸', '#f472b6', '#db2777'], 'Children': ['🧸', '#f472b6', '#db2777'], 'Comedy': ['😂', '#eab308', '#ca8a04'],
  'Crime': ['🕵️', '#64748b', '#334155'], 'Documentary': ['🎥', '#14b8a6', '#0f766e'], 'Drama': ['🎭', '#8b5cf6', '#6d28d9'],
  'Fantasy': ['🧙', '#a855f7', '#7e22ce'], 'Film-Noir': ['🎞️', '#475569', '#1e293b'], 'Horror': ['👻', '#b91c1c', '#450a0a'],
  'Musical': ['🎵', '#06b6d4', '#0e7490'], 'Mystery': ['🔍', '#6366f1', '#4338ca'], 'Romance': ['💘', '#fb7185', '#e11d48'],
  'Sci-Fi': ['🚀', '#3b82f6', '#1d4ed8'], 'Thriller': ['🔪', '#dc2626', '#7f1d1d'], 'War': ['🪖', '#78716c', '#44403c'],
  'Western': ['🤠', '#d97706', '#92400e'], 'IMAX': ['📽️', '#0ea5e9', '#0369a1'],
};
const DEFAULT = ['🎬', '#8b8fa8', '#4b5069'];
const styleOf = m => GENRE[m.genres.find(g => GENRE[g])] || DEFAULT;

const liked = new Map(), disliked = new Map();
let postersOn = false, me = null;

// ---------- poster (tuỳ chọn, cần TMDB_API_KEY) ----------
const posterCache = {}, queue = [];
let running = 0;
function queuePoster(box, m) { if (postersOn) { queue.push([box, m]); pump(); } }
function pump() {
  while (running < 3 && queue.length) {
    const [box, m] = queue.shift();
    running++;
    fetchPoster(m).then(url => { if (url) box.append(h('img', { src: url, alt: m.title, loading: 'lazy' })); })
      .finally(() => { running--; pump(); });
  }
}
async function fetchPoster(m) {
  if (!(m.id in posterCache)) {
    try { posterCache[m.id] = (await api('/api/poster?id=' + m.id)).url; } catch { posterCache[m.id] = null; }
  }
  return posterCache[m.id];
}

// ---------- thành phần giao diện ----------
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => t.hidden = true, 2200);
}
function card(m) {
  const [emoji, c1, c2] = styleOf(m);
  const poster = h('div', { class: 'poster', style: `background:linear-gradient(160deg,${c1},${c2})` }, h('span', { class: 'em' }, emoji));
  queuePoster(poster, m);
  return h('div', { class: 'card', onclick: () => openModal(m) }, poster,
    h('div', { class: 'meta' },
      h('div', { class: 't' }, m.title),
      h('div', { class: 'y' }, [m.year, m.genres.slice(0, 2).join(' · ')].filter(Boolean).join(' · ')),
      m.badge ? h('div', { class: 'badge' }, m.badge) : null,
      m.note ? h('div', { class: 'note' }, m.note) : null));
}
const scrollRow = (title, items) => h('div', { class: 'row' }, h('h3', {}, title), h('div', { class: 'scroll' }, items.map(card)));
const gridOf = items => h('div', { class: 'grid' }, items.map(card));

function makeSearch(input, box, onPick) {
  let t;
  input.addEventListener('input', () => {
    clearTimeout(t);
    t = setTimeout(async () => {
      const q = input.value.trim();
      if (q.length < 2) { box.replaceChildren(); box.hidden = true; return; }
      let res = [];
      try { res = await api('/api/movies?q=' + encodeURIComponent(q)); } catch { }
      box.replaceChildren(...res.map(m => h('div', {
        class: 'sug', onclick: () => { onPick(m); input.value = ''; box.hidden = true; }
      }, h('b', {}, m.title), ' ', h('span', { class: 'dim' }, [m.year, m.genres.slice(0, 2).join(', ')].filter(Boolean).join(' · ')))));
      box.hidden = !res.length;
    }, 200);
  });
  document.addEventListener('click', e => { if (!box.contains(e.target) && e.target !== input) box.hidden = true; });
}

// ---------- chi tiết phim ----------
function closeModal() { document.querySelector('.modal-bg')?.remove(); }
async function openModal(m) {
  closeModal();
  const simBox = h('div', {}, h('p', { class: 'dim' }, 'Đang tìm phim tương tự...'));
  const btnLike = h('button', { class: 'btn' + (liked.has(m.id) ? ' on' : ''), onclick: () => { setPref(m, 'like'); closeModal(); } }, '👍 Thích');
  const btnDis = h('button', { class: 'btn ghost' + (disliked.has(m.id) ? ' on' : ''), onclick: () => { setPref(m, 'dislike'); closeModal(); } }, '👎 Không thích');
  const bg = h('div', { class: 'modal-bg', onclick: e => { if (e.target === bg) closeModal(); } },
    h('div', { class: 'modal' },
      h('button', { class: 'x', onclick: closeModal, 'aria-label': 'Đóng' }, '×'),
      h('h2', {}, m.title), h('div', { class: 'dim' }, [m.year, `${m.n} lượt đánh giá`, m.avg ? `điểm TB ${m.avg}` : ''].filter(Boolean).join(' · ')),
      h('div', { class: 'tags' }, m.genres.map(g => h('span', {}, g))),
      h('div', { class: 'actions' }, btnLike, btnDis),
      h('h3', {}, 'Nếu bạn thích phim này, có thể bạn cũng thích'), simBox));
  document.body.append(bg);
  try { simBox.replaceChildren(h('div', { class: 'scroll' }, (await api('/api/similar?id=' + m.id)).map(card))); }
  catch (e) { simBox.replaceChildren(h('p', { class: 'err' }, e.message)); }
}

// ---------- gu của bạn (lưu theo tài khoản) ----------
let saveT;
function savePrefs() {
  clearTimeout(saveT);
  saveT = setTimeout(() => api('/api/prefs', {
    method: 'PUT', headers: JSON_H, body: JSON.stringify({ liked: [...liked.keys()], disliked: [...disliked.keys()] })
  }).catch(() => toast('Không lưu được gu của bạn')), 400);
}
function setPref(m, kind) {
  liked.delete(m.id); disliked.delete(m.id);
  (kind === 'like' ? liked : disliked).set(m.id, m);
  renderChips(); savePrefs();
  toast(kind === 'like' ? `Đã thêm “${m.title}” vào phim bạn thích` : `Đã thêm “${m.title}” vào phim không thích`);
}
function renderChips() {
  $('#likeCount').textContent = liked.size;
  const mk = map => [...map.values()].map(m => h('span', { class: 'chip', title: 'Bấm để bỏ', onclick: () => { map.delete(m.id); renderChips(); savePrefs(); } }, m.title + ' ×'));
  $('#likedChips').replaceChildren(...mk(liked));
  $('#dislikedChips').replaceChildren(...mk(disliked));
}
$('#goProfile').addEventListener('click', async () => {
  const out = $('#profileOut');
  out.replaceChildren(h('p', { class: 'dim' }, 'Đang tính gợi ý...'));
  try {
    const recs = await api('/api/profile', {
      method: 'POST', headers: JSON_H, body: JSON.stringify({ liked: [...liked.keys()], disliked: [...disliked.keys()] })
    });
    out.replaceChildren(h('h3', {}, 'Dành cho bạn'), gridOf(recs));
  } catch (e) { out.replaceChildren(h('p', { class: 'err' }, e.message)); }
});

// ---------- thử nghiệm theo user_id (admin) ----------
$('#goUser').addEventListener('click', async () => {
  const out = $('#userOut');
  out.replaceChildren(h('p', { class: 'dim' }, 'Đang tính (User-based K lớn có thể mất vài giây)...'));
  try {
    const r = await api(`/api/user?uid=${encodeURIComponent($('#uid').value)}&mode=${$('#mode').value}&k=${$('#kk').value}`);
    out.replaceChildren(h('h3', {}, 'Phim được gợi ý'), gridOf(r.recs), h('h3', {}, 'Phim người này đã chấm cao (để đối chiếu)'), gridOf(r.history));
  } catch (e) { out.replaceChildren(h('p', { class: 'err' }, e.message)); }
});

// ---------- quản trị ----------
async function loadAdmin() {
  try {
    const [s, users] = await Promise.all([api('/api/admin/stats'), api('/api/admin/users')]);
    const items = [['📂 Bộ dữ liệu', s.dataset], ['🎞️ Phim', s.movies.toLocaleString()], ['👥 Người dùng MovieLens', s.users.toLocaleString()],
      ['⭐ Lượt đánh giá', s.ratings.toLocaleString()], ['🕳️ Độ thưa', (s.sparsity * 100).toFixed(2) + '%'],
      ['🧠 K / shrinkage', `${s.k} / ${s.shrink}`], ['💾 RAM ước tính', s.est_mem_gb + ' GB'], ['🙋 Tài khoản đăng ký', s.accounts]];
    $('#stats').replaceChildren(...items.map(([l, v]) => h('div', { class: 'stat' }, h('div', { class: 'sv' }, String(v)), h('div', { class: 'sl' }, l))));
    const body = $('#usersTbl tbody');
    if (!users.length) { body.replaceChildren(h('tr', {}, h('td', { colspan: '4', class: 'dim' }, 'Chưa có tài khoản nào.'))); return; }
    body.replaceChildren(...users.map(u => h('tr', {}, h('td', {}, u.username), h('td', {}, u.created), h('td', {}, String(u.n_liked)),
      h('td', {}, h('button', {
        class: 'del', onclick: async () => {
          if (!confirm(`Xoá tài khoản “${u.username}”?`)) return;
          try { await api('/api/admin/users/' + encodeURIComponent(u.username), { method: 'DELETE' }); toast('Đã xoá tài khoản'); loadAdmin(); }
          catch (e) { toast(e.message); }
        }
      }, 'Xoá')))));
  } catch (e) { $('#stats').replaceChildren(h('p', { class: 'err' }, e.message)); }
}

// ---------- menu bật/tắt, điều hướng, giao diện sáng/tối ----------
function setMenu(open) {
  $('#drawer').classList.toggle('open', open);
  $('#drawer').setAttribute('aria-hidden', String(!open));
  $('#overlay').hidden = !open;
  $('#menuBtn').setAttribute('aria-expanded', String(open));
}
$('#menuBtn').addEventListener('click', () => setMenu(!$('#drawer').classList.contains('open')));
$('#overlay').addEventListener('click', () => setMenu(false));
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeModal(); setMenu(false); } });

function showView(v) {
  $$('.side button[data-view]').forEach(x => x.classList.toggle('on', x.dataset.view === v));
  $$('.view').forEach(x => x.hidden = x.id !== 'view-' + v);
  if (v === 'admin') loadAdmin();
  setMenu(false);
}
$$('.side button[data-view]').forEach(b => b.addEventListener('click', () => showView(b.dataset.view)));

function markTheme() { $$('#themeSeg button').forEach(b => b.classList.toggle('on', b.dataset.t === Theme.get())); }
$$('#themeSeg button').forEach(b => b.addEventListener('click', () => { Theme.set(b.dataset.t); markTheme(); }));
markTheme();

$('#logout').addEventListener('click', async () => {
  try { await fetch('/api/auth/logout', { method: 'POST', headers: JSON_H, body: '{}' }); } catch { }
  window.location.href = '/login';
});
makeSearch($('#q'), $('#sug'), openModal);
makeSearch($('#q2'), $('#sug2'), m => setPref(m, 'like'));

// ---------- khởi động ----------
(async () => {
  try { me = await api('/api/auth/me'); } catch { return; }
  const admin = me.role === 'admin';
  $('#whoName').textContent = me.username;
  $('#whoRole').textContent = admin ? 'Quản trị viên' : 'Người dùng';
  $('#roleChip').textContent = admin ? '🛡️ Quản trị viên' : '👤 ' + me.username;
  $('#avatar').textContent = me.username.charAt(0).toUpperCase();
  $$('[data-admin]').forEach(b => b.hidden = !admin);
  try {
    const info = await api('/api/info');
    postersOn = info.posters; $('#tmdb').hidden = !info.posters;
    $('#uid').min = info.user_id_min; $('#uid').max = info.user_id_max; $('#uid').value = info.user_id_min;
  } catch { }
  try {
    const rows = await api('/api/rows');
    $('#rows').replaceChildren(...rows.map(r => scrollRow(r.title, r.items)));
  } catch (e) { $('#rows').replaceChildren(h('p', { class: 'err' }, 'Không tải được dữ liệu: ' + e.message)); }
  try {
    const p = await api('/api/prefs');
    p.liked.forEach(m => liked.set(m.id, m)); p.disliked.forEach(m => disliked.set(m.id, m)); renderChips();
  } catch { }
})();
