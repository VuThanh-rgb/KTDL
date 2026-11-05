const $ = s => document.querySelector(s);
const st = { role: 'user', mode: 'login' };

function render() {
  const user = st.role === 'user', reg = user && st.mode === 'register';
  document.querySelectorAll('.roletabs button').forEach(b => b.classList.toggle('on', b.dataset.role === st.role));
  document.querySelectorAll('#modeSeg button').forEach(b => b.classList.toggle('on', b.dataset.mode === st.mode));
  $('#modeSeg').hidden = !user;
  $('#confirmWrap').hidden = !reg;
  $('#submit').textContent = reg ? 'Tạo tài khoản' : (user ? 'Đăng nhập' : 'Đăng nhập quản trị');
  $('#password').autocomplete = reg ? 'new-password' : 'current-password';
  $('#hint').textContent = user
    ? (reg ? 'Mật khẩu từ 6 ký tự. Gu phim của bạn sẽ được lưu theo tài khoản.' : 'Chưa có tài khoản? Chọn "Tạo tài khoản".')
    : 'Tài khoản quản trị do người cài đặt hệ thống cấu hình.';
  $('#msg').hidden = true;
}
function showMsg(t) { const m = $('#msg'); m.textContent = t; m.hidden = false; }

document.querySelectorAll('.roletabs button').forEach(b => b.addEventListener('click', () => { st.role = b.dataset.role; render(); }));
document.querySelectorAll('#modeSeg button').forEach(b => b.addEventListener('click', () => { st.mode = b.dataset.mode; render(); }));
document.querySelectorAll('#themeSeg button').forEach(b => b.addEventListener('click', () => { Theme.set(b.dataset.t); markTheme(); }));
function markTheme() { document.querySelectorAll('#themeSeg button').forEach(b => b.classList.toggle('on', b.dataset.t === Theme.get())); }

$('#form').addEventListener('submit', async e => {
  e.preventDefault();
  const username = $('#username').value.trim(), password = $('#password').value;
  const reg = st.role === 'user' && st.mode === 'register';
  if (!username || !password) return showMsg('Vui lòng nhập tên đăng nhập và mật khẩu');
  if (reg && password !== $('#confirm').value) return showMsg('Hai mật khẩu không khớp');
  const btn = $('#submit'); btn.disabled = true;
  try {
    const r = await fetch(reg ? '/api/auth/register' : '/api/auth/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role: st.role, username, password })
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) showMsg(body.error || 'Có lỗi xảy ra');
    else window.location.href = '/';
  } catch { showMsg('Không kết nối được tới máy chủ'); }
  btn.disabled = false;
});
markTheme(); render();
