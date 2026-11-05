/* Chế độ giao diện: 'light' | 'dark' | 'system' (theo hệ điều hành). Chạy sớm trong <head> để không bị nháy màu. */
(function () {
  var KEY = 'kmv-theme';
  var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : { matches: false };
  function get() { try { return localStorage.getItem(KEY) || 'system'; } catch (e) { return 'system'; } }
  function apply() {
    var p = get(), eff = p === 'system' ? (mq.matches ? 'dark' : 'light') : p;
    document.documentElement.setAttribute('data-theme', eff);
    document.documentElement.setAttribute('data-pref', p);
  }
  window.Theme = {
    get: get, apply: apply,
    set: function (p) { try { localStorage.setItem(KEY, p); } catch (e) { } apply(); }
  };
  if (mq.addEventListener) mq.addEventListener('change', function () { if (get() === 'system') apply(); });
  apply();
})();
