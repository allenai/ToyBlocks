// Play each video only while it is on screen (with a pause/play button), and mark the current page in the nav.
(function () {
  // Open every page at the top. Browsers (and some embedding viewers) restore the previous
  // scroll offset on navigation, which dropped people mid-page; only a #section link scrolls.
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";
  let touched = false;  // never pull someone back up once they have started scrolling
  for (const ev of ["wheel", "touchstart", "keydown", "mousedown"]) {
    window.addEventListener(ev, () => { touched = true; }, { once: true, passive: true });
  }
  const toTop = () => { if (!location.hash && !touched) window.scrollTo(0, 0); };
  toTop();
  window.addEventListener("pageshow", toTop);
  window.addEventListener("load", toTop);
  // Videos autoplay (muted, looping) while on screen. Each gets a pause/play button; a video the
  // viewer paused stays paused when it scrolls back into view. Reduced-motion users start paused.
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ICON_PAUSE = '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="3.5" y="2.5" width="3" height="11" rx="1"/><rect x="9.5" y="2.5" width="3" height="11" rx="1"/></svg>';
  const ICON_PLAY = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 2.8v10.4a.6.6 0 0 0 .9.5l8.2-5.2a.6.6 0 0 0 0-1L5.4 2.3a.6.6 0 0 0-.9.5z"/></svg>';
  const load = (v) => { if (v.dataset.src && !v.getAttribute("src")) v.src = v.dataset.src; };
  const tryPlay = (v) => { load(v); const p = v.play(); if (p && p.catch) p.catch(() => {}); };
  const sync = (v) => {
    const b = v._btn; if (!b) return;
    const paused = v.paused;
    b.innerHTML = paused ? ICON_PLAY : ICON_PAUSE;
    b.setAttribute("aria-label", paused ? "Play video" : "Pause video");
    v.parentElement.classList.toggle("paused", paused);
  };
  const toggle = (v) => {
    if (v.paused) { v._userPaused = false; tryPlay(v); }
    else { v._userPaused = true; v.pause(); }
  };
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      if (e.isIntersecting) {
        load(v);
        if (!v._userPaused) tryPlay(v);
      } else {
        v.pause();
      }
    }
  }, { threshold: 0.25 });
  document.querySelectorAll("video").forEach((v) => {
    v.muted = true; v.loop = true; v.playsInline = true;
    v._userPaused = reduced;
    const wrap = document.createElement("span");
    wrap.className = "vwrap";
    v.parentNode.insertBefore(wrap, v);
    wrap.appendChild(v);
    const b = document.createElement("button");
    b.type = "button"; b.className = "vbtn";
    b.addEventListener("click", (ev) => { ev.preventDefault(); ev.stopPropagation(); toggle(v); });
    wrap.appendChild(b);
    v._btn = b;
    // Clicking the picture itself also toggles, except inside a link card (there it opens the page).
    if (!v.closest("a")) { v.style.cursor = "pointer"; v.addEventListener("click", () => toggle(v)); }
    for (const ev of ["play", "pause"]) v.addEventListener(ev, () => sync(v));
    sync(v);
    io.observe(v);
  });
  const here = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".nav .links a").forEach((a) => {
    if (a.getAttribute("href") === here) a.classList.add("on");
  });
  // On narrow screens the section links scroll sideways; bring the current page's link into view.
  const on = document.querySelector(".nav .links a.on");
  if (on) {
    const row = on.parentElement;
    if (row.scrollWidth > row.clientWidth) row.scrollLeft = Math.max(0, on.offsetLeft - row.offsetLeft - 20);
  }
})();
