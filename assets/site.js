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
  const ICON_FS = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2 6V2.8c0-.4.4-.8.8-.8H6v1.6H3.6V6zm8-4h3.2c.4 0 .8.4.8.8V6h-1.6V3.6H10zm4 8v3.2c0 .4-.4.8-.8.8H10v-1.6h2.4V10zM6 14H2.8c-.4 0-.8-.4-.8-.8V10h1.6v2.4H6z"/></svg>';
  const ICON_FS_EXIT = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.4 2H6v3.2c0 .4-.4.8-.8.8H2V4.4h2.4zM10 2h1.6v2.4H14V6h-3.2c-.4 0-.8-.4-.8-.8zm4 8v1.6h-2.4V14H10v-3.2c0-.4.4-.8.8-.8zM6 14H4.4v-2.4H2V10h3.2c.4 0 .8.4.8.8z"/></svg>';
  const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement;
  // Full screen on the wrapper keeps the pause and full-screen buttons on top of the video.
  const fullscreen = (wrap, v) => {
    if (fsElement() === wrap) {
      (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      return;
    }
    const req = wrap.requestFullscreen || wrap.webkitRequestFullscreen;
    if (req) {
      const p = req.call(wrap);
      if (p && p.catch) p.catch(() => {});
    } else if (v.webkitEnterFullscreen) {
      load(v); v.webkitEnterFullscreen();
    }
    if (!v._userPaused) tryPlay(v);
  };
  const onFsChange = () => {
    document.querySelectorAll(".vwrap").forEach((w) => {
      const f = w.querySelector(".vfs"); if (!f) return;
      const on = fsElement() === w;
      w.classList.toggle("fs", on);
      f.innerHTML = on ? ICON_FS_EXIT : ICON_FS;
      f.setAttribute("aria-label", on ? "Exit full screen" : "Full screen");
    });
  };
  document.addEventListener("fullscreenchange", onFsChange);
  document.addEventListener("webkitfullscreenchange", onFsChange);
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
    // Full-screen button beside it (iOS Safari only has the video element's own full screen).
    const canFs = wrap.requestFullscreen || wrap.webkitRequestFullscreen || v.webkitEnterFullscreen;
    if (canFs) {
      const f = document.createElement("button");
      f.type = "button"; f.className = "vbtn vfs";
      f.innerHTML = ICON_FS; f.setAttribute("aria-label", "Full screen");
      f.addEventListener("click", (ev) => { ev.preventDefault(); ev.stopPropagation(); fullscreen(wrap, v); });
      wrap.appendChild(f);
    }
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
