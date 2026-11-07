// Play each video only while it is on screen, and mark the current page in the nav.
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
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      if (e.isIntersecting) {
        if (v.dataset.src && !v.getAttribute("src")) v.src = v.dataset.src;
        const p = v.play();
        if (p && p.catch) p.catch(() => {});
      } else {
        v.pause();
      }
    }
  }, { threshold: 0.25 });
  document.querySelectorAll("video").forEach((v) => {
    v.muted = true; v.loop = true; v.playsInline = true;
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
