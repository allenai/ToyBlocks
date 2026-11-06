// Play each video only while it is on screen, and mark the current page in the nav.
(function () {
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
})();
