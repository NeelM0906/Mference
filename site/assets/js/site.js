/* Mference site behaviour. No dependencies; every block runs only when its markup is on the page. */
(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* nav */
  var top = $(".top");
  if (top) {
    var onScroll = function () { top.classList.toggle("scrolled", window.scrollY > 8); };
    onScroll(); window.addEventListener("scroll", onScroll, { passive: true });
    var btn = $(".menu-btn"), links = $(".links");
    if (btn && links) btn.addEventListener("click", function () {
      var open = links.classList.toggle("open"); btn.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  /* reveal on scroll */
  var rv = $$(".reveal");
  if (rv.length && "IntersectionObserver" in window && !reduce) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("seen"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px" });
    rv.forEach(function (el) { io.observe(el); });
  } else rv.forEach(function (el) { el.classList.add("seen"); });

  /* Find: a small sample index, matched by words and by meaning-ish synonyms */
  var INDEX = [
    { k: "PDF", n: "Lease 2026, signed.pdf", m: "Clause 14: one cat or dog under 25 lb", w: "lease clause pets pet cat dog apartment rent landlord" },
    { k: "Photo", n: "IMG_2041.heic", m: "Network: Fernway-5G, password on the router label", w: "wifi wi-fi password router network screenshot internet" },
    { k: "Audio", n: "Voice memo, Tuesday", m: "0:42 “the plumber can come Thursday”", w: "plumber leak sink voice memo audio thursday repair" },
    { k: "Photo", n: "Kenya trip 031.jpg", m: "A zebra at the water hole, sunset", w: "zebra animal safari kenya photo sunset trip" },
    { k: "Video", n: "Birthday 2025.mov", m: "1:18 blowing out the candles", w: "birthday cake candles video party" },
    { k: "Doc", n: "Board deck Q3.key", m: "Slide 9: hiring plan for Q4", w: "board deck hiring plan q3 q4 slides headcount" },
    { k: "PDF", n: "Passport scan.pdf", m: "Expires 14 March 2031", w: "passport expiry expires travel id document" }
  ];
  function search(q) {
    var words = q.toLowerCase().split(/[^a-z0-9-]+/).filter(function (w) { return w.length > 1; });
    if (!words.length) return [];
    return INDEX.map(function (it) {
      var s = 0; words.forEach(function (w) { if (it.w.indexOf(w) >= 0) s += 2; else if (it.n.toLowerCase().indexOf(w) >= 0) s += 1; });
      return { it: it, s: s };
    }).filter(function (x) { return x.s > 0; }).sort(function (a, b) { return b.s - a.s; }).slice(0, 3).map(function (x) { return x.it; });
  }
  function hitHTML(it) {
    return '<span class="k">' + it.k + '</span><b>' + it.n + '</b><em>' + it.m + '</em>';
  }

  $$("[data-finder]").forEach(function (box) {
    var input = $("input", box), out = $(".hits", box);
    function run() {
      var r = search(input.value);
      out.innerHTML = "";
      if (!r.length) { out.innerHTML = '<div class="hit"><span class="k">Tip</span><b>Try a phrase from the chips below</b></div>'; return; }
      r.forEach(function (it) { var d = document.createElement("div"); d.className = "hit"; d.innerHTML = hitHTML(it); out.appendChild(d); });
    }
    input.addEventListener("input", run);
    $$(".try button", box).forEach(function (b) { b.addEventListener("click", function () { input.value = b.textContent; run(); input.focus(); }); });
    run();
  });

  /* animated Find screen inside the product tour */
  function playFind(root, token) {
    var q = "that screenshot with the wifi password", typed = $(".q", root), hits = $(".hits", root), i = 0;
    typed.textContent = ""; hits.innerHTML = "";
    var r = search("wifi password screenshot").concat(search("lease pets").slice(0, 1));
    function addHits() {
      r.forEach(function (it, n) {
        var d = document.createElement("div"); d.className = "hit"; d.innerHTML = hitHTML(it); hits.appendChild(d);
        setTimeout(function () { if (token()) d.classList.add("in"); }, reduce ? 0 : 160 * n);
      });
    }
    if (reduce) { typed.textContent = q; addHits(); return; }
    (function step() {
      if (!token()) return;
      if (i <= q.length) { typed.textContent = q.slice(0, i++); setTimeout(step, 38); }
      else addHits();
    })();
  }
  function playNotes(root, token) {
    var lines = $$(".line", root), sum = $(".summary", root);
    lines.forEach(function (l) { l.classList.remove("in"); }); sum.classList.remove("in");
    if (reduce) { lines.forEach(function (l) { l.classList.add("in"); }); sum.classList.add("in"); return; }
    lines.forEach(function (l, n) { setTimeout(function () { if (token()) l.classList.add("in"); }, 300 + n * 900); });
    setTimeout(function () { if (token()) sum.classList.add("in"); }, 300 + lines.length * 900 + 300);
  }

  /* product tour: tabs that cycle on their own until someone picks one */
  $$("[data-tour]").forEach(function (tour) {
    var tabs = $$('[role="tab"]', tour), panels = $$('[role="tabpanel"]', tour), cap = $(".caption", tour);
    var cur = 0, timer = null, auto = !reduce, gen = 0, DWELL = 7000;
    tour.style.setProperty("--dwell", DWELL + "ms");
    function show(i, focus) {
      cur = i; var g = ++gen, alive = function () { return g === gen; };
      tabs.forEach(function (t, n) {
        var on = n === i; t.setAttribute("aria-selected", on ? "true" : "false"); t.tabIndex = on ? 0 : -1;
        t.classList.remove("timing"); if (on && auto) { void t.offsetWidth; t.classList.add("timing"); }
      });
      panels.forEach(function (p, n) { p.classList.toggle("on", n === i); p.hidden = false; p.setAttribute("aria-hidden", n === i ? "false" : "true"); });
      if (cap) cap.textContent = tabs[i].getAttribute("data-caption") || "";
      var kind = panels[i].getAttribute("data-play");
      if (kind === "find") playFind(panels[i], alive);
      if (kind === "notes") playNotes(panels[i], alive);
      if (focus) tabs[i].focus();
      clearTimeout(timer);
      if (auto) timer = setTimeout(function () { show((cur + 1) % tabs.length); }, DWELL);
    }
    function stop() { auto = false; clearTimeout(timer); tabs.forEach(function (t) { t.classList.remove("timing"); }); }
    tabs.forEach(function (t, n) {
      t.addEventListener("click", function () { stop(); show(n); });
      t.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0; if (!d) return;
        e.preventDefault(); stop(); show((cur + d + tabs.length) % tabs.length, true);
      });
    });
    /* only cycle while the tour is on screen */
    if ("IntersectionObserver" in window && auto) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (!auto) return; if (e.isIntersecting) show(cur); else clearTimeout(timer); });
      }, { threshold: 0.35 }).observe(tour);
    }
    show(0);
  });

  /* offline switch: flips every data-flow="out" row in the same scope; "local"
     rows stay on the Mac and "system" rows (Apple Mail and Calendar sync) are
     not paused by it */
  $$("[data-offline]").forEach(function (scope) {
    var sw = $(".switch", scope), sub = $("[data-offline-sub]", scope);
    function set(on) {
      sw.setAttribute("aria-checked", on ? "true" : "false");
      if (sub) sub.textContent = on ? sub.getAttribute("data-on") : sub.getAttribute("data-off");
      $$("[data-flow]", scope).forEach(function (row) {
        if (row.getAttribute("data-flow") !== "out") return;
        var tag = $(".tag", row), wire = $(".wire", row);
        tag.textContent = on ? (row.getAttribute("data-off-label") || "Off") : row.getAttribute("data-on-label");
        tag.className = "tag " + (on ? "off" : "ask");
        if (wire) wire.className = "wire" + (on ? " cut" : "");
        row.classList.toggle("cut", on);
      });
    }
    sw.addEventListener("click", function () { set(sw.getAttribute("aria-checked") !== "true"); });
    set(false);
  });

  /* memory picker (data.js) */
  var M = window.MF;
  $$("[data-models]").forEach(function (root) {
    if (!M) return;
    var seg = $(".seg", root), grid = $(".model-grid", root), body = $("tbody", root), verdict = $(".verdict", root);
    var cur = M.pickHost();
    M.HOSTS.forEach(function (h) {
      var b = document.createElement("button"); b.type = "button"; b.textContent = h.label; b.title = h.mac; b.dataset.id = h.id;
      b.addEventListener("click", function () { cur = h; M.setHash(h.id); update(); });
      seg.appendChild(b);
    });
    var fams = M.FAMS.slice();
    function update() {
      $$("button", seg).forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.id === cur.id ? "true" : "false"); });
      var runs = 0;
      var rows = fams.map(function (f) {
        var need = M.status(f, cur) === "need";
        if (!need) runs++;
        return { f: f, need: need };
      });
      var badgeFor = function (r) { return r.need ? '<span class="badge">Needs a bigger Mac</span>' : '<span class="badge ok">Runs</span>'; };
      if (grid) grid.innerHTML = rows.map(function (r) {
        return '<div class="mcard' + (r.need ? " need" : "") + '"><div class="row"><h3>' + r.f.short + '</h3>' + badgeFor(r) + '</div><p class="sub">' + r.f.blurb + '</p>' +
          '<div class="nums"><div><b>' + M.fmtGB(r.f.disk) + ' GB</b><span>download</span></div><div><b>' + r.f.p + 'B</b><span>parameters</span></div></div></div>';
      }).join("");
      if (body) body.innerHTML = rows.map(function (r) {
        return '<tr class="' + (r.need ? "need" : "") + '"><td><strong>' + r.f.short + '</strong><small>' + r.f.blurb + '</small></td><td>' + (r.f.maker || "\u2014") + '</td><td>' + r.f.p + 'B</td><td>' +
          M.fmtGB(r.f.disk) + ' GB</td><td>' + (r.f.install === "app" ? "In the app" : '<a href="developers.html#models">Terminal</a>') + '</td><td>' + badgeFor(r) + '</td></tr>';
      }).join("");
      if (verdict) verdict.innerHTML = '<b>' + runs + '</b> of ' + fams.length + ' models run on a ' + cur.label + ' Mac.';
    }
    update();
  });

  /* copy buttons */
  $$("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var pre = document.getElementById(b.getAttribute("data-copy"));
      var text = pre.innerText.split("\n").filter(function (l) { return l.trim() && l.trim()[0] !== "#"; }).join("\n");
      (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(function () {
        b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1600);
      }, function () { b.textContent = "Select and copy"; });
    });
  });

  /* features sub-nav follows the section on screen */
  var sub = $$(".subnav a");
  if (sub.length && "IntersectionObserver" in window) {
    var map = {}; sub.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var so = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (!e.isIntersecting) return;
        sub.forEach(function (a) { a.classList.remove("on"); a.removeAttribute("aria-current"); });
        var a = map[e.target.id];
        if (a) { a.classList.add("on"); a.setAttribute("aria-current", "true"); a.scrollIntoView({ block: "nearest", inline: "nearest" }); }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(map).forEach(function (id) { var el = document.getElementById(id); if (el) so.observe(el); });
  }
})();
