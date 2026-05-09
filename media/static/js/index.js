/* AgentArk project page — interactive bits
   - sticky-nav active section highlighting
   - mobile nav toggle
   - BibTeX copy-to-clipboard
   - reveal-on-scroll
   Vanilla JS, no jQuery dependency required. (jQuery is still on the page
   for the legacy Bulma carousel/slider plugins.)
*/

(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    initMobileNavToggle();
    initActiveSectionHighlight();
    initRevealOnScroll();
    initBibtexCopy();
    initLegacyBulma();
  });

  /* ---------- Mobile nav toggle ---------- */
  function initMobileNavToggle() {
    var btn = document.querySelector(".topnav .nav-toggle");
    var links = document.querySelector(".topnav .links");
    if (!btn || !links) return;

    btn.addEventListener("click", function () {
      var open = links.classList.toggle("is-open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });

    links.querySelectorAll("a").forEach(function (a) {
      a.addEventListener("click", function () {
        if (links.classList.contains("is-open")) {
          links.classList.remove("is-open");
          btn.setAttribute("aria-expanded", "false");
        }
      });
    });
  }

  /* ---------- Sticky-nav active section highlight ---------- */
  function initActiveSectionHighlight() {
    var navLinks = Array.prototype.slice.call(
      document.querySelectorAll(".topnav .links a[href^='#']")
    );
    if (!navLinks.length || !("IntersectionObserver" in window)) return;

    var sections = navLinks
      .map(function (a) {
        var id = a.getAttribute("href").slice(1);
        var el = id ? document.getElementById(id) : null;
        return el ? { id: id, el: el, link: a } : null;
      })
      .filter(Boolean);

    if (!sections.length) return;

    var visibility = {};
    sections.forEach(function (s) { visibility[s.id] = 0; });

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          visibility[entry.target.id] = entry.intersectionRatio;
        });
        var bestId = null;
        var bestRatio = 0;
        Object.keys(visibility).forEach(function (id) {
          if (visibility[id] > bestRatio) {
            bestRatio = visibility[id];
            bestId = id;
          }
        });
        navLinks.forEach(function (link) {
          var match = link.getAttribute("href") === "#" + bestId && bestRatio > 0;
          link.classList.toggle("is-active", !!match);
        });
      },
      {
        rootMargin: "-30% 0px -55% 0px",
        threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
      }
    );

    sections.forEach(function (s) { observer.observe(s.el); });
  }

  /* ---------- Reveal on scroll ---------- */
  function initRevealOnScroll() {
    var nodes = document.querySelectorAll(".reveal");
    if (!nodes.length) return;

    if (!("IntersectionObserver" in window)) {
      nodes.forEach(function (n) { n.classList.add("in-view"); });
      return;
    }

    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            io.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
    );

    nodes.forEach(function (n) { io.observe(n); });
  }

  /* ---------- BibTeX copy ---------- */
  function initBibtexCopy() {
    var btns = document.querySelectorAll(".copy-bib");
    btns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        var sel = btn.getAttribute("data-target");
        var target = sel ? document.querySelector(sel) : null;
        if (!target) return;

        var text = target.innerText.trim();
        copyText(text).then(function (ok) {
          if (!ok) return;
          var label = btn.querySelector(".copy-label");
          var icon = btn.querySelector("i");
          var prevLabel = label ? label.textContent : "";
          var prevIcon = icon ? icon.className : "";

          btn.classList.add("is-copied");
          if (label) label.textContent = "Copied";
          if (icon) icon.className = "fas fa-check";

          setTimeout(function () {
            btn.classList.remove("is-copied");
            if (label) label.textContent = prevLabel || "Copy";
            if (icon) icon.className = prevIcon || "far fa-copy";
          }, 1800);
        });
      });
    });
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).then(
        function () { return true; },
        function () { return fallbackCopy(text); }
      );
    }
    return Promise.resolve(fallbackCopy(text));
  }

  function fallbackCopy(text) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (e) {
      return false;
    }
  }

  /* ---------- Legacy Bulma carousel/slider init (kept for safety) ---------- */
  function initLegacyBulma() {
    if (typeof window.bulmaCarousel !== "undefined") {
      try {
        window.bulmaCarousel.attach(".carousel", {
          slidesToScroll: 1,
          slidesToShow: 3,
          loop: true,
          infinite: true,
          autoplay: false,
        });
      } catch (e) { /* no-op */ }
    }
    if (typeof window.bulmaSlider !== "undefined") {
      try { window.bulmaSlider.attach(); } catch (e) { /* no-op */ }
    }
  }
})();
