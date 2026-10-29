/* Optional enhancements only. All content, navigation and video work without JS. */
(function () {
  "use strict";

  function initActiveSectionHighlight() {
    const nav = document.querySelector(".topnav");
    const sections = Array.from(document.querySelectorAll(".links a[href^='#'], .mobile-links a[href^='#']"))
      .map(link => ({ link, section: document.getElementById(link.hash.slice(1)) }))
      .filter(item => item.section);
    let scheduled = false;

    function updateActiveSection() {
      const threshold = nav.getBoundingClientRect().height + 80;
      let active = null;
      for (const item of sections) {
        if (item.section.getBoundingClientRect().top <= threshold) active = item.section.id;
      }
      for (const item of sections) {
        if (item.section.id === active) item.link.setAttribute("aria-current", "location");
        else item.link.removeAttribute("aria-current");
      }
      scheduled = false;
    }

    function scheduleUpdate() {
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(updateActiveSection);
    }

    if (!nav || !sections.length) return;
    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    updateActiveSection();
  }

  function fallbackCopy(text) {
    const previousFocus = document.activeElement;
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.className = "sr-only";
    document.body.appendChild(textarea);
    textarea.focus({ preventScroll: true });
    textarea.select();
    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (error) {
      copied = false;
    } finally {
      textarea.remove();
      if (previousFocus) previousFocus.focus({ preventScroll: true });
    }
    return copied;
  }

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (error) {
        // Local files and denied clipboard permissions use the same fallback.
      }
    }
    return fallbackCopy(text);
  }

  function initBibtexCopy() {
    document.querySelectorAll(".copy-bib").forEach(button => {
      const target = document.querySelector(button.dataset.target);
      const label = button.querySelector(".copy-label");
      const status = button.closest(".bibtex-wrap").querySelector(".copy-status");
      if (!target || !label) return;
      let resetTimer;
      button.addEventListener("click", async () => {
        button.disabled = true;
        const copied = await copyText(target.textContent.trim());
        label.textContent = copied ? "Copied" : "Select text";
        button.classList.toggle("is-copied", copied);
        if (status) status.textContent = copied ? "BibTeX copied to clipboard." : "Select and copy the citation text below.";
        if (!copied) {
          const range = document.createRange();
          range.selectNodeContents(target);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
        }
        button.disabled = false;
        clearTimeout(resetTimer);
        resetTimer = setTimeout(() => {
          label.textContent = "Copy";
          button.classList.remove("is-copied");
          if (status) status.textContent = "";
        }, 2400);
      });
      button.hidden = false;
    });
  }

  initBibtexCopy();
  initActiveSectionHighlight();
  document.querySelectorAll(".mobile-links a").forEach(link => {
    link.addEventListener("click", () => { link.closest("details").open = false; });
  });
})();
