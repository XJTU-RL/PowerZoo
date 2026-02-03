// PowerZoo GitHub Pages - Navigation & Interactions

(function () {
  "use strict";

  // Hamburger menu toggle
  var hamburger = document.getElementById("hamburger");
  var navLinks = document.getElementById("nav-links");

  if (hamburger && navLinks) {
    hamburger.addEventListener("click", function () {
      navLinks.classList.toggle("open");
    });

    // Close menu on link click (mobile)
    navLinks.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        navLinks.classList.remove("open");
      });
    });
  }

  // Active nav link highlighting via IntersectionObserver
  var sections = document.querySelectorAll("section[id]");
  var navAnchors = document.querySelectorAll(".nav-links a");

  if (sections.length && navAnchors.length) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            var id = entry.target.getAttribute("id");
            navAnchors.forEach(function (a) {
              a.classList.toggle("active", a.getAttribute("href") === "#" + id);
            });
          }
        });
      },
      { rootMargin: "-30% 0px -70% 0px" }
    );

    sections.forEach(function (s) {
      observer.observe(s);
    });
  }

  // Diagram tab switching
  var diagramTabs = document.querySelectorAll(".diagram-tab");
  if (diagramTabs.length) {
    diagramTabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var target = this.getAttribute("data-target");
        // Deactivate all tabs and hide all content
        diagramTabs.forEach(function (t) { t.classList.remove("active"); });
        document.querySelectorAll(".diagram-content").forEach(function (c) {
          c.style.display = "none";
        });
        // Activate clicked tab and show content
        this.classList.add("active");
        var el = document.getElementById(target);
        if (el) el.style.display = "block";
      });
    });
  }

  // Navbar background on scroll
  var navbar = document.getElementById("navbar");
  if (navbar) {
    window.addEventListener("scroll", function () {
      if (window.scrollY > 20) {
        navbar.style.boxShadow = "0 2px 12px rgba(0,0,0,0.1)";
      } else {
        navbar.style.boxShadow = "none";
      }
    });
  }
})();
