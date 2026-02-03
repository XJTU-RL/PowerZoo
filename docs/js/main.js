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

  // Diagram tab switching (scoped per tab group)
  var tabGroups = document.querySelectorAll(".diagram-tabs");
  tabGroups.forEach(function (group) {
    var tabs = group.querySelectorAll(".diagram-tab");
    // Collect all content panels that belong to this tab group
    var contentIds = [];
    tabs.forEach(function (t) { contentIds.push(t.getAttribute("data-target")); });

    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var target = this.getAttribute("data-target");
        // Deactivate only tabs in this group
        tabs.forEach(function (t) { t.classList.remove("active"); });
        // Hide only content panels belonging to this group
        contentIds.forEach(function (id) {
          var el = document.getElementById(id);
          if (el) el.style.display = "none";
        });
        // Activate clicked tab and show its content
        this.classList.add("active");
        var el = document.getElementById(target);
        if (el) el.style.display = "block";
      });
    });
  });

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
