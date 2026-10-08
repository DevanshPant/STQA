// Small progressive enhancements. Everything works without JS too.

document.querySelectorAll("[data-pw-toggle]").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var input = btn.parentElement.querySelector("input");
    var show = input.type === "password";
    input.type = show ? "text" : "password";
    btn.textContent = show ? "Hide" : "Show";
  });
});

// rough strength meter: length + mix of character types
document.querySelectorAll("input[data-strength]").forEach(function (input) {
  var meter = input.closest(".field").querySelector(".meter");
  if (!meter) return;
  input.addEventListener("input", function () {
    var v = input.value, score = 0;
    if (v.length >= 8) score++;
    if (v.length >= 12) score++;
    if (/[a-z]/.test(v) && /[A-Z]/.test(v)) score++;
    if (/\d/.test(v) && /[^A-Za-z0-9]/.test(v)) score++;
    meter.dataset.level = v ? Math.max(1, score) : "";
  });
});

document.querySelectorAll("form[data-confirm]").forEach(function (form) {
  form.addEventListener("submit", function (e) {
    if (!window.confirm(form.dataset.confirm)) e.preventDefault();
  });
});
