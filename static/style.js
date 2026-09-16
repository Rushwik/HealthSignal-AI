const healthForm = document.getElementById("healthForm");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const progressBar = document.getElementById("progressBar");
const loadingText = document.getElementById("loadingText");

healthForm.addEventListener("submit", function () {
    analyzeBtn.disabled = true;

    analyzeBtn.querySelector(".button-text").textContent = "Analyzing...";
    analyzeBtn.querySelector(".button-arrow").textContent = "⟳";

    loadingOverlay.classList.add("active");

    const messages = [
        "Preparing health-related indicators...",
        "Reviewing fictional case details...",
        "Identifying attention-worthy factors...",
        "Preparing explainable observations...",
        "Applying responsible AI safeguards...",
        "Finalizing your awareness report..."
    ];

    let progress = 0;
    let messageIndex = 0;

    const interval = setInterval(function () {
        progress += 16.6;

        if (progress > 100) {
            progress = 100;
        }

        progressBar.style.width = progress + "%";

        if (messageIndex < messages.length) {
            loadingText.textContent = messages[messageIndex];
            messageIndex++;
        }

        if (progress >= 100) {
            clearInterval(interval);
        }
    }, 450);
});