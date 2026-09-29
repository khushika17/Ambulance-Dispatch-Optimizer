const API_BASE = "";

const NUMERIC_FIELDS = new Set([
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "latitude",
    "longitude",
    "severity_level",
    "available_ambulances",
    "hospitals_nearby",
    "temperature",
    "wind_speed",
    "traffic_level",
    "distance_to_hospital",
    "population_density",
]);

document.addEventListener("DOMContentLoaded", () => {
    setupTabs();
    setupPredictForm();
    setupRetrainForm();
    document.getElementById("refresh-insights").addEventListener("click", loadInsights);
});

function setupTabs() {
    const links = document.querySelectorAll(".nav-link");
    const views = document.querySelectorAll(".view");

    links.forEach((link) => {
        link.addEventListener("click", (event) => {
            event.preventDefault();
            const tab = link.dataset.tab;
            links.forEach((l) => l.classList.toggle("active", l === link));
            views.forEach((v) => v.classList.toggle("view-active", v.id === tab));
            if (tab === "insights") loadInsights();
        });
    });
}

function collectFormData(form) {
    const data = {};
    new FormData(form).forEach((value, key) => {
        data[key] = NUMERIC_FIELDS.has(key) ? Number(value) : value;
    });
    return data;
}

function setupPredictForm() {
    const form = document.getElementById("predict-form");
    const resetBtn = document.getElementById("reset-btn");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const submitBtn = form.querySelector("button[type='submit']");
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span>Predicting...';
        hideResultStates();

        try {
            const response = await fetch(`${API_BASE}/api/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(collectFormData(form)),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Prediction failed");
            }
            renderPrediction(data);
        } catch (err) {
            showError(err.message);
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = "Run Prediction";
        }
    });

    resetBtn.addEventListener("click", () => {
        form.reset();
        hideResultStates();
        document.getElementById("result-empty").hidden = false;
    });
}

function hideResultStates() {
    document.getElementById("result").hidden = true;
    document.getElementById("result-empty").hidden = true;
    document.getElementById("result-error").hidden = true;
}

function showError(message) {
    const node = document.getElementById("result-error");
    node.textContent = message;
    node.hidden = false;
}

function renderPrediction({ prediction, optimization_tips }) {
    document.getElementById("result").hidden = false;

    const verdict = document.getElementById("verdict");
    const badge = document.getElementById("verdict-badge");
    const title = document.getElementById("verdict-title");
    const sub = document.getElementById("verdict-sub");

    const optimal = prediction.optimal_dispatch === 1;
    verdict.classList.toggle("optimal", optimal);
    verdict.classList.toggle("suboptimal", !optimal);
    badge.textContent = optimal ? "✓" : "!";
    title.textContent = optimal ? "Optimal Dispatch" : "Suboptimal Dispatch";
    sub.textContent = prediction.recommendation;

    document.getElementById("confidence").textContent = formatPct(prediction.confidence);
    document.getElementById("prob-optimal").textContent = formatPct(prediction.probability_optimal);
    document.getElementById("prob-suboptimal").textContent = formatPct(prediction.probability_suboptimal);
    document.getElementById("probability-fill").style.width = `${prediction.probability_optimal * 100}%`;

    const tips = document.getElementById("tips");
    tips.innerHTML = "";
    optimization_tips.forEach((tip) => {
        const li = document.createElement("li");
        li.textContent = tip;
        tips.appendChild(li);
    });
}

function formatPct(value) {
    return `${(value * 100).toFixed(1)}%`;
}

async function loadInsights() {
    const insights = document.getElementById("insights-content");
    const grid = document.getElementById("images-grid");
    insights.innerHTML = '<p class="empty-state">Loading insights...</p>';
    grid.innerHTML = '<p class="empty-state">Loading visualizations...</p>';

    try {
        const response = await fetch(`${API_BASE}/api/model-info`);
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "Could not load model info");
        }
        const data = await response.json();
        renderInsights(insights, data);
        renderImages(grid, data.images || []);
    } catch (err) {
        insights.innerHTML = `<div class="error-state">${err.message}</div>`;
        grid.innerHTML = "";
    }
}

function renderInsights(container, data) {
    const rows = Object.entries(data.models)
        .map(([name, metrics]) => {
            const isBest = name === data.best_model;
            return `
                <tr class="${isBest ? "best" : ""}">
                    <td>${name}</td>
                    <td>${metrics.accuracy.toFixed(4)}</td>
                    <td>${metrics.precision.toFixed(4)}</td>
                    <td>${metrics.recall.toFixed(4)}</td>
                    <td>${metrics.f1.toFixed(4)}</td>
                    <td>${metrics.roc_auc.toFixed(4)}</td>
                </tr>
            `;
        })
        .join("");

    container.innerHTML = `
        <p>Best model: <strong>${data.best_model}</strong> · ${data.feature_names.length} features</p>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

function renderImages(grid, images) {
    if (!images.length) {
        grid.innerHTML = '<p class="empty-state">No visualizations available yet.</p>';
        return;
    }
    grid.innerHTML = images
        .map(
            (img) => `
        <div class="image-tile">
            <img src="${img.url}" alt="${img.name}" loading="lazy" />
            <div class="image-title">${img.name}</div>
        </div>
    `,
        )
        .join("");
}

function setupRetrainForm() {
    const form = document.getElementById("retrain-form");
    const status = document.getElementById("retrain-status");

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const btn = document.getElementById("retrain-btn");
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Training...';
        status.hidden = false;
        status.className = "status-block running";
        status.textContent = "Training the pipeline. This may take a minute...";

        const payload = { n_samples: Number(document.getElementById("n_samples").value) };

        try {
            const response = await fetch(`${API_BASE}/api/retrain`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Training failed");
            status.className = "status-block success";
            status.textContent = `Training complete with ${data.n_samples} samples. Switch to Model Insights to view results.`;
        } catch (err) {
            status.className = "status-block error";
            status.textContent = err.message;
        } finally {
            btn.disabled = false;
            btn.textContent = "Start Training";
        }
    });
}
