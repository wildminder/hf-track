/**
 * hf-track Web Demo — Frontend logic
 *
 * Connects to the FastAPI SSE backend, renders progress bars,
 * and handles download start/cancel interactions.
 */
"use strict";

// ── Formatting Utilities ─────────────────────────────────────────

/**
 * Format bytes into human-readable string.
 * Mirrors the Python format_bytes() in progress_bar.py.
 */
function formatBytes(n) {
    const units = ["B", "KB", "MB", "GB", "TB"];
    let val = n;
    for (const unit of units) {
        if (Math.abs(val) < 1024.0) {
            return unit === "B" ? `${val.toFixed(0)} ${unit}` : `${val.toFixed(1)} ${unit}`;
        }
        val /= 1024.0;
    }
    return `${val.toFixed(1)} PB`;
}

function formatSpeed(bytesPerSec) {
    return formatBytes(bytesPerSec) + "/s";
}

function formatEta(seconds) {
    if (seconds <= 0 || seconds === Infinity || isNaN(seconds)) return "--:--";
    const totalSecs = Math.floor(seconds);
    const hrs = Math.floor(totalSecs / 3600);
    const mins = Math.floor((totalSecs % 3600) / 60);
    const secs = totalSecs % 60;
    if (hrs > 0) return `${hrs}h${String(mins).padStart(2, "0")}m${String(secs).padStart(2, "0")}s`;
    if (mins > 0) return `${mins}m${String(secs).padStart(2, "0")}s`;
    return `${secs}s`;
}

function formatPercentage(pct) {
    return `${pct.toFixed(1)}%`;
}

// ── Transfer State ───────────────────────────────────────────────

class TransferState {
  constructor(transferId, filename, isSnapshot = false) {
    this.transferId = transferId;
    this.filename = filename;
    this.isSnapshot = isSnapshot;
    this.status = "running"; // running | completed | error | cancelled
    this.eventSource = null;
    this.lastEvent = null;
    this.cardElement = null;
  }
}

// ── App Controller ───────────────────────────────────────────────

const App = {
    transfers: new Map(),

    // ── Form Submission ──────────────────────────────────────────

    async startDownload(repoId, filename, localDir, useXet, allowPatterns) {
        try {
            const isSnapshot = !filename;
            const displayName = filename || `${repoId} (full repo)`;
            let url = `/hf-track/download?repo_id=${encodeURIComponent(repoId)}`;
            if (filename) {
                url += `&filename=${encodeURIComponent(filename)}`;
            }
            if (localDir) {
                url += `&local_dir=${encodeURIComponent(localDir)}`;
            }
            url += `&use_xet=${useXet ? "true" : "false"}`;
            if (allowPatterns) {
                url += `&allow_patterns=${encodeURIComponent(allowPatterns)}`;
            }
        const resp = await fetch(url, { method: "POST" });
        if (!resp.ok) {
          const err = await resp.text();
          alert(`Failed to start download: ${err}`);
          return;
        }
        const data = await resp.json();
        const state = new TransferState(data.transfer_id, displayName, isSnapshot);
        this.transfers.set(data.transfer_id, state);
        this.createTransferCard(state);
        this.listenToEvents(state);
        this.updateStatusBar();
      } catch (e) {
        alert(`Network error: ${e.message}`);
      }
    },

    // ── DOM: Create Transfer Card ────────────────────────────────

    createTransferCard(state) {
        // Remove empty state message
        const container = document.getElementById("active-transfers");
        const emptyMsg = container.querySelector(".empty-state");
        if (emptyMsg) emptyMsg.remove();

        const card = document.createElement("div");
        card.className = "transfer-card";
        card.dataset.transferId = state.transferId;

        card.innerHTML = `
            <div class="transfer-header">
                <span class="filename" title="${state.filename}">${state.filename}</span>
                <span class="status-badge running">Running</span>
                <button class="cancel-btn" title="Cancel transfer">✕ Cancel</button>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: 0%"></div>
            </div>
            <div class="transfer-stats">
                <span class="percentage">0.0%</span>
                <span class="speed">—</span>
                <span class="bytes">0 B / —</span>
                <span class="eta">ETA: --:--</span>
            </div>
        `;

        // Wire cancel button
        card.querySelector(".cancel-btn").addEventListener("click", () => {
            this.cancelTransfer(state.transferId);
        });

        container.appendChild(card);
        state.cardElement = card;
    },

    // ── SSE: Listen to Events ────────────────────────────────────

    listenToEvents(state) {
        const es = new EventSource(`/hf-track/events/${state.transferId}`);
        state.eventSource = es;

        es.onmessage = (e) => {
            let event;
            try {
                event = JSON.parse(e.data);
            } catch {
                return;  // Skip malformed events
            }
            state.lastEvent = event;
            this.updateTransferCard(state, event);

            if (["complete", "error", "cancelled"].includes(event.event_type)) {
                es.close();
                state.status = event.event_type;
                this.moveToHistory(state);
                this.updateStatusBar();
            }
        };

        es.onerror = () => {
            es.close();
            if (state.status === "running") {
                state.status = "error";
                this.updateStatusBadge(state, "error", "Connection lost");
                this.moveToHistory(state);
                this.updateStatusBar();
            }
        };
    },

    // ── DOM: Update Transfer Card ────────────────────────────────

    updateTransferCard(state, event) {
        const card = state.cardElement;
        if (!card) return;

        // Update progress bar
        const fill = card.querySelector(".progress-bar-fill");
        const pct = Math.min(event.percentage || 0, 100);
        fill.style.width = `${pct}%`;

        // Update stats
        const pctEl = card.querySelector(".percentage");
        const speedEl = card.querySelector(".speed");
        const bytesEl = card.querySelector(".bytes");
        const etaEl = card.querySelector(".eta");

        pctEl.textContent = formatPercentage(pct);
        speedEl.textContent = event.speed > 0 ? formatSpeed(event.speed) : "—";
        bytesEl.textContent = `${formatBytes(event.bytes_completed || 0)} / ${event.total_bytes > 0 ? formatBytes(event.total_bytes) : "—"}`;

        // ETA calculation
        if (event.speed > 0 && event.total_bytes > 0) {
            const remaining = event.total_bytes - event.bytes_completed;
            etaEl.textContent = `ETA: ${formatEta(remaining / event.speed)}`;
        } else {
            etaEl.textContent = "ETA: --:--";
        }

        // Update status badge for terminal events
        if (event.event_type === "complete") {
            this.updateStatusBadge(state, "completed", "Completed");
            fill.classList.add("completed");
            fill.style.width = "100%";
            pctEl.textContent = "100.0%";
        } else if (event.event_type === "error") {
            this.updateStatusBadge(state, "error", "Error");
            fill.classList.add("error");
            // Show error message
            if (event.error && event.error.message) {
                const errDiv = document.createElement("div");
                errDiv.className = "error-message";
                errDiv.textContent = event.error.message;
                card.appendChild(errDiv);
            }
        } else if (event.event_type === "cancelled") {
            this.updateStatusBadge(state, "cancelled", "Cancelled");
            fill.classList.add("cancelled");
        }
    },

    // ── DOM: Update Status Badge ─────────────────────────────────

    updateStatusBadge(state, status, label) {
        const card = state.cardElement;
        if (!card) return;

        const badge = card.querySelector(".status-badge");
        badge.className = `status-badge ${status}`;
        badge.textContent = label;

        // Remove cancel button for terminal states
        const cancelBtn = card.querySelector(".cancel-btn");
        if (cancelBtn && status !== "running") {
            cancelBtn.remove();
        }
    },

    // ── Cancel Transfer ──────────────────────────────────────────

    async cancelTransfer(transferId) {
        try {
            await fetch(`/hf-track/cancel/${transferId}`, { method: "POST" });
        } catch (e) {
            console.error("Cancel failed:", e);
        }
    },

    // ── DOM: Move to History ─────────────────────────────────────

    moveToHistory(state) {
        const card = state.cardElement;
        if (!card) return;

        // Remove from active
        card.remove();

        // Add to history
        const history = document.getElementById("transfer-history");
        const emptyMsg = history.querySelector(".empty-state");
        if (emptyMsg) emptyMsg.remove();

        // Remove cancel button if still present
        const cancelBtn = card.querySelector(".cancel-btn");
        if (cancelBtn) cancelBtn.remove();

        history.prepend(card);

        // Check if active section is now empty
        const active = document.getElementById("active-transfers");
        if (!active.querySelector(".transfer-card")) {
            const p = document.createElement("p");
            p.className = "empty-state";
            p.textContent = "No active transfers. Start a download above.";
            active.appendChild(p);
        }
    },

    // ── Status Bar ───────────────────────────────────────────────

    updateStatusBar() {
        let activeCount = 0;
        for (const state of this.transfers.values()) {
            if (state.status === "running") activeCount++;
        }
        document.getElementById("active-count").textContent = `Active: ${activeCount}`;

        // Poll queue size
        fetch("/hf-track/status")
            .then(r => r.json())
            .then(data => {
                document.getElementById("queue-size").textContent = `Queue: ${data.queue_size || 0}`;
            })
            .catch(() => {});
    },
};

// ── Initialize ───────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("download-form");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const repoId = document.getElementById("repo-id").value.trim();
        const filename = document.getElementById("filename").value.trim();
        const localDir = document.getElementById("local-dir").value.trim();
        const useXet = document.getElementById("use-xet").checked;
        const allowPatterns = document.getElementById("allow-patterns").value.trim();
        if (!repoId) return;
        App.startDownload(repoId, filename || null, localDir || null, useXet, allowPatterns || null);
    // Don't clear inputs — user may want to download another file from same repo
  });

    // Poll status bar every 2 seconds
    setInterval(() => App.updateStatusBar(), 2000);
});
