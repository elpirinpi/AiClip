const $ = (selector) => document.querySelector(selector);
const homeScreen = $("#homeScreen");
const settingsScreen = $("#settingsScreen");
const detailScreen = $("#detailScreen");
let activeJobId = null;
let jobsTimer = null;
let toastTimer = null;

const STATUS_LABELS = {
  queued: "En cola",
  processing: "Analizando",
  ready: "Clips listos",
  render_queued: "Exportación en cola",
  rendering: "Exportando",
  completed: "Completado",
  failed: "Error",
};

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(result.detail || "No se pudo completar la solicitud.");
  }
  return result;
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("visible"), 3800);
}

function showScreen(screen) {
  homeScreen.classList.toggle("hidden", screen !== "home");
  settingsScreen.classList.toggle("hidden", screen !== "settings");
  detailScreen.classList.toggle("hidden", screen !== "detail");
  $("#homeTab").classList.toggle("active", screen === "home");
  $("#settingsTab").classList.toggle("active", screen === "settings");
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${remainder}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function jobCard(job) {
  const name = job.source.startsWith("http")
    ? job.source
    : job.source.split(/[\\/]/).pop();
  return `
    <article class="job-card" data-job-id="${escapeHtml(job.id)}">
      <div class="job-icon" aria-hidden="true">▧</div>
      <div class="job-main">
        <div class="job-name">${escapeHtml(name)}</div>
        <div class="job-meta">${new Date(job.created_at + "Z").toLocaleString()}</div>
      </div>
      <span class="job-status ${escapeHtml(job.status)}">${STATUS_LABELS[job.status] || "En proceso"}</span>
    </article>`;
}

async function loadJobs() {
  const jobs = await api("/api/jobs");
  const list = $("#jobsList");
  list.innerHTML = jobs.length
    ? jobs.map(jobCard).join("")
    : '<div class="empty-state">Todavía no hay proyectos. Añade un vídeo para comenzar.</div>';
  list.querySelectorAll("[data-job-id]").forEach((card) => {
    card.addEventListener("click", () => openJob(card.dataset.jobId));
  });
}

function updateConsent() {
  const enabled = $("#consent").checked;
  $("#submitJob").disabled = !enabled || !($("#videoUrl").value.trim() || $("#videoFile").files.length);
}

$("#videoUrl").addEventListener("input", () => {
  if ($("#videoUrl").value.trim()) {
    $("#videoFile").value = "";
    $("#fileName").textContent = "Arrastra tu vídeo aquí";
  }
  updateConsent();
});
$("#videoFile").addEventListener("change", () => {
  const file = $("#videoFile").files[0];
  if (file) {
    $("#videoUrl").value = "";
    $("#fileName").textContent = file.name;
  } else {
    $("#fileName").textContent = "Arrastra tu vídeo aquí";
  }
  updateConsent();
});
$("#consent").addEventListener("change", updateConsent);

const dropzone = $("#dropzone");
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  $("#videoFile").files = transfer.files;
  $("#videoFile").dispatchEvent(new Event("change"));
});

$("#jobForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("#submitJob");
  button.disabled = true;
  button.querySelector("span").textContent = "Añadiendo a la cola…";
  try {
    const body = new FormData();
    const file = $("#videoFile").files[0];
    if (file) body.append("video", file);
    else body.append("source", $("#videoUrl").value.trim());
    const job = await api("/api/jobs", { method: "POST", body });
    $("#videoUrl").value = "";
    $("#videoFile").value = "";
    $("#fileName").textContent = "Arrastra tu vídeo aquí";
    $("#consent").checked = false;
    await loadJobs();
    toast("Vídeo añadido a la cola.");
    await openJob(job.id);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.querySelector("span").textContent = "Analizar vídeo";
    updateConsent();
  }
});

$("#refreshJobs").addEventListener("click", () => loadJobs().catch((error) => toast(error.message, true)));
$("#homeTab").addEventListener("click", () => {
  clearInterval(jobsTimer);
  showScreen("home");
});
$("#settingsTab").addEventListener("click", () => {
  clearInterval(jobsTimer);
  showScreen("settings");
  loadSettings().catch((error) => toast(error.message, true));
});
$("#backHome").addEventListener("click", () => {
  clearInterval(jobsTimer);
  showScreen("home");
  loadJobs().catch((error) => toast(error.message, true));
});

async function loadSettings() {
  const settings = await api("/api/settings");
  $("#outputFolder").value = settings.output_folder;
  $("#subtitleStyle").value = settings.subtitle_style;
  $("#reframeMode").value = settings.reframe_mode;
  $("#groqStatus").textContent = settings.groq_key_configured ? "Configurada · clave oculta" : "Falta configurar";
  $("#geminiStatus").textContent = settings.gemini_key_configured ? "Configurada · clave oculta" : "Falta configurar";
}

$("#settingsForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = $("#settingsStatus");
  status.textContent = "Guardando…";
  const body = {
    groq_api_key: $("#groqKey").value,
    gemini_api_key: $("#geminiKey").value,
    output_folder: $("#outputFolder").value,
    subtitle_style: $("#subtitleStyle").value,
    reframe_mode: $("#reframeMode").value,
  };
  try {
    await api("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("#groqKey").value = "";
    $("#geminiKey").value = "";
    await loadSettings();
    status.textContent = "Ajustes guardados en este PC.";
  } catch (error) {
    status.textContent = "";
    toast(error.message, true);
  }
});

async function openJob(jobId) {
  activeJobId = jobId;
  showScreen("detail");
  await updateJobView();
  clearInterval(jobsTimer);
  jobsTimer = setInterval(() => {
    updateJobView().catch((error) => toast(error.message, true));
    loadJobs().catch(() => {});
  }, 2000);
}

async function updateJobView() {
  if (!activeJobId) return;
  const job = await api(`/api/jobs/${encodeURIComponent(activeJobId)}`);
  $("#detailName").textContent = job.source.startsWith("http")
    ? job.source
    : job.source.split(/[\\/]/).pop();
  $("#detailStatus").textContent = STATUS_LABELS[job.status] || job.status;
  const busy = ["queued", "processing", "render_queued", "rendering"].includes(job.status);
  $("#jobProgress").classList.toggle("hidden", !busy && job.status !== "failed");
  $("#progressLabel").textContent = STATUS_LABELS[job.status] || job.status;
  $("#progressValue").textContent = `${job.progress}%`;
  $("#progressBar").style.width = `${job.progress}%`;
  $("#jobError").textContent = job.error || "";
  $("#jobError").classList.toggle("hidden", !job.error);
  $("#manualClipForm").classList.toggle("hidden", !["ready", "completed"].includes(job.status));
  $("#renderSelected").disabled = busy || !["ready", "completed"].includes(job.status);
  $("#downloadSelected").disabled = !["completed"].includes(job.status);
  $("#downloadRow").classList.toggle("hidden", job.status !== "completed");
  $("#downloadLink").href = `/api/jobs/${encodeURIComponent(activeJobId)}/download`;
  $("#clipCount").textContent = `${job.clips.length} clips`;
  const selectedCount = job.clips.filter((clip) => clip.selected).length;
  $("#selectAll").checked = job.clips.length > 0 && selectedCount === job.clips.length;
  $("#renderSelected").disabled ||= selectedCount === 0;
  renderClipList(job);
  if (!busy) {
    clearInterval(jobsTimer);
    loadJobs().catch(() => {});
  }
}

function renderClipList(job) {
  const list = $("#clipsList");
  if (!job.clips.length) {
    list.innerHTML = job.status === "ready" || job.status === "completed"
      ? '<div class="empty-state">No se encontraron momentos destacados. Puedes crear un clip manual.</div>'
      : '<div class="empty-state">Los clips aparecerán aquí cuando termine el análisis.</div>';
    return;
  }
  list.innerHTML = job.clips.map((clip) => {
    const preview = job.status === "completed" && clip.selected
      ? `<video class="clip-preview" controls preload="none" src="/api/jobs/${encodeURIComponent(job.id)}/clips/${clip.id}/preview"></video>`
      : "";
    return `
      <article class="clip-card" data-clip-id="${clip.id}">
        <div class="clip-card-top">
          <input class="clip-check" type="checkbox" aria-label="Seleccionar clip" ${clip.selected ? "checked" : ""}>
          <div class="clip-info">
            <div class="clip-title">${escapeHtml(clip.title)}</div>
            <p class="clip-reason">${escapeHtml(clip.reason)}</p>
          </div>
          <span class="score">${clip.score}/10</span>
        </div>
        <div class="clip-edit-grid">
          <label><span class="field-label">Inicio · s</span><input class="text-input clip-start" type="number" min="0" step="0.1" value="${clip.start.toFixed(1)}"></label>
          <label><span class="field-label">Fin · s</span><input class="text-input clip-end" type="number" min="0" step="0.1" value="${clip.end.toFixed(1)}"></label>
          <label><span class="field-label">Título</span><input class="text-input clip-title-input" maxlength="120" value="${escapeHtml(clip.title)}"></label>
          <button class="button button-quiet clip-save" type="button">Guardar</button>
        </div>
        ${preview}
      </article>`;
  }).join("");

  list.querySelectorAll(".clip-card").forEach((card) => {
    const clipId = Number(card.dataset.clipId);
    card.querySelector(".clip-check").addEventListener("change", async (event) => {
      const clip = job.clips.find((item) => item.id === clipId);
      await saveClip(card, clipId, job.id, event.target.checked, clip);
    });
    card.querySelector(".clip-save").addEventListener("click", async () => {
      const clip = job.clips.find((item) => item.id === clipId);
      await saveClip(card, clipId, job.id, card.querySelector(".clip-check").checked, clip);
    });
  });
}

async function saveClip(card, clipId, jobId, selected, original) {
  try {
    await api(`/api/jobs/${encodeURIComponent(jobId)}/clips/${clipId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: Number(card.querySelector(".clip-start").value),
        end: Number(card.querySelector(".clip-end").value),
        title: card.querySelector(".clip-title-input").value,
        selected,
      }),
    });
    toast("Clip actualizado.");
    await updateJobView();
  } catch (error) {
    if (original) card.querySelector(".clip-check").checked = original.selected;
    toast(error.message, true);
  }
}

$("#selectAll").addEventListener("change", async (event) => {
  if (!activeJobId) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(activeJobId)}/selection`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected: event.target.checked }),
    });
    await updateJobView();
  } catch (error) {
    toast(error.message, true);
  }
});

$("#manualClipForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api(`/api/jobs/${encodeURIComponent(activeJobId)}/clips`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        start: Number($("#manualStart").value),
        end: Number($("#manualEnd").value),
        title: $("#manualTitle").value,
      }),
    });
    $("#manualTitle").value = "";
    toast("Clip manual añadido.");
    await updateJobView();
  } catch (error) {
    toast(error.message, true);
  }
});

$("#renderSelected").addEventListener("click", async () => {
  if (!activeJobId) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(activeJobId)}/render`, { method: "POST" });
    toast("Los clips seleccionados se añadieron a la cola de exportación.");
    await updateJobView();
  } catch (error) {
    toast(error.message, true);
  }
});

$("#downloadSelected").addEventListener("click", () => {
  if (activeJobId) window.location.href = `/api/jobs/${encodeURIComponent(activeJobId)}/download`;
});

loadJobs().catch((error) => toast(error.message, true));
updateConsent();
