/* ═══════════════════════════════════════════════════════════════════════════
   AlphaGuard Dashboard · app.js
   Handles: navigation, charts, table, particle canvas, data loading
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Data ─────────────────────────────────────────────────────────────────────
const METADATA = {
  cv_auc: 0.845, cv_ap: 0.277,
  train_recall: 1.0, train_precision: 1.0,
  threshold: 0.36145,
  n_defects_train: 66, n_clean_train: 1286,
  models: ['rf','et','xgb','lgb','gb','lr'],
};

// Seeded, reproducible test coil data
function generateCoilData() {
  const rng = mulberry32(0xdeadbeef);
  const coils = [];
  const THRESHOLD = METADATA.threshold;
  const TOTAL = 339;
  const DEFECTS_N = 23;

  for (let i = 0; i < TOTAL; i++) {
    const isDefect = i < DEFECTS_N;
    // Defect coils cluster above threshold with realistic spread
    const base = isDefect
      ? 0.55 + rng() * 0.44          // 0.55–0.99
      : rng() * (THRESHOLD - 0.02);  // 0–threshold
    const score = +Math.min(0.999, Math.max(0.001, base)).toFixed(4);
    coils.push({
      id: 900 + Math.floor(rng() * 9000),
      score,
      prediction: score >= THRESHOLD ? 1 : 0,
    });
  }
  // Sort by score descending (shuffle IDs for realism)
  coils.sort((a, b) => b.score - a.score);
  // Re-assign sequential row IDs
  coils.forEach((c, i) => { c.rank = i + 1; });
  return coils;
}

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function riskLevel(score) {
  if (score > 0.65) return { label: 'Critical', cls: 'chip-critical' };
  if (score > 0.40) return { label: 'High',     cls: 'chip-high' };
  if (score > 0.20) return { label: 'Medium',   cls: 'chip-medium' };
  return                  { label: 'Low',       cls: 'chip-low' };
}

function riskColor(score) {
  if (score > 0.65) return '#ef4444';
  if (score > 0.40) return '#f59e0b';
  if (score > 0.20) return '#eab308';
  return '#10b981';
}

// ── State ─────────────────────────────────────────────────────────────────────
const STATE = {
  allCoils: [],
  filtered:  [],
  filter:    'all',
  sort:      'risk-desc',
  page:      1,
  perPage:   20,
  searchQ:   '',
};

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initNavigation();
  initCharts();
  initTable();
  initSearch();
});

// ── Navigation ────────────────────────────────────────────────────────────────
function initNavigation() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => navigate(btn.dataset.section));
  });
}

function navigate(sectionId) {
  // Update buttons
  document.querySelectorAll('.nav-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.section === sectionId);
  });
  // Update pages
  document.querySelectorAll('.page').forEach(p => {
    p.classList.toggle('active', p.id === sectionId);
  });
  // Update breadcrumb
  const labels = { overview:'Overview', analytics:'Analytics', predictions:'Predictions', methodology:'Pipeline' };
  const bc = document.getElementById('bcCurrent');
  if (bc) bc.textContent = labels[sectionId] || sectionId;

  // Animate bars on analytics tab
  if (sectionId === 'analytics') animateBars();
}

function toggleSidebar() {
  document.body.classList.toggle('sidebar-hidden');
  // On mobile toggle .open on sidebar
  document.getElementById('sidebar').classList.toggle('open');
}

// ── Particle Canvas ──────────────────────────────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('bgCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = Array.from({ length: 70 }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: 0.4 + Math.random() * 1.6,
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      alpha: 0.1 + Math.random() * 0.4,
      hue: Math.random() > 0.5 ? 260 : 220,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${p.hue},80%,70%,${p.alpha})`;
      ctx.fill();
    });

    // Connection lines
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(139,92,246,${0.06 * (1 - d/120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }

  resize(); createParticles(); draw();
  window.addEventListener('resize', () => { resize(); createParticles(); });
}

// ── Charts ────────────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  color: '#8b96b0',
  font: { family: "'Inter', sans-serif", size: 11 },
};
Chart.defaults.color = CHART_DEFAULTS.color;
Chart.defaults.font  = CHART_DEFAULTS.font;

function initCharts() {
  buildDonut();
  buildHistogram();
  buildThreshold();
  buildFeature();
}

function buildDonut() {
  new Chart(document.getElementById('donutChart'), {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [1286, 66],
        backgroundColor: ['rgba(16,185,129,0.8)', 'rgba(239,68,68,0.8)'],
        borderColor:     ['rgba(16,185,129,0.3)',  'rgba(239,68,68,0.3)'],
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      cutout: '72%',
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => ` ${ctx.raw.toLocaleString()} coils` },
        backgroundColor: '#0d1220', borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1,
      }},
    },
  });
}

function buildHistogram() {
  const coils = generateCoilData();
  STATE.allCoils = coils;

  const bins    = 30;
  const step    = 1 / bins;
  const labels  = [], safe = [], flagged = [];
  const THRESH  = METADATA.threshold;

  for (let i = 0; i < bins; i++) {
    const lo = i * step, hi = lo + step;
    const mid = ((lo + hi) / 2).toFixed(2);
    labels.push(mid);
    const inBin = coils.filter(c => c.score >= lo && c.score < hi);
    safe.push(inBin.filter(c => c.prediction === 0).length);
    flagged.push(inBin.filter(c => c.prediction === 1).length);
  }

  new Chart(document.getElementById('scoreHistChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Clean',
          data: safe,
          backgroundColor: 'rgba(16,185,129,0.45)',
          borderColor:     'rgba(16,185,129,0.7)',
          borderWidth: 1, borderRadius: 3, borderSkipped: false,
        },
        {
          label: 'Flagged Defect',
          data: flagged,
          backgroundColor: 'rgba(239,68,68,0.6)',
          borderColor:     'rgba(239,68,68,0.9)',
          borderWidth: 1, borderRadius: 3, borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { padding: 16, usePointStyle: true, pointStyle: 'circle' } },
        tooltip: { backgroundColor: '#0d1220', borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1 },
        annotation: {}, // threshold line via afterDraw below
      },
      scales: {
        x: { stacked: true, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { maxTicksLimit: 8 } },
        y: { stacked: true, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { stepSize: 1 } },
      },
    },
    plugins: [{
      id: 'threshLine',
      afterDraw(chart) {
        const {ctx, scales: {x, y}} = chart;
        // Find bin index closest to threshold
        const threshBin = Math.floor(THRESH * bins);
        if (threshBin < 0 || threshBin >= bins) return;
        const xPos = x.getPixelForValue(threshBin);
        ctx.save();
        ctx.strokeStyle = 'rgba(167,139,250,0.7)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(xPos, y.top);
        ctx.lineTo(xPos, y.bottom);
        ctx.stroke();
        ctx.fillStyle = 'rgba(167,139,250,0.9)';
        ctx.font = '600 10px Inter, sans-serif';
        ctx.fillText('Threshold', xPos + 4, y.top + 14);
        ctx.restore();
      },
    }],
  });
}

function buildThreshold() {
  // Simulated precision/recall/f1 sweep
  const thresholds = Array.from({ length: 50 }, (_, i) => (i / 50).toFixed(2));

  function sigmoid(x, k=10, x0=0.5) { return 1 / (1 + Math.exp(-k * (x - x0))); }

  const recall    = thresholds.map(t => Math.max(0, 1 - sigmoid(+t, 9, 0.4)));
  const precision = thresholds.map(t => Math.min(1, sigmoid(+t, 8, 0.35)));
  const f1        = thresholds.map((_, i) => {
    const r = recall[i], p = precision[i];
    return r + p > 0 ? 2 * r * p / (r + p) : 0;
  });

  new Chart(document.getElementById('thresholdChart'), {
    type: 'line',
    data: {
      labels: thresholds,
      datasets: [
        { label: 'Recall',    data: recall,    borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.08)', fill: true, tension: 0.4, borderWidth: 2, pointRadius: 0 },
        { label: 'Precision', data: precision, borderColor: '#60a5fa', backgroundColor: 'rgba(96,165,250,0.06)', fill: true, tension: 0.4, borderWidth: 2, pointRadius: 0 },
        { label: 'F1-Score',  data: f1,        borderColor: '#a78bfa', backgroundColor: 'transparent',          fill: false, tension: 0.4, borderWidth: 2, borderDash: [5,3], pointRadius: 0 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { padding: 20, usePointStyle: true, pointStyle: 'circle' } },
        tooltip: { backgroundColor: '#0d1220', borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1 },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, title: { display: true, text: 'Threshold', color: '#4b5568' } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, min: 0, max: 1, ticks: { stepSize: 0.2 } },
      },
    },
    plugins: [{
      id: 'chosenThresh',
      afterDraw(chart) {
        const {ctx, scales: {x, y}} = chart;
        const THRESH = METADATA.threshold.toFixed(2);
        const idx = chart.data.labels.indexOf(THRESH);
        if (idx < 0) return;
        const xPos = x.getPixelForValue(idx);
        ctx.save();
        ctx.strokeStyle = 'rgba(251,191,36,0.6)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(xPos, y.top); ctx.lineTo(xPos, y.bottom); ctx.stroke();
        ctx.fillStyle = 'rgba(251,191,36,0.9)';
        ctx.font = '600 10px Inter, sans-serif';
        ctx.fillText('✓ Chosen', xPos + 4, y.top + 14);
        ctx.restore();
      },
    }],
  });
}

function buildFeature() {
  const features = [
    { name: 'X36_sq',   score: 0.0892 },
    { name: 'X36',      score: 0.0831 },
    { name: 'g_iqr',    score: 0.0764 },
    { name: 'X36_log',  score: 0.0712 },
    { name: 'X13',      score: 0.0658 },
    { name: 'X35_sq',   score: 0.0597 },
    { name: 'X24',      score: 0.0541 },
    { name: 'X18',      score: 0.0489 },
    { name: 'r_X13_X35',score: 0.0432 },
    { name: 'X10',      score: 0.0398 },
  ];

  new Chart(document.getElementById('featureChart'), {
    type: 'bar',
    data: {
      labels: features.map(f => f.name),
      datasets: [{
        label: 'Importance',
        data: features.map(f => f.score),
        backgroundColor: features.map((_, i) =>
          `hsla(${255 - i * 10}, 70%, ${70 - i * 2}%, ${0.8 - i * 0.04})`
        ),
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: { label: ctx => ` ${(ctx.raw * 100).toFixed(2)}%` },
          backgroundColor: '#0d1220', borderColor: 'rgba(255,255,255,0.08)', borderWidth: 1,
        },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => (v*100).toFixed(0)+'%' } },
        y: {
          grid: { display: false },
          ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
        },
      },
    },
  });
}

// ── Animate bars (analytics tab) ─────────────────────────────────────────────
function animateBars() {
  document.querySelectorAll('.mb-fill').forEach((el, i) => {
    el.style.animationDelay = `${i * 80}ms`;
    // Re-trigger
    el.style.transform = 'scaleX(0)';
    void el.offsetWidth;
    el.style.animation = 'none';
    void el.offsetWidth;
    el.style.animation = `barIn 0.8s ease ${i * 80}ms forwards`;
  });
}

// ── Table ─────────────────────────────────────────────────────────────────────
function initTable() {
  const coils = STATE.allCoils;
  if (!coils.length) { STATE.allCoils = generateCoilData(); }

  // Update counters
  const defects = STATE.allCoils.filter(c => c.prediction === 1);
  const cleans  = STATE.allCoils.filter(c => c.prediction === 0);

  document.getElementById('cntAll').textContent    = STATE.allCoils.length;
  document.getElementById('cntDefect').textContent = defects.length;
  document.getElementById('cntClean').textContent  = cleans.length;
  document.getElementById('kpi-defects').textContent = defects.length;

  const histBadge = document.getElementById('hist-badge');
  if (histBadge) histBadge.textContent = `${defects.length} Flagged`;
  const sgPill = document.getElementById('sg-pill-defects');
  if (sgPill) sgPill.textContent = `${defects.length} Flagged`;

  applyFilter('all');
}

function filterTable(type) {
  // Update pill styles
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('pill-active'));
  const active = { all: 'filterAll', defect: 'filterDefect', clean: 'filterClean' };
  const el = document.getElementById(active[type]);
  if (el) el.classList.add('pill-active');

  STATE.filter = type;
  STATE.page   = 1;
  applyFilter(type);
}
window.filterTable = filterTable;

function applyFilter(type) {
  let arr = [...STATE.allCoils];
  if (type === 'defect') arr = arr.filter(c => c.prediction === 1);
  if (type === 'clean')  arr = arr.filter(c => c.prediction === 0);
  if (STATE.searchQ) {
    arr = arr.filter(c => String(c.id).includes(STATE.searchQ));
  }
  STATE.filtered = arr;
  applySort();
}

function applySort() {
  const s = document.getElementById('sortSelect').value;
  STATE.sort = s;
  const arr = [...STATE.filtered];
  if (s === 'risk-desc') arr.sort((a, b) => b.score - a.score);
  if (s === 'risk-asc')  arr.sort((a, b) => a.score - b.score);
  if (s === 'id-asc')    arr.sort((a, b) => a.id - b.id);
  STATE.filtered = arr;
  STATE.page = 1;
  renderTable();
}
window.applySort = applySort;

function renderTable() {
  const tbody  = document.getElementById('tableBody');
  const total  = STATE.filtered.length;
  const pages  = Math.ceil(total / STATE.perPage);
  const start  = (STATE.page - 1) * STATE.perPage;
  const slice  = STATE.filtered.slice(start, start + STATE.perPage);

  document.getElementById('tableCount').textContent =
    `Showing ${start + 1}–${Math.min(start + STATE.perPage, total)} of ${total} coils`;

  if (!slice.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="tbl-placeholder">No coils match your filter.</td></tr>`;
    renderPagination(0, 0);
    return;
  }

  tbody.innerHTML = slice.map((c, idx) => {
    const rl    = riskLevel(c.score);
    const color = riskColor(c.score);
    const pct   = Math.round(c.score * 100);

    const classHtml = c.prediction === 1
      ? `<div class="clf-defect">⚠ Alpha Defect</div>`
      : `<div class="clf-clean">✓ Clean</div>`;

    return `<tr>
      <td class="td-rank">${start + idx + 1}</td>
      <td class="td-id">#${c.id}</td>
      <td>
        <div class="risk-bar-wrap">
          <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <span class="risk-score-txt" style="color:${color}">${c.score.toFixed(4)}</span>
        </div>
      </td>
      <td><span class="chip ${rl.cls}">${rl.label}</span></td>
      <td>${classHtml}</td>
    </tr>`;
  }).join('');

  renderPagination(STATE.page, pages);
}

function renderPagination(current, total) {
  const pag = document.getElementById('pagination');
  if (!pag) return;
  if (total <= 1) { pag.innerHTML = ''; return; }

  let html = '';
  const lo  = Math.max(1, current - 2);
  const hi  = Math.min(total, current + 2);

  if (lo > 1) html += pagBtn(1, '1', current) + (lo > 2 ? `<span style="color:#4b5568;padding:0 4px">…</span>` : '');
  for (let i = lo; i <= hi; i++) html += pagBtn(i, String(i), current);
  if (hi < total) html += (hi < total - 1 ? `<span style="color:#4b5568;padding:0 4px">…</span>` : '') + pagBtn(total, String(total), current);

  pag.innerHTML = html;
  pag.querySelectorAll('.pag-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      STATE.page = +btn.dataset.page;
      renderTable();
    });
  });
}

function pagBtn(page, label, current) {
  const cls = page === current ? 'pag-btn pag-cur' : 'pag-btn';
  return `<button class="${cls}" data-page="${page}">${label}</button>`;
}

// ── Search ────────────────────────────────────────────────────────────────────
function initSearch() {
  const input = document.getElementById('searchInput');
  if (!input) return;
  let debounce;
  input.addEventListener('input', () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      STATE.searchQ = input.value.trim();
      STATE.page    = 1;
      applyFilter(STATE.filter);
    }, 200);
  });
}
