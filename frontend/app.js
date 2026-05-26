// ── AlphaGuard v3 Dashboard ───────────────────────────────
'use strict';

// ── State ────────────────────────────────────────────────
let meta = null;
let allRows = [];
let filtered = [];
let currentPage = 1;
const PAGE = 20;
let activeFilter = 'all';
let activeSort   = 'risk-desc';
const charts = {};

// ── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initNav();
  loadData();
  document.getElementById('searchInput').addEventListener('input', () => { currentPage=1; applyAll(); });
});

// ── Particle Background ───────────────────────────────────
function initParticles() {
  const canvas = document.getElementById('particleCanvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const N = 55;
  for (let i = 0; i < N; i++) {
    particles.push({
      x: Math.random() * W, y: Math.random() * H,
      r: Math.random() * 1.5 + 0.3,
      vx: (Math.random() - 0.5) * 0.18,
      vy: (Math.random() - 0.5) * 0.18,
      alpha: Math.random() * 0.5 + 0.1,
      color: Math.random() > 0.5 ? '124,58,237' : '37,99,235',
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color},${p.alpha})`;
      ctx.fill();
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    });
    // Subtle connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i+1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(124,58,237,${0.06 * (1 - dist/120)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
}

// ── Navigation ────────────────────────────────────────────
function initNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', e => {
      e.preventDefault();
      const section = item.dataset.section;
      showSection(section);
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      item.classList.add('active');
    });
  });
}

function showSection(name) {
  document.querySelectorAll('.content-section').forEach(s => s.classList.add('hidden'));
  const el = document.getElementById(name);
  if (el) el.classList.remove('hidden');
  // Init charts when analytics section becomes visible
  if (name === 'analytics' && meta && !charts.threshold) buildAnalyticsCharts();
}

window.toggleSidebar = function() {
  document.getElementById('sidebar').classList.toggle('open');
};

// ── Data Loading ──────────────────────────────────────────
async function loadData() {
  try {
    const r = await fetch('model_metadata.json');
    if (!r.ok) throw new Error('no metadata');
    meta = await r.json();
    render();
  } catch(e) {
    console.warn('Using demo data', e);
    meta = buildDemo();
    render();
  }
}

function render() {
  updateTopMetrics();
  buildTableRows();
  buildDonutChart();
  buildScoreHistChart();
  applyAll();
  // Threshold card
  const el = document.getElementById('dc-thresh');
  if (el) el.textContent = meta.best_threshold?.toFixed(3) ?? '—';
}

// ── Metric Cards ──────────────────────────────────────────
function updateTopMetrics() {
  set('mc-precision', meta.train_precision != null
    ? (meta.train_precision * 100).toFixed(0) + '%'
    : (meta.cv_precision != null ? (meta.cv_precision * 100).toFixed(0) + '%' : '—'));
  set('mc-defects',   meta.n_defects_predicted ?? '—');
  const pSub = document.getElementById('mc-precision-sub');
  if (pSub && meta.train_precision != null)
    pSub.textContent = meta.train_precision === 1.0 ? 'Perfect Precision' : `≈${(meta.train_precision*100).toFixed(0)}%`;
}

function set(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Donut Chart ───────────────────────────────────────────
function buildDonutChart() {
  const ctx = document.getElementById('donutChart')?.getContext('2d');
  if (!ctx) return;
  charts.donut = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Clean', 'Defect'],
      datasets: [{
        data: [meta.n_clean_train ?? 1286, meta.n_defects_train ?? 66],
        backgroundColor: ['rgba(5,150,105,0.75)', 'rgba(220,38,38,0.8)'],
        borderColor:     ['rgba(52,211,153,0.4)', 'rgba(248,113,113,0.4)'],
        borderWidth: 2, hoverOffset: 10,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: { display: false },
        tooltip: tooltip({
          callbacks: {
            label: c => {
              const total = (meta.n_clean_train??1286) + (meta.n_defects_train??66);
              return `  ${c.label}: ${c.raw.toLocaleString()} (${(c.raw/total*100).toFixed(1)}%)`;
            }
          }
        })
      }
    }
  });
}

// ── Score Histogram ───────────────────────────────────────
function buildScoreHistChart() {
  const ctx = document.getElementById('scoreHistChart')?.getContext('2d');
  if (!ctx || !meta.test_probs) return;

  const probs = meta.test_probs;
  const preds = meta.test_predictions;
  const BINS = 25;
  const cleanBins  = new Array(BINS).fill(0);
  const defectBins = new Array(BINS).fill(0);
  probs.forEach((p, i) => {
    const bin = Math.min(Math.floor(p * BINS), BINS - 1);
    (preds[i] === 1 ? defectBins : cleanBins)[bin]++;
  });
  const labels = Array.from({length: BINS}, (_, i) => (i/BINS).toFixed(2));

  const dCount = document.getElementById('sg-pill-defects');
  if (dCount) dCount.textContent = `${meta.n_defects_predicted ?? '?'} Flagged`;

  charts.hist = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Clean',  data: cleanBins,  backgroundColor: 'rgba(5,150,105,0.55)',  borderColor: 'rgba(52,211,153,0.7)',  borderWidth: 1, borderRadius: 3 },
        { label: 'Defect', data: defectBins, backgroundColor: 'rgba(220,38,38,0.65)', borderColor: 'rgba(248,113,113,0.8)', borderWidth: 1, borderRadius: 3 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8b96ae', font: { family: 'Inter', size: 11 }, boxWidth: 10, padding: 14 } },
        tooltip: tooltip()
      },
      scales: {
        x: { stacked: true, ticks: { color: '#4b5672', font: { size: 10 } }, grid: { display: false },
             title: { display: true, text: 'Risk Score', color: '#4b5672', font: { size: 11 } } },
        y: { stacked: true, ticks: { color: '#4b5672', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

// ── Analytics Charts ──────────────────────────────────────
function buildAnalyticsCharts() {
  buildThresholdChart();
  buildFeatureChart();
}

function buildThresholdChart() {
  const ctx = document.getElementById('thresholdChart')?.getContext('2d');
  if (!ctx || !meta.threshold_sweep) return;
  const sw = meta.threshold_sweep;
  charts.threshold = new Chart(ctx, {
    type: 'line',
    data: {
      labels: sw.map(d => d.threshold),
      datasets: [
        { label: 'Recall',    data: sw.map(d=>d.recall),    borderColor:'#34d399', backgroundColor:'rgba(52,211,153,0.07)', borderWidth:2.5, pointRadius:0, fill:true, tension:0.4 },
        { label: 'Precision', data: sw.map(d=>d.precision), borderColor:'#a78bfa', backgroundColor:'rgba(167,139,250,0.06)', borderWidth:2.5, pointRadius:0, fill:true, tension:0.4 },
        { label: 'F1',        data: sw.map(d=>d.f1),        borderColor:'#fbbf24', borderWidth:1.8, pointRadius:0, fill:false, tension:0.4, borderDash:[5,3] },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      interaction: { mode:'index', intersect:false },
      plugins: {
        legend: { labels: { color:'#8b96ae', font:{family:'Inter',size:11}, boxWidth:12, padding:14 } },
        tooltip: tooltip()
      },
      scales: {
        x: { ticks:{color:'#4b5672',font:{size:10}}, grid:{color:'rgba(255,255,255,0.04)'},
             title:{display:true,text:'Decision Threshold',color:'#4b5672',font:{size:11}} },
        y: { min:0, max:1, ticks:{color:'#4b5672',font:{size:11}}, grid:{color:'rgba(255,255,255,0.04)'} }
      }
    }
  });
}

function buildFeatureChart() {
  const ctx = document.getElementById('featureChart')?.getContext('2d');
  if (!ctx || !meta.top_features) return;
  const feats = meta.top_features.slice(0,15);
  const palette = feats.map((_, i) => `rgba(${lerp(167,96,i/feats.length)},${lerp(139,165,i/feats.length)},${lerp(250,250,i/feats.length)},${0.9-i*0.04})`);

  charts.feature = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: feats.map(f => f.name),
      datasets: [{ label:'Importance', data: feats.map(f=>f.importance), backgroundColor: palette, borderRadius:5, borderSkipped:false }]
    },
    options: {
      indexAxis: 'y',
      responsive:true, maintainAspectRatio:false,
      plugins: {
        legend:{display:false},
        tooltip: tooltip({ callbacks:{ label: c => `  ${c.raw.toFixed(4)}` } })
      },
      scales: {
        x: { ticks:{color:'#4b5672',font:{size:10}}, grid:{color:'rgba(255,255,255,0.04)'} },
        y: { ticks:{color:'#93c5fd',font:{family:'JetBrains Mono',size:11}}, grid:{display:false} }
      }
    }
  });
}

function lerp(a, b, t) { return Math.round(a + (b-a)*t); }

// ── Shared Tooltip ────────────────────────────────────────
function tooltip(extra = {}) {
  return {
    backgroundColor:'#111520', borderColor:'rgba(255,255,255,0.08)', borderWidth:1,
    titleColor:'#eef0f8', bodyColor:'#8b96ae',
    padding:10, cornerRadius:8,
    ...extra
  };
}

// ── Table ─────────────────────────────────────────────────
function buildTableRows() {
  if (!meta.test_coil_ids) return;
  allRows = meta.test_coil_ids.map((id, i) => ({
    coilId: id,
    prob:   meta.test_probs?.[i] ?? 0,
    pred:   meta.test_predictions?.[i] ?? 0,
  }));
  // Update filter counts
  const nDefect = allRows.filter(r=>r.pred===1).length;
  const nClean  = allRows.filter(r=>r.pred===0).length;
  set('cntAll',    allRows.length);
  set('cntDefect', nDefect);
  set('cntClean',  nClean);
}

window.filterTable = function(type) {
  activeFilter = type;
  currentPage  = 1;
  document.querySelectorAll('.flt').forEach(b => b.classList.remove('active'));
  document.getElementById('filter' + type[0].toUpperCase() + type.slice(1))?.classList.add('active');
  applyAll();
};

window.applySort = function() {
  activeSort = document.getElementById('sortSelect').value;
  applyAll();
};

function applyAll() {
  const q = document.getElementById('searchInput').value.trim().toLowerCase();
  filtered = allRows.filter(r => {
    const matchF = activeFilter === 'all' || (activeFilter==='defect' && r.pred===1) || (activeFilter==='clean' && r.pred===0);
    const matchS = !q || String(r.coilId).includes(q);
    return matchF && matchS;
  });
  // Sort
  filtered.sort((a, b) => {
    if (activeSort === 'risk-desc') return b.prob - a.prob;
    if (activeSort === 'risk-asc')  return a.prob - b.prob;
    return a.coilId - b.coilId;
  });
  renderTable();
  renderPag();
  set('tableCount', `${filtered.length} result${filtered.length!==1?'s':''}`);
}

function renderTable() {
  const tbody = document.getElementById('tableBody');
  const start = (currentPage-1)*PAGE;
  const page  = filtered.slice(start, start+PAGE);

  if (!page.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="tbl-empty">No results found</td></tr>`;
    return;
  }

  tbody.innerHTML = page.map((row, idx) => {
    const rank = start + idx + 1;
    const pct  = (row.prob * 100);
    const pctS = pct.toFixed(1);
    const col  = pct > 65 ? '#f87171' : pct > 40 ? '#fb923c' : pct > 20 ? '#fbbf24' : '#34d399';
    const lvl  = pct > 65
      ? `<span class="lvl-pill lvl-critical">● Critical</span>`
      : pct > 40 ? `<span class="lvl-pill lvl-high">● High</span>`
      : pct > 20 ? `<span class="lvl-pill lvl-medium">● Medium</span>`
      : `<span class="lvl-pill lvl-low">● Low</span>`;
    const cls = row.pred === 1
      ? `<span class="cls-defect">⚠ Defect</span>`
      : `<span class="cls-clean">✓ Clean</span>`;
    return `
      <tr class="${row.pred===1?'row-defect':''}">
        <td><span class="tbl-rank">${rank}</span></td>
        <td><span class="tbl-coil">${row.coilId}</span></td>
        <td>
          <div class="risk-wrap">
            <div class="risk-track"><div class="risk-fill" style="width:${Math.max(pct,2)}%;background:${col}"></div></div>
            <span class="risk-pct">${pctS}%</span>
          </div>
        </td>
        <td>${lvl}</td>
        <td>${cls}</td>
      </tr>`;
  }).join('');
}

function renderPag() {
  const total = Math.ceil(filtered.length / PAGE);
  const pag   = document.getElementById('pagination');
  if (!pag || total <= 1) { if (pag) pag.innerHTML=''; return; }

  const pages = new Set([1, total, currentPage, currentPage-1, currentPage+1, currentPage-2, currentPage+2].filter(p=>p>=1&&p<=total));
  const sorted = [...pages].sort((a,b)=>a-b);
  let html = '';
  sorted.forEach((p, i) => {
    if (i>0 && p - sorted[i-1] > 1) html += `<span class="pg-ellipsis">…</span>`;
    html += `<button class="pg-btn ${p===currentPage?'active':''}" onclick="goPage(${p})">${p}</button>`;
  });
  pag.innerHTML = html;
}

window.goPage = function(p) { currentPage=p; renderTable(); renderPag(); };

// ── Demo Data ─────────────────────────────────────────────
function buildDemo() {
  const ids   = Array.from({length:339},(_,i)=>1000+i);
  const probs = ids.map(()=> Math.random()<0.08 ? Math.random()*0.5+0.15 : Math.random()*0.1);
  const preds = probs.map(p => p>=0.12 ? 1 : 0);
  return {
    train_size:1352, test_size:339, n_features:74,
    n_defects_train:66, n_clean_train:1286,
    n_defects_predicted: preds.filter(Boolean).length,
    best_threshold:0.113, cv_precision:0.91,
    train_precision:1.0, train_recall:1.0,
    top_features: ['X36','X13','X21','X32','X30','X4','X7','stage2_mean','X48','X39','X31','X24','X6','X10','X25'].map((n,i)=>({name:n,importance:+(0.103-i*0.005).toFixed(4)})),
    threshold_sweep: Array.from({length:50},(_,i)=>{
      const t=i/50; const r=Math.min(1,0.4+0.6*(1-t*1.2)); const p=Math.min(1,t*1.5);
      return {threshold:+t.toFixed(2),recall:+r.toFixed(3),precision:+p.toFixed(3),f1:+(2*r*p/(r+p||1)).toFixed(3)};
    }),
    test_probs: probs.map(p=>+p.toFixed(4)),
    test_coil_ids: ids, test_predictions: preds,
  };
}
