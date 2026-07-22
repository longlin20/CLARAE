"""
plots/plot_misclassified.py
===========================
Interactive HTML viewer for misclassified EGM signals.

Exports
-------
visualize_misclassified  — main entry point
plot_misclassified       — alias (backwards-compatible name)
"""

import json as _json

import numpy as np

from eval_utils import RHYTHM_COLORS, WANDB_AVAILABLE

try:
    import wandb
except ImportError:
    pass


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_MISCLASSIFIED_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Misclassified Signals - __PTH_STEM__</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; padding: 20px; color: #333; }
h1 { font-size: 22px; color: #222; margin-bottom: 5px; }
.subtitle { color: #666; font-size: 14px; margin-bottom: 16px; }
.card { background: white; border-radius: 10px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 16px; }
.filter-title { font-weight: 600; font-size: 14px; margin-bottom: 10px; color: #444; }
.checkbox-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.cb-label { display: flex; align-items: center; gap: 6px; padding: 5px 14px; border-radius: 20px; cursor: pointer; font-size: 13px; font-weight: 600; user-select: none; transition: opacity 0.15s; border: 2px solid transparent; }
.cb-label input[type="checkbox"] { cursor: pointer; width: 14px; height: 14px; }
.btn-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
button { padding: 7px 18px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: opacity 0.15s; }
button:hover:not(:disabled) { opacity: 0.82; }
button:disabled { opacity: 0.35; cursor: not-allowed; }
#btnPrev { background: #6c757d; color: white; }
#btnNext { background: #0d6efd; color: white; }
#btnRandom { background: #198754; color: white; }
.counter { font-size: 13px; color: #555; margin-left: 4px; }
#plotContainer { background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 10px; min-height: 200px; }
.no-results { display: flex; align-items: center; justify-content: center; height: 200px; color: #aaa; font-size: 16px; }
</style>
</head>
<body>
<h1>Misclassified EGM Signals</h1>
<p class="subtitle">
  Model: <strong>__PTH_STEM__</strong> &nbsp;|&nbsp;
  Misclassified signals shown: <strong id="totalCount">0</strong>
</p>

<div class="card">
  <div class="filter-title">Filter by True Label:</div>
  <div class="checkbox-row" id="filterGroup"></div>
  <div class="btn-row">
    <button id="btnPrev" onclick="changePage(-1)">&#8592; Prev 5</button>
    <button id="btnNext" onclick="changePage(1)">Next 5 &#8594;</button>
    <button id="btnRandom" onclick="randomPage()">&#127922; Random 5</button>
    <span class="counter" id="counter"></span>
  </div>
</div>

<div id="plotContainer">
  <div class="no-results" id="noResults">No signals match the selected filters.</div>
</div>

<script>
var ALL_DATA = __DATA_JSON__;
var RHYTHM_COLORS = __COLORS_JSON__;
var PAGE_SIZE = 5;
var currentStart = 0;
var filteredData = [];
var activeFilters = {};

var seenLabels = {};
var uniqueLabels = [];
for (var i = 0; i < ALL_DATA.length; i++) {
  var lbl = ALL_DATA[i].true_label;
  if (!seenLabels[lbl]) { seenLabels[lbl] = true; uniqueLabels.push(lbl); }
}
uniqueLabels.sort();
for (var j = 0; j < uniqueLabels.length; j++) { activeFilters[uniqueLabels[j]] = true; }

function initFilters() {
  var grp = document.getElementById('filterGroup');
  for (var k = 0; k < uniqueLabels.length; k++) {
    var label = uniqueLabels[k];
    var col = RHYTHM_COLORS[label] || '#888888';
    var lbl = document.createElement('label');
    lbl.className = 'cb-label';
    lbl.style.background = col + '28';
    lbl.style.borderColor = col;
    lbl.style.color = col;
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.value = label;
    (function(l) { cb.onchange = function() { toggleFilter(l, this.checked); }; })(label);
    lbl.appendChild(cb);
    var span = document.createElement('span');
    span.textContent = label;
    lbl.appendChild(span);
    grp.appendChild(lbl);
  }
}

function toggleFilter(label, checked) {
  activeFilters[label] = checked;
  currentStart = 0;
  updateView();
}

function applyFilter() {
  filteredData = [];
  for (var i = 0; i < ALL_DATA.length; i++) {
    if (activeFilters[ALL_DATA[i].true_label]) filteredData.push(ALL_DATA[i]);
  }
  document.getElementById('totalCount').textContent = filteredData.length;
}

function changePage(delta) {
  var maxStart = Math.max(0, filteredData.length - PAGE_SIZE);
  currentStart = Math.max(0, Math.min(currentStart + delta * PAGE_SIZE, maxStart));
  renderView();
}

function randomPage() {
  if (filteredData.length === 0) return;
  var maxStart = Math.max(0, filteredData.length - PAGE_SIZE);
  currentStart = Math.floor(Math.random() * (maxStart + 1));
  renderView();
}

function updateView() {
  applyFilter();
  currentStart = Math.min(currentStart, Math.max(0, filteredData.length - 1));
  renderView();
}

function renderView() {
  var end = Math.min(currentStart + PAGE_SIZE, filteredData.length);
  var page = filteredData.slice(currentStart, end);
  var noRes = document.getElementById('noResults');
  var ctr = document.getElementById('counter');

  document.getElementById('btnPrev').disabled = (currentStart <= 0);
  document.getElementById('btnNext').disabled = (end >= filteredData.length);

  if (filteredData.length === 0) {
    ctr.textContent = '';
    noRes.style.display = 'flex';
    Plotly.purge('plotContainer');
    return;
  }
  noRes.style.display = 'none';
  ctr.textContent = 'Showing ' + (currentStart + 1) + '\u2013' + end + ' of ' + filteredData.length;
  renderPlot(page);
}

function getSubplotDomains(n) {
  var gap = 0.035;
  var h = (1.0 - (n - 1) * gap) / n;
  var domains = [];
  for (var i = 0; i < n; i++) {
    var top = 1.0 - i * (h + gap);
    var bot = top - h;
    domains.push([Math.max(0, bot), Math.min(1.0, top)]);
  }
  return domains;
}

function renderPlot(page) {
  if (page.length === 0) return;
  var n = page.length;
  var domains = getSubplotDomains(n);
  var traces = [];
  var layout = {
    height: Math.max(300, 190 * n),
    showlegend: false,
    margin: { t: 15, b: 40, l: 65, r: 20 },
    paper_bgcolor: 'white',
    plot_bgcolor: '#fafafa',
    annotations: [],
    font: { family: 'Segoe UI, Arial, sans-serif', size: 11 },
  };

  for (var i = 0; i < n; i++) {
    var item = page[i];
    var axSuf = (i === 0) ? '' : String(i + 1);
    var col = RHYTHM_COLORS[item.true_label] || '#5555cc';
    var bot = domains[i][0];
    var top = domains[i][1];

    traces.push({
      x: item.x,
      y: item.signal,
      type: 'scatter',
      mode: 'lines',
      line: { color: col, width: 1.3 },
      hovertemplate: 'Sample: %{x}<br>Amp: %{y:.4f}<extra></extra>',
      xaxis: 'x' + axSuf,
      yaxis: 'y' + axSuf,
    });

    layout['xaxis' + axSuf] = {
      domain: [0.0, 1.0],
      anchor: 'y' + axSuf,
      showgrid: true, gridcolor: '#ebebeb',
      zeroline: false,
      showticklabels: (i === n - 1),
      title: (i === n - 1) ? { text: 'Sample', font: { size: 12 } } : '',
    };
    layout['yaxis' + axSuf] = {
      domain: [bot, top],
      anchor: 'x' + axSuf,
      showgrid: true, gridcolor: '#ebebeb',
      zeroline: true, zerolinecolor: '#ddd', zerolinewidth: 1,
      title: { text: 'Amp', font: { size: 10 } },
    };

    layout.annotations.push({
      text: '<b>Signal #' + item.index + '</b> &nbsp;|&nbsp; True: <b>' +
            item.true_label + '</b> \u2192 Pred: <b>' + item.pred_label + '</b>',
      xref: 'paper', yref: 'paper',
      x: 0.0, y: top,
      xanchor: 'left', yanchor: 'bottom',
      showarrow: false,
      bgcolor: col + '22',
      bordercolor: col,
      borderwidth: 1,
      borderpad: 3,
      font: { size: 11, color: '#222' },
    });
  }

  Plotly.react('plotContainer', traces, layout, { responsive: true });
}

initFilters();
updateView();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_misclassified_html(data_list: list, rhythm_colors: dict,
                               pth_stem: str) -> str:
    data_json   = _json.dumps(data_list)
    colors_json = _json.dumps(rhythm_colors)
    html = _MISCLASSIFIED_HTML_TEMPLATE
    html = html.replace('__DATA_JSON__',   data_json)
    html = html.replace('__COLORS_JSON__', colors_json)
    html = html.replace('__PTH_STEM__',    pth_stem)
    return html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_misclassified(signals_test: np.ndarray,
                             labels: np.ndarray,
                             preds: np.ndarray,
                             class_names,
                             pth_stem: str,
                             max_signals: int = 2000,
                             n_display_samples: int = 1250,
                             p_inf: float = None,
                             p_sup: float = None):
    """
    Generate interactive HTML viewer for misclassified signals and log to WandB.

    Parameters
    ----------
    signals_test      : (N, 1, L) float array — normalised test signals
    labels            : (N,) int array — true class indices
    preds             : (N,) int array — predicted class indices
    class_names       : sequence of class name strings
    pth_stem          : model filename stem (used as title in the HTML)
    max_signals       : max misclassified signals embedded in the HTML
    n_display_samples : max samples per signal trace (downsampled if longer)
    p_inf, p_sup      : denormalisation bounds (optional)
    """
    misc_mask = preds != labels
    misc_idx  = np.where(misc_mask)[0]
    n_misc    = len(misc_idx)
    n_total   = len(labels)

    print(f"\n  Misclassified: {n_misc:,} / {n_total:,} "
          f"({100 * n_misc / max(n_total, 1):.1f}%)")

    if n_misc == 0:
        print("  No misclassified signals — skipping HTML viewer.")
        return

    rng = np.random.RandomState(42)
    if n_misc > max_signals:
        misc_idx = rng.choice(misc_idx, max_signals, replace=False)
        misc_idx.sort()
        print(f"  (subsampled to {max_signals} signals for HTML)")

    if p_inf is not None and p_sup is not None:
        signals_test = ((signals_test.astype(np.float32) + 1.0) / 2.0) * (p_sup - p_inf) + p_inf

    data_list = []
    for i in misc_idx:
        sig = signals_test[i, 0, :].astype(np.float32)
        L   = len(sig)
        if L > n_display_samples:
            step    = max(1, L // n_display_samples)
            indices = np.arange(0, L, step)[:n_display_samples]
            sig     = sig[indices]
        else:
            indices = np.arange(L)
        data_list.append({
            'signal':     [round(float(v), 4) for v in sig],
            'x':          list(map(int, indices)),
            'true_label': str(class_names[labels[i]]),
            'pred_label': str(class_names[preds[i]]),
            'index':      int(i),
        })

    html = _build_misclassified_html(data_list, RHYTHM_COLORS, pth_stem)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({'clf/misclassified_viewer': wandb.Html(html)})
        print("  Misclassified viewer logged to WandB.")
    else:
        print("  WandB not active — viewer not saved.")


# Backwards-compatible alias used by latent_space_analysis.py
plot_misclassified = visualize_misclassified
