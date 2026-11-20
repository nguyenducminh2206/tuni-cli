// the mi-race report: tabs, switches and charts. Every number comes from the JSON block the report writes
// (#mi-lab-report-data); the tables are already in the page, this only draws and switches.
(() => {
  const root = document.getElementById('mi-lab-report');
  const data = JSON.parse(document.getElementById('mi-lab-report-data').textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const q = s => root.querySelector(s);
  const all = (s, scope = root) => [...scope.querySelectorAll(s)];
  const tabs = all('[data-lab-tab]').map(b => b.dataset.labTab);
  const hash = location.hash.slice(1);
  const state = {tab: tabs.includes(hash) ? hash : tabs[0], unit: {}, select: {}, changes: {}};
  const num = (n, digits = 3, sign = false) => (n < 0 ? '−' : sign ? '+' : '') + Math.abs(n).toFixed(digits);
  const tone = t => `var(--lab-${t})`;

  // --- svg ----------------------------------------------------------------------------------------
  function el(tag, attrs = {}, content = '') {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (content !== '') node.textContent = content;
    return node;
  }
  function tip(node, words) { node.append(el('title', {}, words)); return node; }
  function svgFor(box, height) {
    const width = box.getBoundingClientRect().width;
    if (width < 10) return null;
    const svg = el('svg', {viewBox: `0 0 ${width} ${height}`, width, height, role: 'img',
                           'aria-label': box.getAttribute('aria-label') || ''});
    box.replaceChildren(svg);
    return {svg, width};
  }
  function niceTicks(lo, hi, count) {
    const span = hi - lo || Math.abs(hi) || 1;
    const raw = span / Math.max(1, count - 1), power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map(m => m * power).find(s => s >= raw);
    const ticks = [];
    for (let t = Math.floor(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(+t.toFixed(10));
    if (ticks[ticks.length - 1] < hi - step * 1e-9) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(10));
    return ticks;
  }
  const tickText = t => String(+t.toFixed(6)).replace('-', '−');

  // --- the comparison plot: a codebook against the reference, per decoder, ±1 SE ---------------------
  function drawComparison(box, spec) {
    const multi = spec.rows.some(r => r.label);
    const layout = [];
    let y = 26, group = null;
    spec.rows.forEach(r => {
      if (multi && r.group !== group) { y += group === null ? 0 : 12; layout.push({head: r.group, y}); y += 26; group = r.group; }
      layout.push({row: r, y}); y += multi ? 46 : 60;
    });
    const height = y + 14;
    const setup = svgFor(box, height);
    if (!setup) return;
    const {svg, width} = setup;
    const longest = Math.max(...spec.rows.flatMap(r => multi ? [r.label.length + 2, r.group.length] : [r.group.length]));
    const left = Math.max(119, Math.min(260, 26 + 7 * longest)), right = width - 21, bottom = height - 39;
    const values = spec.rows.flatMap(r => [r.ref[0] - r.ref[1], r.ref[0] + r.ref[1], r.value[0] - r.value[1], r.value[0] + r.value[1]]);
    let lo = Math.min(...values), hi = Math.max(...values);
    const pad = (hi - lo) * 0.08 || 0.05;
    const ticks = niceTicks(lo - pad, hi + pad, width < 600 ? 5 : 8);
    lo = ticks[0]; hi = ticks[ticks.length - 1];
    const x = v => left + (right - left) * (v - lo) / (hi - lo);
    ticks.forEach(t => {
      svg.append(el('line', {x1: x(t), x2: x(t), y1: 12, y2: bottom, class: t === 0 ? 'lab-zero' : 'lab-guide'}));
      svg.append(el('text', {x: x(t), y: bottom + 23, 'text-anchor': 'middle', class: 'lab-tick'}, tickText(t)));
    });
    layout.forEach(item => {
      if (item.head) { svg.append(el('text', {x: 0, y: item.y + 4, class: 'lab-group'}, item.head)); return; }
      const {row: r, y} = item;
      if (multi) svg.append(el('text', {x: left - 14, y: y + 4, 'text-anchor': 'end', class: 'lab-tick'}, r.label));
      else svg.append(el('text', {x: 0, y: y + 4, 'font-weight': 500}, r.group));
      svg.append(el('line', {x1: x(r.ref[0]), x2: x(r.value[0]), y1: y, y2: y, class: 'lab-connector'}));
      [[r.ref, 'lab-start-err'], [r.value, 'lab-end-err']].forEach(([[mean, se], cls]) => {
        svg.append(el('line', {x1: x(mean - se), x2: x(mean + se), y1: y, y2: y, class: cls}));
        [-1, 1].forEach(side => svg.append(el('line', {x1: x(mean + side * se), x2: x(mean + side * se), y1: y - 5, y2: y + 5, class: cls})));
      });
      svg.append(tip(el('circle', {cx: x(r.ref[0]), cy: y, r: 4.5, fill: 'var(--lab-paper)', stroke: 'var(--lab-mark)', 'stroke-width': 1.8}),
                     `${spec.reference}: ${num(r.ref[0])} ± ${num(r.ref[1])} bits`));
      svg.append(tip(el('rect', {x: x(r.value[0]) - 4, y: y - 4, width: 8, height: 8, fill: 'var(--lab-blue)'}),
                     `${r.label || spec.other}: ${num(r.value[0])} ± ${num(r.value[1])} bits`));
      svg.append(el('text', {x: x(r.ref[0]), y: y + 20, 'text-anchor': 'middle', class: 'lab-tick'}, num(r.ref[0])));
      svg.append(el('text', {x: x(r.value[0]), y: y - 12, 'text-anchor': 'middle', 'font-weight': 500}, num(r.value[0])));
    });
  }

  // --- release schedules: one row per symbol, one baseline per codebook -----------------------------
  function drawCodebook(box, spec, key) {
    const schedules = spec.schedules, n = schedules[0].symbols.length;
    const differs = u => schedules.some(s => s.symbols[u].some((v, k) => v !== schedules[0].symbols[u][k]));
    const shown = [...Array(n).keys()].filter(u => !state.changes[key] || differs(u));
    const rowH = 26 + 43 * schedules.length, height = Math.max(1, shown.length) * rowH + 39;
    const setup = svgFor(box, height);
    if (!setup) return;
    const {svg, width} = setup, left = 99, right = width - 15, plot = right - left;
    const x = t => left + plot * t / spec.T;
    const barW = Math.max(3, Math.min(12, plot * spec.slot_dt / spec.T * 0.65));
    niceTicks(0, spec.T, 7).filter(t => t <= spec.T + 1e-9).forEach(t => {
      svg.append(el('line', {x1: x(t), x2: x(t), y1: 0, y2: height - 33, class: 'lab-guide'}));
      svg.append(el('text', {x: x(t), y: height - 10, 'text-anchor': 'middle', class: 'lab-tick'}, tickText(t)));
    });
    if (!shown.length) svg.append(el('text', {x: left, y: 30, class: 'lab-tick'}, 'No symbol changed.'));
    shown.forEach((u, row) => {
      const offset = row * rowH;
      svg.append(el('text', {x: 0, y: offset + 16, 'font-weight': 600}, `Symbol ${u}`));
      schedules.forEach((s, i) => {
        const base = offset + 45 + i * 43;
        if (s.label) svg.append(el('text', {x: left - 9, y: base + 2, 'text-anchor': 'end', class: 'lab-tick'}, s.label));
        svg.append(el('line', {x1: left, x2: right, y1: base, y2: base, class: 'lab-axis'}));
        let lastLabel = -Infinity;
        s.symbols[u].forEach((amount, k) => {
          if (!amount) return;
          const t = k * spec.slot_dt, h = amount / spec.budget * 33;
          svg.append(tip(el('rect', {x: x(t), y: base - h, width: barW, height: h, fill: tone(s.tone)}),
                         `Symbol ${u}${s.label ? ' · ' + s.label.toLowerCase() : ''}: ${amount} molecules at ${tickText(+t.toFixed(4))} s`));
          if (width > 680 && x(t) - lastLabel >= 22) {
            svg.append(el('text', {x: x(t) + barW / 2, y: base - h - 5, 'text-anchor': 'middle', class: 'lab-tick'}, amount));
            lastLabel = x(t);
          }
        });
      });
    });
  }

  // --- lines: signals, losses, learning curves; panels side by side share the y axis ----------------
  function drawLines(box, spec) {
    const panels = spec.panels, P = panels.length, height = spec.height || 260;
    const setup = svgFor(box, height);
    if (!setup) return;
    const {svg, width} = setup;
    const left = 58, right = spec.end_labels ? 104 : 14, gap = 30, top = P > 1 ? 24 : 12, bottom = 46;
    const panelW = (width - left - right - gap * (P - 1)) / P, plotH = height - top - bottom;
    const points = panels.flatMap(p => p.series.flatMap(s => s.points));
    const xMax = spec.x_max ?? Math.max(...points.map(p => p[0])), xMin = spec.x_min ?? 0;
    let yLo, yHi, yTicks, ty;
    if (spec.log) {
      const positive = points.map(p => p[1]).filter(v => v > 0);
      yLo = 10 ** Math.floor(Math.log10(Math.min(...positive)));
      yHi = 10 ** Math.ceil(Math.log10(Math.max(...positive)));
      yTicks = [];
      for (let v = yLo; v <= yHi * 1.0001; v *= 10) yTicks.push(v);
      ty = v => Math.log10(Math.max(v, yLo));
    } else {
      yTicks = niceTicks(Math.min(0, ...points.map(p => p[1])), Math.max(...points.map(p => p[1])) * 1.04, 6);
      yLo = yTicks[0]; yHi = yTicks[yTicks.length - 1];
      ty = v => v;
    }
    const y = v => top + plotH * (1 - (ty(v) - ty(yLo)) / (ty(yHi) - ty(yLo)));
    const xTicks = niceTicks(xMin, xMax, panelW < 260 ? 4 : 6).filter(t => t <= xMax + 1e-9);
    const ends = [];
    panels.forEach((panel, p) => {
      const x0 = left + p * (panelW + gap), x = v => x0 + panelW * (v - xMin) / (xMax - xMin);
      yTicks.forEach(t => {
        svg.append(el('line', {x1: x0, x2: x0 + panelW, y1: y(t), y2: y(t), class: t === 0 && !spec.log ? 'lab-axis' : 'lab-guide'}));
        if (p === 0) svg.append(el('text', {x: x0 - 8, y: y(t) + 4, 'text-anchor': 'end', class: 'lab-tick'},
                                   spec.log ? tickText(+t.toPrecision(1)) : tickText(t)));
      });
      svg.append(el('line', {x1: x0, x2: x0 + panelW, y1: top + plotH, y2: top + plotH, class: 'lab-axis'}));
      xTicks.forEach(t => svg.append(el('text', {x: x(t), y: top + plotH + 18, 'text-anchor': 'middle', class: 'lab-tick'}, tickText(t))));
      if (panel.title) svg.append(el('text', {x: x0, y: top - 10, 'font-weight': 600}, panel.title));
      [...panel.series].sort((a, b) => (a.strong ? 1 : 0) - (b.strong ? 1 : 0)).forEach(s => {
        let d = '';
        s.points.forEach(([px, py], i) => {
          if (i === 0) d += `M${x(px)},${y(py)}`;
          else if (s.step) d += `H${x(px)}V${y(py)}`;
          else d += `L${x(px)},${y(py)}`;
        });
        svg.append(tip(el('path', {d, fill: 'none', stroke: tone(s.tone), 'stroke-width': s.width || 2,
                                   'stroke-linejoin': 'round', 'stroke-linecap': 'round'}), s.label));
        const [ex, ey] = s.points[s.points.length - 1];
        if (s.marker) svg.append(el('circle', {cx: x(ex), cy: y(ey), r: 3.5, fill: tone(s.tone), stroke: 'var(--lab-paper)', 'stroke-width': 1.5}));
        if (spec.end_labels && p === P - 1) ends.push({label: s.label, strong: s.strong, x: x(ex), y: y(ey)});
      });
    });
    if (ends.length) {                                  // names at the ends; one that would overlap waits
      const placed = [], rest = [];
      ends.sort((a, b) => (b.strong ? 1 : 0) - (a.strong ? 1 : 0) || a.y - b.y).forEach(e => {
        if (placed.some(o => Math.abs(o.y - e.y) < 13)) { rest.push(e); return; }
        placed.push(e);
        svg.append(el('text', {x: e.x + 7, y: e.y + 4, class: e.strong ? '' : 'lab-tick', 'font-weight': e.strong ? 600 : 400}, e.label));
      });
      if (rest.length) {
        const low = Math.max(...rest.map(e => e.y)) + 14;
        if (!placed.some(o => Math.abs(o.y - low) < 13))
          svg.append(el('text', {x: rest[0].x + 7, y: low + 4, class: 'lab-tick'}, rest.length > 1 ? (spec.rest_label || 'others') : rest[0].label));
      }
    }
    svg.append(el('text', {x: left + (width - left - right) / 2, y: height - 6, 'text-anchor': 'middle', class: 'lab-tick'}, spec.x_label || ''));
    svg.append(el('text', {x: 12, y: top + plotH / 2, 'text-anchor': 'middle', class: 'lab-tick',
                           transform: `rotate(-90 12 ${top + plotH / 2})`}, spec.y_label || ''));
  }

  const drawers = {comparison: drawComparison, codebook: drawCodebook, lines: drawLines};
  function drawCharts(scope) {
    all('[data-chart]', scope).forEach(box => {
      const spec = data.charts[box.dataset.chart];
      if (spec) drawers[spec.type](box, spec, box.dataset.chart);
    });
  }

  // --- confusion matrices: counts or row %, for the chosen codebook or pipeline ---------------------
  function renderMatrices(key) {
    const group = data.matrices[key], box = q(`[data-matrices="${key}"]`);
    if (!group || !box) return;
    const unit = state.unit[key] || 'count', option = state.select[key] || group.selected;
    const panels = group.panels[option] || [];
    box.style.setProperty('--panels', Math.max(panels.length, 1));
    box.innerHTML = panels.map(p => {
      const n = p.matrix.length, rows = p.matrix.map(r => r.reduce((a, b) => a + b, 0));
      const total = rows.reduce((a, b) => a + b, 0), correct = p.matrix.reduce((a, r, i) => a + r[i], 0);
      const quiet = n > 8 ? ' lab-quiet' : '';
      const cells = ['<span></span>', ...p.matrix.map((_, j) => `<span class="lab-matrix-label" role="columnheader">${j}</span>`)];
      p.matrix.forEach((row, i) => {
        cells.push(`<span class="lab-matrix-label lab-matrix-row" role="rowheader">${i}</span>`);
        row.forEach((v, j) => {
          const share = rows[i] ? 100 * v / rows[i] : 0;
          const shown = unit === 'percent' ? `${share.toFixed(1)}%` : v;
          const kind = v === 0 ? 'lab-zero' : i === j ? 'lab-correct' : 'lab-error';
          const shade = i === j ? Math.max(75, share) : Math.min(26, 7 + share);
          cells.push(`<span class="lab-matrix-cell ${kind}${quiet}" style="--lab-shade:${shade}%" role="cell" title="Sent ${i}, predicted ${j}: ${v} of ${rows[i]}">${shown}</span>`);
        });
      });
      return `<figure class="lab-matrix-item"><h3>${p.name}</h3><p class="lab-matrix-stat">${(100 * correct / total).toFixed(1)}% accuracy · ${(total - correct).toLocaleString('en-US')} errors</p>`
        + `<div class="lab-matrix-axis">Predicted symbol</div><div class="lab-matrix-grid" style="--n:${n}" role="table" aria-label="${p.name} confusion matrix">${cells.join('')}</div></figure>`;
    }).join('');
    all(`[data-unit-for="${key}"] button`).forEach(b => b.setAttribute('aria-pressed', String(b.dataset.labUnit === unit)));
  }

  // --- tabs ------------------------------------------------------------------------------------------
  function renderTab() {
    all('[data-lab-tab]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.labTab === state.tab)));
    tabs.forEach(id => { q(`#lab-${id}`).hidden = id !== state.tab; });
    const source = q('#lab-source-label');
    if (source && data.sources[state.tab]) source.textContent = `Source: ${data.sources[state.tab]}`;
    drawCharts(q(`#lab-${state.tab}`));
  }
  root.addEventListener('click', event => {
    const tab = event.target.closest('[data-lab-tab]'), unit = event.target.closest('[data-lab-unit]');
    if (tab) { state.tab = tab.dataset.labTab; history.replaceState(null, '', `#${state.tab}`); renderTab(); }
    if (unit) { const key = unit.closest('[data-unit-for]').dataset.unitFor; state.unit[key] = unit.dataset.labUnit; renderMatrices(key); }
  });
  root.addEventListener('change', event => {
    const select = event.target.closest('[data-select-for]'), changes = event.target.closest('[data-changes-for]');
    if (select) {
      const key = select.dataset.selectFor;
      state.select[key] = select.value;
      renderMatrices(key);
      all(`[data-show-for="${key}"]`).forEach(node => { node.hidden = node.dataset.value !== select.value; });
    }
    if (changes) { state.changes[changes.dataset.changesFor] = changes.checked; drawCharts(q(`#lab-${state.tab}`)); }
  });
  q('.lab-tabs').addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    const buttons = all('[data-lab-tab]'), index = buttons.indexOf(document.activeElement);
    if (index < 0) return;
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1
      : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
    event.preventDefault(); buttons[next].focus(); buttons[next].click();
  });
  window.addEventListener('beforeprint', () => { root.classList.add('lab-printing'); drawCharts(root); });
  window.addEventListener('afterprint', () => { root.classList.remove('lab-printing'); renderTab(); });
  Object.keys(data.matrices).forEach(renderMatrices);
  renderTab();
  let lastWidth = 0;
  new ResizeObserver(entries => {
    const width = entries[0].contentRect.width;
    if (Math.abs(width - lastWidth) < 1) return;
    lastWidth = width;
    drawCharts(q(`#lab-${state.tab}`));
  }).observe(root);
})();
