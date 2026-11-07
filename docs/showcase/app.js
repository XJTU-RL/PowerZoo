"use strict";

(() => {
  const root = document.body.dataset.showcaseRoot;
  const $ = (id) => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  const controls = ["load", "tap", "capacitors", "phase", "bus", "play", "reset"].map($);
  const state = { data: null, selected: "675", timer: null };
  const phaseNames = { 1: "A", 2: "B", 3: "C" };
  const palette = { low: "#b64b26", normal: "#087e86", high: "#79498b", missing: "#b5c1c8" };

  function svg(tag, attrs = {}, text) {
    const element = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function finite(value) { return typeof value === "number" && Number.isFinite(value); }
  function phaseValue(snapshot, id, phase) {
    const values = snapshot.voltages[id] || snapshot.voltages[id.toLowerCase()];
    const value = values && values[String(phase)];
    return finite(value) ? value : null;
  }
  function colour(value) {
    if (value === null) return palette.missing;
    return value < 0.95 ? palette.low : value > 1.05 ? palette.high : palette.normal;
  }
  function currentSnapshot() {
    const load = Number($("load").value);
    const tap = Number($("tap").value);
    const cap = Number($("capacitors").value);
    return state.data.snapshots.find((row) => Math.abs(row.load_mult - load) < 0.00001 && row.tap_step === tap && Number(row.capacitors) === cap);
  }
  function stopPlayback() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = null;
    $("play").innerHTML = '<span class="play-icon" aria-hidden="true">▷</span>播放负荷扫掠';
    $("play").setAttribute("aria-pressed", "false");
  }
  function showError(message) {
    stopPlayback();
    $("load-error").hidden = false;
    $("load-error").textContent = "工况数据无法加载：" + message + "。请刷新页面，或从仓库查看数据与复现说明。";
    $("data-badge").textContent = "数据不可用";
    $("bench-body").setAttribute("aria-busy", "false");
    controls.forEach((element) => { element.disabled = true; });
    $("snapshot-status").textContent = "未显示替代或生成数据";
  }
  function validate(data) {
    if (!data || !Array.isArray(data.nodes) || !Array.isArray(data.edges) || !Array.isArray(data.snapshots)) throw new Error("数据结构不完整");
    if (data.snapshots.length !== 54 || !data.nodes.length) throw new Error("工况数量与展示约定不一致");
    const ids = new Set(data.nodes.map((node) => node.id));
    if (ids.size !== data.nodes.length) throw new Error("母线标识重复");
    data.nodes.forEach((node) => {
      if (typeof node.id !== "string" || !finite(node.x) || !finite(node.y)) throw new Error("母线坐标无效");
    });
    data.edges.forEach((edge) => {
      if (!ids.has(edge.source) || !ids.has(edge.target)) throw new Error("拓扑连接无效");
    });
    const keys = new Set();
    data.snapshots.forEach((row) => {
      if (![row.load_mult, row.tap_step, row.min_pu, row.max_pu, row.loss_kw, row.grid_kw, row.balance_error_kw].every(finite) || !row.voltages) throw new Error("工况量测无效");
      if (Math.abs(row.balance_error_kw) >= 0.05) throw new Error("工况功率平衡校验未通过");
      const key = [row.load_mult.toFixed(1), row.tap_step, Number(row.capacitors)].join("|");
      if (keys.has(key)) throw new Error("工况键重复");
      keys.add(key);
      Object.values(row.voltages).forEach((phases) => {
        Object.entries(phases).forEach(([phase, voltage]) => {
          if (!["1", "2", "3"].includes(phase) || !finite(voltage)) throw new Error("相电压数据无效");
        });
      });
    });
    for (let step = 6; step <= 14; step += 1) for (const tap of [-4, 0, 4]) for (const cap of [0, 1]) {
      if (!keys.has([(step / 10).toFixed(1), tap, cap].join("|"))) throw new Error("存在缺失工况");
    }
  }
  function renderNetwork(snapshot, phase) {
    const network = $("network");
    network.replaceChildren();
    const positions = new Map(state.data.nodes.map((node) => [node.id, { x: 75 + node.x * 630, y: 55 + (1 - node.y) * 350 }]));
    state.data.edges.forEach((edge) => {
      const a = positions.get(edge.source), b = positions.get(edge.target);
      const line = svg("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, class: "network-edge" + (String(edge.element).toLowerCase().startsWith("transformer.") ? " transformer" : "") });
      line.append(svg("title", {}, edge.element));
      network.append(line);
    });
    state.data.nodes.forEach((node) => {
      const pos = positions.get(node.id);
      const value = phaseValue(snapshot, node.id, phase);
      const group = svg("g", { class: "node" + (state.selected === node.id ? " selected" : ""), transform: `translate(${pos.x},${pos.y})`, tabindex: "0", role: "button", "data-node-id": node.id, "aria-label": `母线${node.id}，${phaseNames[phase]}相${value === null ? "缺相" : value.toFixed(4) + "标幺"}` });
      group.append(svg("circle", { r: 14, class: "node-ring" }));
      group.append(svg("circle", { r: 10, class: "node-circle", fill: colour(value) }));
      group.append(svg("text", { x: 18, y: -5, class: "node-label" }, node.id));
      group.append(svg("text", { x: 18, y: 13, class: "node-measurement" }, value === null ? "缺相" : value.toFixed(3)));
      const select = (keepFocus = false) => { state.selected = node.id; render(); if (keepFocus) [...network.querySelectorAll("[data-node-id]")].find((el) => el.dataset.nodeId === node.id)?.focus({ preventScroll: true }); };
      group.addEventListener("click", () => select());
      group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(true); } });
      network.append(group);
    });
  }
  function renderProfile(snapshot, phase) {
    const chart = $("profile");
    chart.replaceChildren();
    const values = state.data.nodes.map((node) => ({ id: node.id, value: phaseValue(snapshot, node.id, phase) }));
    const valid = values.map((row) => row.value).filter((value) => value !== null);
    const low = Math.min(0.85, Math.floor(Math.min(...valid) * 20) / 20);
    const high = Math.max(1.10, Math.ceil(Math.max(...valid) * 20) / 20);
    const y = (value) => 20 + (high - value) / (high - low) * 150;
    chart.append(svg("rect", { x: 54, y: y(1.05), width: 1040, height: y(0.95) - y(1.05), fill: "#e2f1ef" }));
    [low, 0.95, 1.0, 1.05, high].filter((value, index, arr) => arr.indexOf(value) === index).forEach((value) => {
      chart.append(svg("line", { x1: 54, y1: y(value), x2: 1094, y2: y(value), stroke: "#d5e2e8", "stroke-width": 1, "stroke-dasharray": "3 4" }));
      chart.append(svg("text", { x: 45, y: y(value) + 4, "text-anchor": "end", class: "profile-axis" }, value.toFixed(2)));
    });
    const spacing = 1040 / values.length;
    values.forEach(({ id, value }, index) => {
      const x = 54 + spacing * (index + .5);
      if (value !== null) {
        chart.append(svg("rect", { x: x - Math.min(13, spacing / 4), y: y(value), width: Math.min(26, spacing / 2), height: Math.max(1, y(low) - y(value)), rx: 2, fill: colour(value), opacity: .88 }));
        chart.append(svg("text", { x, y: y(value) - 7, "text-anchor": "middle", class: "profile-value" }, value.toFixed(3)));
      } else chart.append(svg("text", { x, y: 160, "text-anchor": "middle", class: "profile-value" }, "缺相"));
      chart.append(svg("text", { x, y: 193, "text-anchor": "middle", class: "profile-label" }, id));
    });
    chart.append(svg("text", { x: 54, y: 220, class: "profile-axis" }, phaseNames[phase] + " 相  ·  电压单位 p.u.  ·  含源侧及调压器辅助母线"));
  }
  function metric(id, value, digits, unit) {
    $(id).replaceChildren(document.createTextNode(value.toFixed(digits)), Object.assign(document.createElement("small"), { textContent: unit }));
  }
  function render() {
    const snapshot = currentSnapshot();
    if (!snapshot) { showError("所选工况不存在"); return; }
    const phase = $("phase").value;
    $("load-value").textContent = Number($("load").value).toFixed(1) + "×";
    $("phase-title").textContent = phaseNames[phase] + " 相电压";
    metric("min-voltage", snapshot.min_pu, 4, "p.u.");
    metric("max-voltage", snapshot.max_pu, 4, "p.u.");
    metric("loss", snapshot.loss_kw, 2, "kW");
    metric("grid", snapshot.grid_kw, 2, "kW");
    $("snapshot-status").textContent = `有功平衡残差 ${Math.abs(snapshot.balance_error_kw).toExponential(2)} kW。统计覆盖全部实际相节点。`;
    $("selected-node").textContent = state.selected;
    $("bus").value = state.selected;
    ["a", "b", "c"].forEach((name, index) => {
      const value = phaseValue(snapshot, state.selected, index + 1);
      $("node-" + name).textContent = value === null ? "缺相" : value.toFixed(4) + " p.u.";
    });
    renderNetwork(snapshot, phase);
    renderProfile(snapshot, phase);
  }
  ["load", "tap", "capacitors", "phase"].forEach((id) => {
    $(id).addEventListener(id === "load" ? "input" : "change", () => { stopPlayback(); render(); });
  });
  $("bus").addEventListener("change", () => { state.selected = $("bus").value; render(); });
  $("play").addEventListener("click", () => {
    if (state.timer) { stopPlayback(); return; }
    $("play").innerHTML = '<span class="play-icon" aria-hidden="true">Ⅱ</span>暂停负荷扫掠';
    $("play").setAttribute("aria-pressed", "true");
    state.timer = window.setInterval(() => {
      let step = Math.round(Number($("load").value) * 10) + 1;
      if (step > 14) step = 6;
      $("load").value = String(step / 10);
      render();
    }, 1100);
  });
  $("reset").addEventListener("click", () => {
    stopPlayback(); $("load").value = "1"; $("tap").value = "0"; $("capacitors").value = "1"; $("phase").value = "1"; state.selected = state.data.nodes.some((node) => node.id === "675") ? "675" : state.data.nodes[0].id; render();
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopPlayback(); });
  fetch(root + "data/vvc-13bus.json")
    .then((response) => { if (!response.ok) throw new Error("HTTP " + response.status); return response.json(); })
    .then((data) => {
      validate(data); state.data = data;
      data.nodes.forEach((node) => { const option = document.createElement("option"); option.value = node.id; option.textContent = node.id; $("bus").append(option); });
      if (!data.nodes.some((node) => node.id === state.selected)) state.selected = data.nodes[0].id;
      controls.forEach((element) => { element.disabled = false; });
      $("data-badge").textContent = `${data.snapshots.length}个原生求解工况 · ${data.nodes.length}个母线`;
      $("bench-body").setAttribute("aria-busy", "false"); render();
    })
    .catch((error) => showError(error.message || "未知读取错误"));
})();
