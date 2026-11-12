// Builds the results charts in docs/figures/ (light and dark SVGs, no dependencies):
//   node docs/figures/src/charts.mjs
// Numbers are the ones reported in RESULTS.md (sources: scripts/bench/results/, scripts/skill_eval/results/).
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..");
const W = 760;

// One hue, two steps (before = lighter step, after = accent), validated per theme with the ordinal check.
const THEMES = {
  light: { surface: "#fcfcfb", border: "rgba(11,11,11,.10)", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7", before: "#86b6ef", after: "#2a78d6", good: "#006300" },
  dark: { surface: "#1a1a19", border: "rgba(255,255,255,.10)", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835", before: "#184f95", after: "#3987e5", good: "#0ca30c" },
};
const css = (t) => `
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:${t.ink2};font-size:12px}
.bg{fill:${t.surface};stroke:${t.border}}
.h{font-size:15.5px;font-weight:600;fill:${t.ink}}.sub{font-size:12.5px;fill:${t.ink2}}
.kl{font-size:11.5px;fill:${t.muted}}.kv{font-size:27px;font-weight:600;fill:${t.ink}}.kd{font-size:12px;fill:${t.ink2}}.up{fill:${t.good};font-weight:600}
.row{font-size:12.5px;fill:${t.ink}}.rowb{font-weight:600}.note{fill:${t.muted};font-size:11.5px}
.tick{font-size:11px;fill:${t.muted};font-variant-numeric:tabular-nums}.val{font-size:11.5px;fill:${t.ink};font-variant-numeric:tabular-nums}.val2{fill:${t.ink2}}
.grid{stroke:${t.grid};stroke-width:1}.axis{stroke:${t.axis};stroke-width:1}
.link{stroke:${t.before};stroke-width:2;stroke-linecap:round}.ci{stroke-width:2;stroke-linecap:round}
.dot{stroke:${t.surface};stroke-width:2}.b{fill:${t.before}}.a{fill:${t.after}}.g{fill:${t.muted}}.sa{stroke:${t.after}}.sg{stroke:${t.muted}}
.start{fill:${t.surface};stroke:${t.muted};stroke-width:1.5}
.tile{fill:none;stroke:${t.grid}}`;

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const T = (x, y, s, cls = "", anchor = "") => `<text x="${x}" y="${y}"${cls ? ` class="${cls}"` : ""}${anchor ? ` text-anchor="${anchor}"` : ""}>${esc(s)}</text>`;
const f3 = (v) => v.toFixed(3);
const signed = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(3);

function frame(height, title, subtitle, body, label) {
  return (t) => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${height}" role="img" aria-label="${esc(label)}"><style>${css(t)}</style>` +
    `<rect class="bg" x=".5" y=".5" width="${W - 1}" height="${height - 1}" rx="10"/>${T(24, 34, title, "h")}${T(24, 54, subtitle, "sub")}${body}</svg>`;
}
function tiles(y, list) {
  const w = (W - 48 - 16 * (list.length - 1)) / list.length;
  return list.map(([label, value, delta, up], i) => {
    const x = 24 + i * (w + 16);
    return `<rect class="tile" x="${x}" y="${y}" width="${w}" height="74" rx="8"/>${T(x + 14, y + 21, label, "kl")}${T(x + 14, y + 50, value, "kv")}` +
      `<text x="${x + 14}" y="${y + 66}" class="kd">${up ? `<tspan class="up">${esc(up)}</tspan> ` : ""}${esc(delta)}</text>`;
  }).join("");
}
function legend(xRight, y, items) { // items: [class, label]; laid out right to left
  let x = xRight, out = "";
  for (const [cls, label] of [...items].reverse()) {
    const width = label.length * 6.3;
    x -= width;
    out += T(x, y + 4, label, "note");
    x -= 10;
    out += `<circle class="${cls}" cx="${x}" cy="${y}" r="${cls === "start" ? 4 : 5}"/>`;
    x -= 18;
  }
  return out;
}
function scale(x0, x1, v0, v1) { return (v) => x0 + ((v - v0) / (v1 - v0)) * (x1 - x0); }
function axis(sx, ticks, top, bottom, fmt = (v) => v.toFixed(1)) {
  return ticks.map((v) => `<line class="grid" x1="${sx(v)}" x2="${sx(v)}" y1="${top}" y2="${bottom}"/>${T(sx(v), bottom + 16, fmt(v), "tick", "middle")}`).join("");
}
// before -> after on one row: lighter dot, darker dot, a link between them, the lower value labelled on its left
function dumbbell(sx, y, before, after, start = null) {
  let out = "";
  if (start !== null) out += `<circle class="start" cx="${sx(start)}" cy="${y}" r="4"/>${T(sx(start) + 9, y + 4, start.toFixed(2), "val val2")}`;
  if (Math.abs(after - before) < 1e-9) return out + `<circle class="dot a" cx="${sx(after)}" cy="${y}" r="5.5"/>${T(sx(after) - 11, y + 4, `no change  ${f3(after)}`, "val", "end")}`;
  const lo = Math.min(before, after), hi = Math.max(before, after);
  out += `<line class="link" x1="${sx(before)}" x2="${sx(after)}" y1="${y}" y2="${y}"/>`;
  out += `<circle class="dot b" cx="${sx(before)}" cy="${y}" r="5.5"/><circle class="dot a" cx="${sx(after)}" cy="${y}" r="5.5"/>`;
  const cls = (v) => (v === after ? "val" : "val val2");
  out += T(sx(lo) - 11, y + 4, f3(lo), cls(lo), "end") + T(sx(hi) + 11, y + 4, f3(hi), cls(hi));
  return out;
}

// ---- 1. engine: original vs new on held-out test ---------------------------------------------
function engine() {
  const rows = [["parse_dates", "write a date parser", 0.333, 0.822, 0.844], ["regex_emails", "write a regex", 0.5, 1.0, 0.989], ["ticket_router", "two-part routing rules", 0.444, 0.8, 0.833], ["Mean of the three", "", 0.426, 0.874, 0.889]];
  const sx = scale(196, 690, 0.3, 1.0), top = 190, rowH = 38, bottom = top + rows.length * rowH;
  let body = tiles(74, [
    ["Held-out test score, new engine", "0.889", "vs 0.874 for the original", "+0.015"],
    ["Cost per run", "0.73×", "than the original engine", "27% lower"],
    ["Chance the score gain is real", "95%", "95% interval −0.003 to +0.031", ""],
  ]);
  body += T(24, 174, "Held-out test score by task (higher is better)", "row rowb") + legend(W - 24, 170, [["start", "Starting text"], ["dot b", "Original engine"], ["dot a", "New engine"]]);
  body += axis(sx, [0.4, 0.6, 0.8, 1.0], top, bottom);
  rows.forEach(([name, what, start, before, after], i) => {
    const y = top + i * rowH + rowH / 2;
    if (i === rows.length - 1) body += `<line class="axis" x1="24" x2="${W - 24}" y1="${y - rowH / 2}" y2="${y - rowH / 2}"/>`;
    body += T(24, what ? y - 1 : y + 4, name, i === rows.length - 1 ? "row rowb" : "row") + (what ? T(24, y + 13, what, "note") : "");
    body += dumbbell(sx, y, before, after, start);
  });
  return frame(bottom + 34, "The new engine scores slightly higher and costs about a quarter less",
    "Same budget of 300 evaluator calls per run. 5 seeds per task for the original, 10 for the new engine.", body,
    "Held-out test score by task for the starting text, the original engine and the new engine. Mean 0.874 for the original and 0.889 for the new engine, at 0.73 times the cost.");
}

// ---- 2. every upgrade, measured against the original engine ----------------------------------
function upgrades() {
  const rows = [
    ["Core fixes: GEPA pruning, eval cache", 0.013, -0.028, 0.044, "On by default", true],
    ["Core fixes + statistical winner check", 0.015, -0.003, 0.031, "On by default", true],
    ["… + successive halving", 0.017, -0.004, 0.037, "Off: no extra gain", false],
    ["… + memory of rejected edits", 0.011, -0.028, 0.043, "Off: no extra gain", false],
    ["… + bandit over edit types", 0.002, -0.015, 0.019, "Off: no extra gain", false],
    ["… + counterexample mining", -0.02, -0.046, 0.007, "Off: scored lower", false],
    ["Section edits only", -0.224, -0.265, -0.181, "Rejected", false],
    ["… + cap on edit size", -0.3, -0.324, -0.28, "Rejected", false],
  ];
  const sx = scale(268, 560, -0.35, 0.08), top = 96, rowH = 31, bottom = top + rows.length * rowH;
  let body = T(24, 82, "Upgrade", "note") + T(584, 82, "Change", "note") + T(640, 82, "Decision", "note");
  body += axis(sx, [-0.3, -0.2, -0.1], top, bottom, (v) => `−${Math.abs(v).toFixed(1)}`) + `<line class="axis" x1="${sx(0)}" x2="${sx(0)}" y1="${top}" y2="${bottom}"/>${T(sx(0), bottom + 16, "0", "tick", "middle")}`;
  rows.forEach(([name, d, lo, hi, decision, on], i) => {
    const y = top + i * rowH + rowH / 2, c = on ? "a" : "g";
    body += T(24, y + 4, name, on ? "row rowb" : "row");
    body += `<line class="ci s${c}" x1="${sx(lo)}" x2="${sx(hi)}" y1="${y}" y2="${y}"/><circle class="dot ${c}" cx="${sx(d)}" cy="${y}" r="5.5"/>`;
    body += T(584, y + 4, signed(d), "val") + T(640, y + 4, decision, on ? "row rowb" : "row");
  });
  body += T(24, bottom + 38, "Dot = mean change over 3 tasks; line = 95% bootstrap interval. “…” rows add one idea on top of the winner check.", "note");
  return frame(bottom + 54, "Every upgrade was measured before it became a default",
    "Change in held-out test score against the original engine. Only what held up is switched on.", body,
    "Change in held-out test score for eight engine upgrades against the original engine, with 95 percent intervals. Core fixes and the statistical winner check are on by default; section-only edits and an edit-size cap were rejected.");
}

// ---- 3. skill: before vs after, by part ------------------------------------------------------
function skill() {
  const rows = [["Picks the skill correctly", "trigger F1", [0.97, 1.0], [1.0, 1.0]], ["Writes a call that runs", "validity", [0.75, 0.95], [0.8, 1.0]], ["Call quality", "judged against a rubric", [0.718, 0.882], [0.764, 0.914]], ["Total score", "", [0.831, 0.95], [0.869, 0.974]]];
  const panels = [["Training examples (70%)", scale(214, 396, 0.6, 1.0), 2], ["Held-out examples (30%)", scale(504, 686, 0.6, 1.0), 3]];
  const top = 212, rowH = 38, bottom = top + rows.length * rowH;
  let body = tiles(74, [
    ["Held-out score of the skill", "0.974", "from 0.869 before", "+0.105"],
    ["Calls that run, held-out", "100%", "of requests, from 80% before", ""],
    ["Cost of the whole run", "$6.68", "8 rounds, 17 minutes", ""],
  ]);
  body += T(24, 174, "Score by part (higher is better)", "row rowb") + legend(W - 24, 170, [["dot b", "Original skill"], ["dot a", "Optimized skill"]]);
  for (const [title, sx] of panels) body += T(sx(0.6), 198, title, "row rowb") + axis(sx, [0.6, 0.7, 0.8, 0.9, 1.0], top, bottom);
  rows.forEach(([name, what, train, val], i) => {
    const y = top + i * rowH + rowH / 2;
    if (i === rows.length - 1) body += `<line class="axis" x1="24" x2="${W - 24}" y1="${y - rowH / 2}" y2="${y - rowH / 2}"/>`;
    body += T(24, what ? y - 1 : y + 4, name, i === rows.length - 1 ? "row rowb" : "row") + (what ? T(24, y + 13, what, "note") : "");
    body += dumbbell(panels[0][1], y, train[0], train[1]) + dumbbell(panels[1][1], y, val[0], val[1]);
  });
  return frame(bottom + 34, "The engine improved its own skill, and the gain holds on examples it never saw",
    "The skill teaches coding agents when and how to call the optimizer. It was optimized by the engine itself.", body,
    "Skill score before and after optimization, on training and held-out examples, by part. Held-out total rose from 0.869 to 0.974.");
}

for (const [name, build] of [["engine", engine()], ["upgrades", upgrades()], ["skill", skill()]]) {
  for (const theme of ["light", "dark"]) {
    writeFileSync(join(OUT, `results-${name}-${theme}.svg`), build(THEMES[theme]));
  }
  console.log("wrote", `results-${name}-{light,dark}.svg`);
}
