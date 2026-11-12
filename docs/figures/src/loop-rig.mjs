// Builds docs/figures/loop-{light,dark}.svg: the optimization loop drawn as a small proving bench.
// Needs the "anatomy" isometric kit:  ANATOMY_KIT=/path/to/anatomy/kit node docs/figures/src/loop-rig.mjs
// The SVGs are self-contained (inline CSS, CSS-only animation), so GitHub renders them from <img>.
import { writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const KIT = process.env.ANATOMY_KIT || join(homedir(), ".claude/skills/anatomy/kit");
const k = await import(pathToFileURL(join(KIT, "iso-kit.mjs")).href);
const { tubePieces, tubeSvg } = await import(pathToFileURL(join(KIT, "tube.mjs")).href);
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..");

// ---- truth: the real skill hill-climb (RESULTS.md) -------------------------------------------
const SCORE_BEFORE = 0.869; // held-out score of the original skill
const SCORE_AFTER = 0.974; // held-out score of the shipped skill

// ---- world ------------------------------------------------------------------------------------
const W = 760, H = 376;
const PLATE = { x: -170, y: -62, w: 340, d: 124, r: 8 };
const PLATE_H = 10, FOOT_H = 5;
const RACK_X = -122, BRIDGE_X = -48, HEAD_X = 30, GAUGE_X = 100;
const CARD_TOP = 8.6;
const PROBE_LIFT = 11.4, STYLUS_DROP = 9.4, GAUGE_DROP = 12;
const PROBES_Y = [-10, -5, 0, 5, 10];
const DIAL = [BRIDGE_X, -30], DIAL_R = 12.5, DIAL_TOP = 64;
const SWEEP_FROM = (-150 * Math.PI) / 180, SWEEP = (300 * Math.PI) / 180;
const CABLE_A = [BRIDGE_X, 32, 61.5], CABLE_B = [HEAD_X, 0, 62.5];

const cablePoints = Array.from({ length: 61 }, (_, i) => {
  const t = i / 60;
  return [CABLE_A[0] + (CABLE_B[0] - CABLE_A[0]) * t, CABLE_A[1] + (CABLE_B[1] - CABLE_A[1]) * t,
    CABLE_A[2] + (CABLE_B[2] - CABLE_A[2]) * t + 20 * Math.sqrt(Math.max(0, 4 * t * (1 - t)))];
});

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [HEAD_X, -46, 80], [DIAL[0] - DIAL_R, DIAL[1], DIAL_TOP], ...cablePoints],
  W, H, { pad: 22, azimuth: 66 },
);

const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const slab = (plan, z, h, steps = 3, bevel = 0) => k.slabOf({ r: 1, ...plan }, z, h, P, steps, bevel);
const cyl = (x, y, r, z, h, steps = 20, bevel = 0) => k.cylinder(x, y, r, z, h, P, steps, bevel);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const move = (v) => k.translateAlong(P, v);
const px = (v) => { const m = /translate\(([-\d.]+)[ ,]+([-\d.]+)\)/.exec(move(v)); return [Number(m[1]), Number(m[2])]; };

// ---- static parts -----------------------------------------------------------------------------
function base() {
  const o = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of byDepth([...k.corners(PLATE, 22), [0, PLATE.y + 22], [0, PLATE.y + PLATE.d - 22]])) {
    o.push(S(cyl(x, y, 7, -PLATE_H - FOOT_H, 1.6, 28), { tone: "lo" }));
    o.push(S(cyl(x, y, 5.4, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, 28), { tone: "mid" }));
  }
  o.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  o.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 10);
  o.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  o.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // ruler strip along the front edge
  const strip = { x: -150, y: 50, w: 300, d: 6.5, r: 1 };
  o.push(S(slab(strip, 0, 0.6), { tone: "lo", crease: "none" }));
  const t = k.topTicks(strip.x + 5, strip.x + strip.w - 5, 5, 5, strip.y + 0.8, 0.6, [2, 3.6], P);
  o.push(L(t.minor, { tone: "lo" }), L(t.major, { tone: "mid" }));
  o.push(D([at([strip.x + 2, strip.y + 3.2, 0.6]), at([strip.x + strip.w - 2, strip.y + 3.2, 0.6])], { size: 0.45 }));
  // maker's plate
  const lab = { x: -164, y: 32, w: 26, d: 13, r: 1.2 };
  o.push(S(slab(lab, 0, 0.8, 4), { tone: "lo", crease: "none" }));
  o.push(L(k.planOutline(k.insetPlan(lab, 1.6), 0.8, P), { tone: "faint" }));
  o.push(L([[4, 15], [7, 11], [10, 17]].map(([dy, w]) => k.lineOnTop([lab.x + 4, lab.y + dy], [lab.x + 4 + w, lab.y + dy], 0.8, P)).join(""), { tone: "lo", free: true }));
  o.push(D([at([lab.x + 2.2, lab.y + 2.2, 0.8]), at([lab.x + lab.w - 2.2, lab.y + lab.d - 2.2, 0.8])], { size: 0.4 }));
  // station plates, stamped with what happens there
  [[RACK_X, "1 POOL"], [BRIDGE_X, "2 SCORE"], [HEAD_X, "3 REWRITE"], [GAUGE_X, "4 KEEP?"]].forEach(([sx, word]) => {
    const plan = { x: sx + 8, y: 35.5, w: 33, d: 11, r: 1.2 };
    o.push(S(slab(plan, 0, 0.8, 4), { tone: "mid", crease: "none" }));
    o.push(k.faceTextSvg(k.topMatrix([plan.x + plan.w / 2, plan.y + 7.6], 0.8, P, "x"), word, { size: 5.4, tone: "hi", anchor: "middle" }));
    o.push(D([at([plan.x + 1.6, plan.y + 1.6, 0.8]), at([plan.x + plan.w - 1.6, plan.y + 1.6, 0.8])], { size: 0.35 }));
  });
  // budget knob: a tick ring with a knurled knob and pointer
  const [bx, by] = [154, 40];
  o.push(S(cyl(bx, by, 9.5, 0, 0.6, 48), { tone: "lo", crease: "none" }));
  const bt = k.radialTicks(bx, by, 9, 40, 5, 0.6, [1.2, 2.4], P);
  o.push(L(bt.minor, { tone: "lo" }), L(bt.major, { tone: "mid" }));
  o.push(S(cyl(bx, by, 5, 0.6, 4.4, 32), { tone: "mid" }));
  o.push(L(k.knurl(bx, by, 5, 0.6, 5, 32, P), { tone: "lo" }));
  o.push(L(k.lineOnTop([bx, by], [bx + 3.6, by - 3], 5, P), { tone: "hi" }));
  o.push(D([at([bx, by, 5])], { size: 0.5, tone: "hi" }));
  return o.join("");
}

function rails() {
  const o = [];
  for (const y of [-19, 14]) {
    o.push(S(slab({ x: -150, y, w: 300, d: 5 }, 0, 3, 3, 0.4), { tone: "mid" }));
    o.push(D([-144, -74, -4, 66, 136].map((x) => at([x, y + 2.5, 3])), { size: 0.4 }));
  }
  for (const x of [-156, 150]) {
    o.push(S(slab({ x, y: -23, w: 6, d: 46, r: 1.2 }, 0, 7, 3, 0.5), { tone: "mid" }));
    o.push(D([at([x + 3, -18, 7]), at([x + 3, 18, 7])], { size: 0.45 }));
  }
  return o.join("");
}

function rack() {
  const o = [];
  const plan = { x: -148, y: -58, w: 52, d: 30, r: 2 };
  o.push(S(slab(plan, 0, 5, 4, 0.5), { tone: "mid" }));
  o.push(S(slab({ x: plan.x, y: plan.y, w: 2.4, d: plan.d, r: 0.8 }, 5, 14, 2), { tone: "mid" }));
  for (let i = 0; i < 6; i++) {
    const x = -142 + i * 7.2;
    if (i === 3) { o.push(L(k.lineOnTop([x + 0.8, plan.y], [x + 0.8, plan.y + plan.d], 5, P), { tone: "lo" })); continue; } // the empty slot
    o.push(S(slab({ x, y: -55, w: 1.6, d: 24, r: 0.3 }, 5, 24, 2), { tone: "hi", crease: "lo" }));
    o.push(S(slab({ x, y: -55 + i * 3.8, w: 1.6, d: 5, r: 0.3 }, 29, 2.6, 2), { tone: "hi", crease: "lo" })); // index tab
  }
  o.push(S(slab({ x: plan.x + plan.w - 2.4, y: plan.y, w: 2.4, d: plan.d, r: 0.8 }, 5, 14, 2), { tone: "mid" }));
  o.push(D([at([plan.x + 1.2, plan.y + 4, 19]), at([plan.x + 1.2, plan.y + plan.d - 4, 19]), at([plan.x + plan.w - 1.2, plan.y + 4, 19]), at([plan.x + plan.w - 1.2, plan.y + plan.d - 4, 19])], { size: 0.4 }));
  return o.join("");
}

function behind() {
  const o = [];
  // probe bridge: back post
  o.push(S(cyl(BRIDGE_X, -38, 5, 0, 3, 28), { tone: "mid" }));
  o.push(S(cyl(BRIDGE_X, -38, 3.2, 3, 43, 28), { tone: "mid", crease: "none" }));
  o.push(L(k.sideArc(BRIDGE_X, -38, 3.2, 24, P, 16), { tone: "faint" }));
  // engraving head: magnetic base, knob, post, clamp, thumb screw
  const mag = { x: HEAD_X - 16, y: -58, w: 32, d: 24, r: 3 };
  o.push(S(slab(mag, 0, 10, 6, 0.8), { tone: "mid" }));
  o.push(S(slab(k.insetPlan(mag, 0.8), 10, 7, 6, 0.8), { tone: "mid" }));
  o.push(L(k.planOutline(k.insetPlan(mag, 3), 17, P), { tone: "faint" }));
  o.push(D([at([mag.x + 4, mag.y + 4, 17]), at([mag.x + mag.w - 4, mag.y + 4, 17]), at([mag.x + 4, mag.y + mag.d - 4, 17])], { size: 0.45 }));
  const knob = [HEAD_X + 10.5, -39.5];
  o.push(S(cyl(knob[0], knob[1], 4.4, 17, 4, 28), { tone: "mid" }));
  o.push(L(k.knurl(knob[0], knob[1], 4.4, 17, 21, 28, P), { tone: "lo" }));
  o.push(L(k.lineOnTop(knob, [knob[0] + 4.4, knob[1]], 21, P), { tone: "hi" }));
  o.push(D([at([knob[0], knob[1], 21])], { size: 0.5, tone: "hi" }));
  const post = [HEAD_X, -46];
  o.push(S(cyl(post[0], post[1], 5, 17, 3, 28), { tone: "mid" }));
  o.push(S(cyl(post[0], post[1], 3.2, 20, 30, 28), { tone: "mid", crease: "none" }));
  o.push(L(k.sideArc(post[0], post[1], 3.2, 36, P, 16), { tone: "faint" }));
  const clamp = { x: post[0] - 6.5, y: post[1] - 6.5, w: 13, d: 13, r: 2.5 };
  o.push(S(slab(clamp, 50, 12, 4, 0.6), { tone: "mid" }));
  o.push(S(cyl(post[0], post[1], 3.2, 62, 16, 28), { tone: "mid", crease: "none" }));
  o.push(S(cyl(post[0], post[1], 3.8, 78, 2, 28), { tone: "mid" }));
  o.push(S(slab({ x: clamp.x + clamp.w, y: post[1] - 1.4, w: 3, d: 2.8, r: 0.8 }, 53.5, 4, 2), { tone: "mid", crease: "none" }));
  o.push(S(cyl(clamp.x + clamp.w + 4.4, post[1], 2.6, 52.5, 6, 20), { tone: "mid" }));
  o.push(L(k.knurl(clamp.x + clamp.w + 4.4, post[1], 2.6, 52.5, 58.5, 18, P), { tone: "lo" }));
  // height gauge: base and column
  const gb = { x: GAUGE_X - 12, y: -59, w: 24, d: 22, r: 2.5 };
  o.push(S(slab(gb, 0, 8, 5, 0.7), { tone: "mid" }));
  o.push(L(k.planOutline(k.insetPlan(gb, 2.6), 8, P), { tone: "faint" }));
  o.push(D([at([gb.x + 3.4, gb.y + 3.4, 8]), at([gb.x + gb.w - 3.4, gb.y + 3.4, 8])], { size: 0.45 }));
  o.push(S(cyl(GAUGE_X, -47, 4.6, 8, 2.4, 28), { tone: "mid" }));
  o.push(S(cyl(GAUGE_X, -47, 3, 10.4, 61.6, 28), { tone: "mid", crease: "none" }));
  o.push(L([22, 34, 46, 58].map((z) => k.sideArc(GAUGE_X, -47, 3, z, P, 16)).join(""), { tone: "faint" }));
  o.push(S(cyl(GAUGE_X, -47, 3.6, 72, 2, 28), { tone: "mid" }));
  // the master card (the incumbent) clamped on a plinth beside the gauge
  const pl = { x: 118, y: -57, w: 30, d: 26, r: 2 };
  o.push(S(slab(pl, 0, 6, 4, 0.6), { tone: "mid" }));
  const mc = { x: 122, y: -53, w: 22, d: 18, r: 1 };
  o.push(S(slab(mc, 6, 1.4, 3), { tone: "hi", crease: "lo" }));
  o.push(L([[4, 14], [8, 10], [12, 15]].map(([dy, w]) => k.lineOnTop([mc.x + 3, mc.y + dy], [mc.x + 3 + w, mc.y + dy], 7.4, P)).join(""), { tone: "mid", free: true }));
  o.push(D([at([pl.x + 2.2, pl.y + 2.2, 6]), at([pl.x + pl.w - 2.2, pl.y + pl.d - 2.2, 6])], { size: 0.45 }));
  return o.join("");
}

function carriage() {
  const o = [`<path class="iso-halo" d="${k.haloOf({ x: -17, y: -19, w: 34, d: 38, r: 3 }, 3, 0.1, P)}" style="filter:blur(4px)"/>`];
  const plan = { x: -20, y: -22, w: 40, d: 44, r: 3 };
  o.push(S(slab(plan, 3, 4, 5, 0.6), { tone: "mid" }));
  for (const [x, y] of k.corners(plan, 4.5)) {
    o.push(S(slab({ x: x - 2.6, y: y - 2.6, w: 5.2, d: 5.2, r: 1.2 }, 3, 5.2, 3), { tone: "mid" }));
  }
  o.push(D(k.corners(plan, 4.5).map(([x, y]) => at([x, y, 8.2])), { size: 0.4 }));
  o.push(D([-10, -3.5, 3.5, 10].map((x) => at([x, plan.y + plan.d - 2.2, 7])), { size: 0.35, tone: "lo" }));
  // the candidate: a card with engraved text lines, held by two jaws
  o.push(S(slab({ x: -18, y: -6, w: 3, d: 12, r: 0.8 }, 7, 2.6, 2), { tone: "mid" }));
  const card = { x: -15, y: -12, w: 30, d: 24, r: 1.2 };
  o.push(S(slab(card, 7, CARD_TOP - 7, 3), { tone: "hi", lit: true }));
  const rule = (dy, w) => k.lineOnTop([card.x + 4, card.y + dy], [card.x + 4 + w, card.y + dy], CARD_TOP, P);
  o.push(L([[5, 21], [9.5, 15], [14, 19]].map(([dy, w]) => rule(dy, w)).join(""), { tone: "hi", free: true }));
  o.push(`<g class="an an-new">${L(rule(18.5, 12), { tone: "lit", free: true })}</g>`); // the rewritten line
  o.push(S(slab({ x: 15, y: -6, w: 3, d: 12, r: 0.8 }, 7, 2.6, 2), { tone: "mid" }));
  const handle = [11, plan.y + plan.d + 3.6];
  o.push(S(slab({ x: handle[0] - 2, y: plan.y + plan.d - 1, w: 4, d: 3.6, r: 1 }, 4, 2.6, 2), { tone: "mid", crease: "none" }));
  o.push(S(cyl(handle[0], handle[1], 3, 3, 5, 20), { tone: "mid" }));
  o.push(L(k.knurl(handle[0], handle[1], 3, 3, 8, 18, P), { tone: "lo" }));
  return o.join("");
}

const probe = (y) => S(cyl(BRIDGE_X, y, 1.5, CARD_TOP, 1, 14), { tone: "hi", crease: "none" }) + S(cyl(BRIDGE_X, y, 0.9, CARD_TOP + 1, 33, 12), { tone: "hi", crease: "none" });

function bridgeFront() {
  const o = [];
  for (const y of PROBES_Y) o.push(S(cyl(BRIDGE_X, y, 1.9, 41, 5, 14), { tone: "mid" }));
  o.push(S(cyl(BRIDGE_X, 38, 5, 0, 3, 28), { tone: "mid" }));
  o.push(S(cyl(BRIDGE_X, 38, 3.2, 3, 43, 28), { tone: "mid", crease: "none" }));
  o.push(L(k.sideArc(BRIDGE_X, 38, 3.2, 24, P, 16), { tone: "faint" }));
  const beam = { x: BRIDGE_X - 8, y: -46, w: 16, d: 92, r: 2 };
  o.push(S(slab(beam, 46, 8, 4, 0.6), { tone: "mid" }));
  o.push(D([at([BRIDGE_X + 5, -42, 54]), at([BRIDGE_X + 5, 42, 54]), at([BRIDGE_X - 5, 42, 54])], { size: 0.45 }));
  // one lamp per probe (per example)
  o.push(L(PROBES_Y.map((y) => k.ring(BRIDGE_X - 4.6, y, 1.3, 54, P, 14)).join(""), { tone: "lo" }));
  PROBES_Y.forEach((y, i) => { const [cx, cy] = at([BRIDGE_X - 4.6, y, 54]); o.push(`<circle class="an lamp an-lamp${i}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="1.25"/>`); });
  // junction box, where the diagnostics cable leaves
  const jb = { x: BRIDGE_X - 7, y: 19, w: 14, d: 19, r: 1.5 };
  o.push(S(slab(jb, 54, 5, 3, 0.5), { tone: "mid" }));
  o.push(D([at([jb.x + 2, jb.y + 2, 59]), at([jb.x + jb.w - 2, jb.y + 2, 59]), at([jb.x + 2, jb.y + jb.d - 2, 59])], { size: 0.4 }));
  o.push(S(cyl(CABLE_A[0], CABLE_A[1], 2.5, 59, 2.5, 18), { tone: "mid" }));
  // score dial
  const [x, y] = DIAL;
  o.push(S(cyl(x, y, 5.5, 54, 3, 24), { tone: "mid" }));
  o.push(S(cyl(x, y, DIAL_R, 57, 7, 64, 0.8), { tone: "hi", crease: "lo" }));
  o.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 2.8, DIAL_TOP, 64, P, 0.3), { tone: "lo" }));
  o.push(L(k.sideArc(x, y, DIAL_R, DIAL_TOP - 2.8, P, 40), { tone: "faint" }));
  o.push(L(k.ring(x, y, DIAL_R - 1.8, DIAL_TOP, P, 64), { tone: "lo" }));
  const t = k.radialTicks(x, y, DIAL_R - 2.4, 50, 5, DIAL_TOP, [1.1, 2.3], P, SWEEP_FROM, SWEEP);
  o.push(L(t.minor, { tone: "lo" }), L(t.major, { tone: "mid" }));
  o.push(S(slab({ x: x + DIAL_R - 1, y: y - 2.2, w: 4.4, d: 4.4, r: 1.2 }, 58.5, 4.4, 3), { tone: "mid" }));
  o.push(L(k.ring(x, y, 1.5, DIAL_TOP, P, 16), { tone: "mid" }));
  return o.join("");
}

const needlePath = (angle) => k.lineOnTop([DIAL[0] - Math.sin(angle) * 2.6, DIAL[1] + Math.cos(angle) * 2.6], [DIAL[0] + Math.sin(angle) * (DIAL_R - 3.4), DIAL[1] - Math.cos(angle) * (DIAL_R - 3.4)], DIAL_TOP + 0.2, P);
const stylus = () => S(cyl(HEAD_X, 0, 1, CARD_TOP + STYLUS_DROP, 30, 12), { tone: "hi", crease: "none", lit: true });

function headFront() {
  const o = [];
  o.push(S(slab({ x: HEAD_X - 3, y: -39.5, w: 6, d: 33.5, r: 1.6 }, 54, 4, 3, 0.5), { tone: "mid" }));
  o.push(D([at([HEAD_X, -35, 58]), at([HEAD_X, -11, 58])], { size: 0.45 }));
  o.push(S(cyl(HEAD_X, 0, 3.6, 38, 8, 20), { tone: "mid" }));
  o.push(S(cyl(HEAD_X, 0, 7.5, 46, 14, 40, 0.6), { tone: "hi", crease: "lo" }));
  o.push(L(k.knurl(HEAD_X, 0, 7.5, 56, 60, 44, P, 0.3), { tone: "lo" }));
  o.push(L(k.sideArc(HEAD_X, 0, 7.5, 56, P, 28), { tone: "faint" }));
  o.push(L(k.ring(HEAD_X, 0, 5.4, 60, P, 32), { tone: "lo" }));
  o.push(S(cyl(CABLE_B[0], CABLE_B[1], 2.5, 60, 2.5, 18), { tone: "mid" }));
  return o.join("");
}

function gaugeSlider() { // drawn raised; the group drops by GAUGE_DROP to measure
  const o = [];
  const zs = 30;
  o.push(S(slab({ x: GAUGE_X - 6, y: -43.6, w: 12, d: 9.2, r: 1.5 }, zs, 10, 3, 0.5), { tone: "mid" }));
  o.push(D([at([GAUGE_X - 3.6, -39, zs + 10]), at([GAUGE_X + 3.6, -39, zs + 10])], { size: 0.4 }));
  o.push(S(slab({ x: GAUGE_X + 6, y: -40.4, w: 2.6, d: 2.8, r: 0.8 }, zs + 3, 4, 2), { tone: "mid", crease: "none" }));
  o.push(S(cyl(GAUGE_X + 10.6, -39, 2.4, zs + 2, 6, 18), { tone: "mid" }));
  o.push(L(k.knurl(GAUGE_X + 10.6, -39, 2.4, zs + 2, zs + 8, 16, P), { tone: "lo" }));
  o.push(S(slab({ x: GAUGE_X - 3, y: -34.4, w: 6, d: 38, r: 1.6 }, zs + 3, 4, 3, 0.5), { tone: "mid" }));
  o.push(D([at([GAUGE_X, -30, zs + 7]), at([GAUGE_X, -2, zs + 7])], { size: 0.4 }));
  o.push(S(cyl(GAUGE_X, 0, 0.9, CARD_TOP + GAUGE_DROP, 7, 12), { tone: "hi", crease: "none" }));
  o.push(S(cyl(GAUGE_X, 0, 2.1, zs - 3, 6, 16), { tone: "mid" }));
  return o.join("");
}

function cable() {
  const pieces = tubePieces(cablePoints, 1.15, P, { maxLength: 400 });
  return pieces.map((piece) => tubeSvg(piece, { tone: "mid" })).join("");
}
const cableFlow = () => `<path class="an an-flow" pathLength="100" d="M${cablePoints.map((p) => at(p).map((v) => v.toFixed(1)).join(" ")).join("L")}"/>`;

// ---- motion: one CSS cycle, no script (GitHub strips scripts from images) ---------------------
const CYCLE = 20; // seconds
const frames = (name, list, prop) => `@keyframes ${name}{${list.map(([pct, value]) => `${pct}%{${prop(value)}}`).join("")}}`;
const tr = (v) => { const [x, y] = px(v); return `transform:translate(${x.toFixed(2)}px,${y.toFixed(2)}px)`; };
// CSS transforms replace the transform attribute, so every keyframe is an absolute position.
const carriageAt = [[0, RACK_X], [4, RACK_X], [12, BRIDGE_X], [25, BRIDGE_X], [32, HEAD_X], [46, HEAD_X], [53, BRIDGE_X], [66, BRIDGE_X], [75, GAUGE_X], [86, GAUGE_X], [98, RACK_X], [100, RACK_X]];
const probeUp = [[0, 1], [12.5, 1], [15.5, 0], [23, 0], [25, 1], [53.5, 1], [56.5, 0], [64, 0], [66, 1], [100, 1]];
const stylusDown = [[0, 0], [34, 0], [35.5, 1], [37.5, 0], [39.5, 1], [41.5, 0], [43.5, 1], [45.5, 0], [100, 0]];
const gaugeDown = [[0, 0], [76, 0], [79, 1], [84, 1], [86, 0], [100, 0]];
const newLine = [[0, 0], [35, 0], [35.6, 0.34], [39, 0.34], [39.6, 0.67], [43, 0.67], [43.6, 1], [98, 1], [100, 0]];
const flow = [[0, [0, 0]], [24, [0, 0]], [25, [1, 0]], [33, [1, -108]], [34, [0, -108]], [100, [0, -108]]];
const lampOn = (i) => {
  const first = i === 2 ? [] : [[16 + i, 0.12], [16.6 + i, 1], [24.4, 1], [25, 0.12]]; // example 3 fails the first time
  return [[0, 0.12], ...first, [57 + i, 0.12], [57.6 + i, 1], [65.4, 1], [66, 0.12], [100, 0.12]];
};
// the needle turns in the dial's plane; on screen that is an affine map about the hub
const needleMatrix = (theta) => {
  const [cx, cy] = at([DIAL[0], DIAL[1], DIAL_TOP + 0.2]);
  const [ax, ay] = k.axisVector(P, [1, 0, 0]), [bx, by] = k.axisVector(P, [0, 1, 0]);
  const det = ax * by - bx * ay, c = Math.cos(theta), s = Math.sin(theta);
  const m = [[ax * c + bx * s, -ax * s + bx * c], [ay * c + by * s, -ay * s + by * c]]; // [ex ey]·R
  const inv = [[by / det, -bx / det], [-ay / det, ax / det]];
  const a = m[0][0] * inv[0][0] + m[0][1] * inv[1][0], cc = m[0][0] * inv[0][1] + m[0][1] * inv[1][1];
  const b = m[1][0] * inv[0][0] + m[1][1] * inv[1][0], d = m[1][0] * inv[0][1] + m[1][1] * inv[1][1];
  return `matrix(${[a, b, cc, d, cx - a * cx - cc * cy, cy - b * cx - d * cy].map((v) => v.toFixed(4)).join(",")})`;
};
const scoreAt = [[0, 0], [15.5, 0], [21, SCORE_BEFORE], [24, SCORE_BEFORE], [28.5, 0], [56.5, 0], [62, SCORE_AFTER], [65, SCORE_AFTER], [69.5, 0], [100, 0]];
const needleFrames = () => {
  const list = [];
  for (let i = 0; i < scoreAt.length - 1; i++) {
    const [p0, s0] = scoreAt[i], [p1, s1] = scoreAt[i + 1];
    const steps = Math.max(1, Math.ceil((Math.abs(s1 - s0) * SWEEP) / (Math.PI / 12)));
    for (let j = 0; j < steps; j++) { const u = j / steps, e = u * u * (3 - 2 * u); list.push([(p0 + (p1 - p0) * u).toFixed(2), s0 + (s1 - s0) * e]); }
  }
  list.push([100, 0]);
  return list;
};

// one caption per step, shown while that step happens; the legend entry for the station lights up with it
const STEPS = [
  { from: 0, to: 11, key: 1, text: "Pick one version of your text from the pool." },
  { from: 11, to: 26, key: 2, text: "Score it on a few examples. One of the five fails." },
  { from: 26, to: 51, key: 3, text: "An LLM reads what failed and rewrites the text." },
  { from: 51, to: 68, key: 2, text: "Score the rewrite. All five examples pass now." },
  { from: 68, to: 88, key: 4, text: "Check it against the old best. It is better, so it is kept." },
  { from: 88, to: 100, key: 1, text: "Back to the pool. Repeat until the budget runs out." },
];
const STATIC_STEP = 3; // what a still image (reduced motion) shows
const window = (ranges) => { // opacity keyframes: 1 inside the ranges, 0 outside, short fades
  const list = [[0, ranges.some(([a]) => a === 0) ? 1 : 0]];
  for (const [a, b] of ranges) { if (a > 0) list.push([a - 0.01, 0], [a + 1, 1]); if (b < 100) list.push([b - 1, 1], [b, 0]); }
  list.push([100, ranges.some(([, b]) => b === 100) ? 1 : 0]);
  return list.sort((x, y) => x[0] - y[0]);
};
const EASE = "cubic-bezier(.45,0,.2,1)";
const MOTION = `
.an{animation-duration:${CYCLE}s;animation-iteration-count:infinite;animation-timing-function:${EASE}}
.an-carriage{animation-name:carriage}.an-probes{animation-name:probes}.an-stylus{animation-name:stylus}.an-gauge{animation-name:gauge}
.an-new{animation-name:newline;animation-timing-function:linear}
.an-needle{animation-name:needle;animation-timing-function:linear}
.lamp{fill:var(--anatomy-lit);opacity:1}
${PROBES_Y.map((_, i) => `.an-lamp${i}{animation-name:lamp${i};animation-timing-function:linear}`).join("")}
.an-flow{fill:none;stroke:var(--anatomy-lit);stroke-width:1.5;stroke-linecap:round;stroke-dasharray:8 200;opacity:0;animation-name:flow;animation-timing-function:linear}
${frames("carriage", carriageAt, (x) => tr([x, 0, 0]))}
${frames("probes", probeUp, (u) => tr([0, 0, u * PROBE_LIFT]))}
${frames("stylus", stylusDown, (u) => tr([0, 0, -u * STYLUS_DROP]))}
${frames("gauge", gaugeDown, (u) => tr([0, 0, -u * GAUGE_DROP]))}
${frames("newline", newLine, (v) => `opacity:${v}`)}
${frames("flow", flow, ([o, off]) => `opacity:${o};stroke-dashoffset:${off}`)}
${PROBES_Y.map((_, i) => frames(`lamp${i}`, lampOn(i), (v) => `opacity:${v}`)).join("")}
${frames("needle", needleFrames(), (s) => `transform:${needleMatrix((s - SCORE_AFTER) * SWEEP)}`)}
${STEPS.map((step, i) => `.an-step${i}{animation-name:step${i};animation-timing-function:linear;opacity:${i === STATIC_STEP ? 1 : 0}}` + frames(`step${i}`, window([[step.from, step.to]]), (v) => `opacity:${v}`)).join("")}
${[1, 2, 3, 4].map((key) => `.an-key${key}{animation-name:key${key};animation-timing-function:linear;opacity:${STEPS[STATIC_STEP].key === key ? 1 : 0}}` + frames(`key${key}`, window(STEPS.filter((step) => step.key === key).map((step) => [step.from, step.to])), (v) => `opacity:${v}`)).join("")}
@media (prefers-reduced-motion:reduce){.an{animation:none!important}}
.say{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;font-weight:500;fill:var(--anatomy-strong)}
.say-n{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:10.5px;letter-spacing:.1em;fill:var(--anatomy-muted)}
.k-on{fill:var(--anatomy-strong)}
.k-card{fill:var(--anatomy-face);stroke:var(--anatomy-lit);stroke-width:1}.k-card-l{stroke:var(--anatomy-lit);stroke-width:.8}
.card-bg{fill:var(--anatomy-card);stroke:var(--anatomy-faint);stroke-width:1}
.t{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:10.5px;letter-spacing:.04em;fill:var(--anatomy-muted)}
.t-fig{fill:var(--anatomy-text)}.t-title{letter-spacing:.1em;text-anchor:end}
.k{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.k-n{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:11px;fill:var(--anatomy-strong);text-anchor:middle}
.k-box{fill:none;stroke:var(--anatomy-mid);stroke-width:.8}
.k-h{font-size:12.5px;font-weight:600;fill:var(--anatomy-strong)}
.k-d{font-size:11px;fill:var(--anatomy-text)}
.k-rule{stroke:var(--anatomy-faint);stroke-width:1}
`;

// ---- assemble: static pose = card under the probes, scored (what reduced motion shows) --------
const drawing = [
  base(), rails(), rack(), behind(),
  `<g class="an an-carriage" transform="${move([BRIDGE_X, 0, 0])}">${carriage()}</g>`,
  `<g class="an an-probes">${PROBES_Y.map(probe).join("")}</g>`,
  `<g class="an an-stylus">${stylus()}</g>`,
  bridgeFront(),
  `<g class="an an-needle"><path class="iso-line" data-tone="lit" data-free="start" d="${needlePath(SWEEP_FROM + SWEEP * SCORE_AFTER)}" style="stroke-width:1.1"/></g>`,
  headFront(),
  `<g class="an an-gauge">${gaugeSlider()}</g>`,
  cable(), cableFlow(),
].join("");

const TOP = 34, SAY = 30, LEGEND = 96, TOTAL = TOP + H + SAY + LEGEND;
const KEYS = [
  ["Your text", "The card. A prompt,", "regex, config or code", "the engine rewrites."],
  ["Pool", "The rack. Every version", "that is best on at", "least one example."],
  ["Evaluator", "The probes and dial.", "Your code scores the", "text, says what failed."],
  ["Proposer", "The engraving head.", "An LLM reads failures", "and rewrites the text."],
  ["Winner check", "The height gauge. Kept", "only if it beats the", "old best."],
];
const COL = (W - 44) / KEYS.length, LY = TOP + H + SAY;
const legend = KEYS.map(([head, ...lines], i) => {
  const x = 22 + i * COL, y = LY + 26;
  const mark = i === 0
    ? `<rect class="k-card" x="${x}" y="${y - 10}" width="16" height="13" rx="1.5"/><path class="k-card-l" d="M${x + 3} ${y - 6}h10M${x + 3} ${y - 3}h7M${x + 3} ${y}h9"/>`
    : `<rect class="k-box" x="${x}" y="${y - 11}" width="16" height="16" rx="2"/><text class="k-n" x="${x + 8}" y="${y + 1.2}">${i}</text>`;
  const bar = i === 0 ? "" : `<rect class="an k-on an-key${i}" x="${x}" y="${LY + 1}" width="${COL - 14}" height="2"/>`;
  return bar + mark + `<text class="k k-h" x="${x + 24}" y="${y + 1.5}">${head}</text>` +
    lines.map((line, j) => `<text class="k k-d" x="${x}" y="${y + 21 + j * 14}">${line}</text>`).join("");
}).join("");
const say = `<text class="say-n" x="22" y="${TOP + H + 9}">NOW</text>` +
  STEPS.map((step, i) => `<text class="an say an-step${i}" x="60" y="${TOP + H + 10}">${step.text}</text>`).join("");

for (const theme of ["light", "dark"]) {
  const svg = k.figureSvg({
    width: W, height: TOTAL, standalone: theme,
    label: "The optimization loop drawn as a small proving bench. A carriage carries a card (the text being optimized) along a rail between four stations: a rack of cards (the pool of best versions), a bridge of five probes with a score dial (the evaluator), an engraving head fed by a cable from the probes (the LLM proposer reading the diagnostics), and a height gauge beside a master card (the winner check against the old best).",
    body: [
      `<style>${MOTION}</style>`,
      `<rect class="card-bg" x=".5" y=".5" width="${W - 1}" height="${TOTAL - 1}" rx="10"/>`,
      `<text class="t t-fig" x="22" y="26">Fig 1</text><text class="t t-title" x="${W - 22}" y="26">ONE ROUND OF OPTIMIZATION</text>`,
      `<g transform="translate(0 ${TOP + 2})">${drawing}</g>`,
      say,
      `<line class="k-rule" x1="22" x2="${W - 22}" y1="${LY + 2}" y2="${LY + 2}"/>`,
      legend,
    ],
  });
  writeFileSync(join(OUT, `loop-${theme}.svg`), svg);
  console.log("wrote", `loop-${theme}.svg`, `${(svg.length / 1024).toFixed(0)} KB`);
}
