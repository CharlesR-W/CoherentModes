// Colour grammar shared by every chart. Values come from palette.json (meta.palette).
export function hexRgb(h) {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
export const rgbStr = ([r, g, b]) => `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

export class Palette {
  constructor(p) {
    this.p = p;
    this.classes = p.classes;
    this.sets = p.sets;
    this.unassigned = p.unassigned;
    this.neg = hexRgb(p.diverging.neg); this.mid = hexRgb(p.diverging.mid); this.pos = hexRgb(p.diverging.pos);
    this.seqStops = p.sequential.map(hexRgb);
    this.ordStops = p.ordinal.map(hexRgb);
    this.fg = p.fg; this.mute = p.mute; this.line = p.line;
  }
  set(j) { return this.sets[j % this.sets.length]; }
  // v in [-1, 1] -> diverging colour (grey midpoint)
  div(v) {
    const a = Math.max(-1, Math.min(1, v));
    return rgbStr(a < 0 ? mix(this.mid, this.neg, Math.pow(-a, 0.8)) : mix(this.mid, this.pos, Math.pow(a, 0.8)));
  }
  ramp(stops, t) {
    const x = Math.max(0, Math.min(1, t)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x));
    return rgbStr(mix(stops[i], stops[i + 1], x - i));
  }
  seq(t) { return this.ramp(this.seqStops, t); }
  // for points on a white cell the lightest sequential steps vanish: start at 15 %
  seqDot(t) { return this.ramp(this.seqStops, 0.15 + 0.85 * t); }
  ord(t) { return this.ramp(this.ordStops, t); }
  divGradient() { return `linear-gradient(90deg, ${this.div(-1)}, ${this.div(-0.5)}, ${this.div(0)}, ${this.div(0.5)}, ${this.div(1)})`; }
  seqGradient(dot = true) { const f = dot ? t => this.seqDot(t) : t => this.seq(t); return `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map(f).join(",")})`; }
}

// quantised LUTs so that a canvas pass sets fillStyle once per colour bucket
export const LEVELS = 41;
export function divLUT(pal) { return Array.from({ length: LEVELS }, (_, i) => pal.div(-1 + 2 * i / (LEVELS - 1))); }
export function seqLUT(pal) { return Array.from({ length: LEVELS }, (_, i) => pal.seqDot(i / (LEVELS - 1))); }
export const qDiv = v => Math.max(0, Math.min(LEVELS - 1, Math.round((v + 1) / 2 * (LEVELS - 1))));
export const qSeq = t => Math.max(0, Math.min(LEVELS - 1, Math.round(t * (LEVELS - 1))));
