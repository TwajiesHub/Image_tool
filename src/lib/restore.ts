/** Browser anti-blur stack: denoise, Wiener deconvolution, Lanczos, detail. */

export const MAX_PROCESS_EDGE = 960;
export const MAX_OUTPUT_EDGE = 3200;

export type ScaleFactor = 1 | 2 | 4;
export type BlurModel = "auto" | "defocus" | "motion";

export type RestoreOptions = {
  scale: ScaleFactor;
  model: BlurModel;
  /** Gaussian sigma in working-image pixels. Ignored for auto and motion. */
  sigma: number;
  /** Motion kernel length in pixels (odd). */
  motionLength: number;
  /** Degrees, 0 = horizontal. */
  motionAngle: number;
  /** Inverse strength. Higher pushes harder and rings sooner. */
  iterations: number;
  /** 0–1 chroma (and, when high, luma) denoise. */
  denoise: number;
  /** 0–1 local contrast + edge-aware unsharp. */
  detail: number;
  outWidth: number;
  outHeight: number;
};

export type RestoreReport = {
  model: BlurModel;
  sigma: number;
  motionLength: number;
  motionAngle: number;
  iterations: number;
  stack: "frequency";
  width: number;
  height: number;
  ms: number;
};

export type RgbaImage = {
  data: Uint8ClampedArray;
  width: number;
  height: number;
};

export type ProgressFn = (stage: string, value: number) => void;

export function planSizes(srcW: number, srcH: number, scale: number) {
  const long = Math.max(srcW, srcH, 1);
  const pScale = Math.min(1, MAX_PROCESS_EDGE / long);
  const pw = Math.max(2, Math.round(srcW * pScale));
  const ph = Math.max(2, Math.round(srcH * pScale));
  let ow = Math.max(2, Math.round(srcW * scale));
  let oh = Math.max(2, Math.round(srcH * scale));
  const oLong = Math.max(ow, oh);
  let capped = false;
  if (oLong > MAX_OUTPUT_EDGE) {
    capped = true;
    const f = MAX_OUTPUT_EDGE / oLong;
    ow = Math.max(2, Math.round(ow * f));
    oh = Math.max(2, Math.round(oh * f));
  }
  return { pw, ph, ow, oh, capped, scaledDown: pw < srcW || ph < srcH };
}

function clamp(v: number, lo: number, hi: number) {
  return v < lo ? lo : v > hi ? hi : v;
}

function srgbToLinear(c: number) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(c: number) {
  const x = clamp(c, 0, 1);
  return x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055;
}

const kernelCache = new Map<number, Float32Array>();

export function gaussianKernel(sigma: number): Float32Array {
  const s = Math.max(0.3, sigma);
  const key = Math.round(s * 100);
  const cached = kernelCache.get(key);
  if (cached) return cached;
  const radius = Math.max(1, Math.ceil(s * 3));
  const k = new Float32Array(radius * 2 + 1);
  const denom = 2 * s * s;
  let sum = 0;
  for (let i = 0; i < k.length; i++) {
    const x = i - radius;
    const v = Math.exp(-(x * x) / denom);
    k[i] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  kernelCache.set(key, k);
  return k;
}

/** Separable normalized convolution with reflected edges. */
export function convolveSeparable(src: Float32Array, w: number, h: number, kernel: Float32Array) {
  const radius = (kernel.length - 1) >> 1;
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let xx = x + k;
        if (xx < 0) xx = -xx;
        else if (xx >= w) {
          xx = (w - 1) - (xx - (w - 1));
          if (xx < 0) xx = 0;
        }
        acc += src[row + xx] * kernel[k + radius];
      }
      tmp[row + x] = acc;
    }
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let yy = y + k;
        if (yy < 0) yy = -yy;
        else if (yy >= h) {
          yy = (h - 1) - (yy - (h - 1));
          if (yy < 0) yy = 0;
        }
        acc += tmp[yy * w + x] * kernel[k + radius];
      }
      out[y * w + x] = acc;
    }
  }
  return out;
}

function nextPow2(n: number) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function fft1d(re: Float64Array, im: Float64Array, invert: boolean) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI) / len * (invert ? 1 : -1);
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1;
      let wi = 0;
      const half = len >> 1;
      for (let j = 0; j < half; j++) {
        const ur = re[i + j];
        const ui = im[i + j];
        const vr = re[i + j + half] * wr - im[i + j + half] * wi;
        const vi = re[i + j + half] * wi + im[i + j + half] * wr;
        re[i + j] = ur + vr;
        im[i + j] = ui + vi;
        re[i + j + half] = ur - vr;
        im[i + j + half] = ui - vi;
        const nwr = wr * wlenRe - wi * wlenIm;
        wi = wr * wlenIm + wi * wlenRe;
        wr = nwr;
      }
    }
  }
  if (invert) {
    for (let i = 0; i < n; i++) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

function fft2d(re: Float64Array, im: Float64Array, w: number, h: number, invert: boolean) {
  const rowRe = new Float64Array(w);
  const rowIm = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      rowRe[x] = re[row + x];
      rowIm[x] = im[row + x];
    }
    fft1d(rowRe, rowIm, invert);
    for (let x = 0; x < w; x++) {
      re[row + x] = rowRe[x];
      im[row + x] = rowIm[x];
    }
  }
  const colRe = new Float64Array(h);
  const colIm = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      colRe[y] = re[y * w + x];
      colIm[y] = im[y * w + x];
    }
    fft1d(colRe, colIm, invert);
    for (let y = 0; y < h; y++) {
      re[y * w + x] = colRe[y];
      im[y * w + x] = colIm[y];
    }
  }
}

function reflectSource(src: Float32Array, w: number, h: number, x: number, y: number) {
  const px = (w - 1) * 2;
  const py = (h - 1) * 2;
  let xx = x % px;
  if (xx < 0) xx += px;
  if (xx >= w) xx = px - xx;
  let yy = y % py;
  if (yy < 0) yy += py;
  if (yy >= h) yy = py - yy;
  return src[yy * w + xx];
}

function regularizationFor(depth: number) {
  const t = clamp((depth - 2) / 8, 0, 1);
  return 0.045 * 0.02 ** t;
}

/** Frequency-domain Wiener inverse of a point-spread function painted at the origin. */
function wienerDeblur(
  src: Float32Array,
  w: number,
  h: number,
  regularize: number,
  paintPsf: (psf: Float64Array, W: number, H: number) => void,
) {
  const W = nextPow2(Math.max(w, 2));
  const H = nextPow2(Math.max(h, 2));
  const ox = Math.floor((W - w) / 2);
  const oy = Math.floor((H - h) / 2);
  const gRe = new Float64Array(W * H);
  const gIm = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) gRe[y * W + x] = reflectSource(src, w, h, x - ox, y - oy);
  }
  const hRe = new Float64Array(W * H);
  const hIm = new Float64Array(W * H);
  paintPsf(hRe, W, H);
  let sum = 0;
  for (let i = 0; i < hRe.length; i++) sum += hRe[i];
  if (sum > 0) {
    const inv = 1 / sum;
    for (let i = 0; i < hRe.length; i++) hRe[i] *= inv;
  }
  fft2d(gRe, gIm, W, H, false);
  fft2d(hRe, hIm, W, H, false);
  const K = Math.max(1e-6, regularize);
  for (let i = 0; i < gRe.length; i++) {
    const den = hRe[i] * hRe[i] + hIm[i] * hIm[i] + K;
    const nr = hRe[i] * gRe[i] + hIm[i] * gIm[i];
    const ni = hRe[i] * gIm[i] - hIm[i] * gRe[i];
    gRe[i] = nr / den;
    gIm[i] = ni / den;
  }
  fft2d(gRe, gIm, W, H, true);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = gRe[(y + oy) * W + (x + ox)];
      out[y * w + x] = Number.isFinite(v) ? v : src[y * w + x];
    }
  }
  return out;
}

function paintGaussian(sigma: number) {
  const s = Math.max(0.35, sigma);
  const rad = Math.max(2, Math.ceil(s * 4));
  const two = 2 * s * s;
  return (psf: Float64Array, W: number, H: number) => {
    for (let y = -rad; y <= rad; y++) {
      for (let x = -rad; x <= rad; x++) {
        const xx = (x + W) % W;
        const yy = (y + H) % H;
        psf[yy * W + xx] = Math.exp(-(x * x + y * y) / two);
      }
    }
  };
}

function paintMotion(length: number, angleDeg: number) {
  const len = Math.max(3, length % 2 === 0 ? length + 1 : length);
  const half = len >> 1;
  const rad = (angleDeg * Math.PI) / 180;
  const dx = Math.cos(rad);
  const dy = Math.sin(rad);
  return (psf: Float64Array, W: number, H: number) => {
    for (let k = -half; k <= half; k++) {
      const x = Math.round(k * dx);
      const y = Math.round(k * dy);
      const xx = (x + W) % W;
      const yy = (y + H) % H;
      psf[yy * W + xx] += 1;
    }
  };
}

function sampleBilinear(src: Float32Array, w: number, h: number, x: number, y: number) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const wx = x - x0;
  const wy = y - y0;
  const xx0 = clamp(x0, 0, w - 1);
  const xx1 = clamp(x0 + 1, 0, w - 1);
  const yy0 = clamp(y0, 0, h - 1);
  const yy1 = clamp(y0 + 1, 0, h - 1);
  const v0 = src[yy0 * w + xx0] + (src[yy0 * w + xx1] - src[yy0 * w + xx0]) * wx;
  const v1 = src[yy1 * w + xx0] + (src[yy1 * w + xx1] - src[yy1 * w + xx0]) * wx;
  return v0 + (v1 - v0) * wy;
}

/** Edge-width estimator. Returns sigma in pixels of a Gaussian defocus. */
export function estimateDefocusSigma(y: Float32Array, w: number, h: number) {
  const step = Math.max(1, Math.floor(Math.max(w, h) / 200));
  const mags: number[] = [];
  const gradAt = (x: number, y0: number) => {
    const xm = sampleBilinear(y, w, h, x - 1, y0);
    const xp = sampleBilinear(y, w, h, x + 1, y0);
    const ym = sampleBilinear(y, w, h, x, y0 - 1);
    const yp = sampleBilinear(y, w, h, x, y0 + 1);
    const gx = (xp - xm) * 0.5;
    const gy = (yp - ym) * 0.5;
    return { gx, gy, mag: Math.hypot(gx, gy) };
  };
  for (let yy = 1; yy < h - 1; yy += step) {
    for (let xx = 1; xx < w - 1; xx += step) {
      const g = gradAt(xx, yy);
      if (g.mag > 0.015) mags.push(g.mag);
    }
  }
  if (mags.length < 24) return 1.15;
  mags.sort((a, b) => a - b);
  const thresh = mags[Math.floor(mags.length * 0.72)];
  const widths: number[] = [];
  for (let yy = 2; yy < h - 2; yy += step) {
    for (let xx = 2; xx < w - 2; xx += step) {
      const g = gradAt(xx, yy);
      if (g.mag < thresh) continue;
      const inv = 1 / g.mag;
      const ux = g.gx * inv;
      const uy = g.gy * inv;
      const plus = gradAt(xx + ux, yy + uy).mag;
      const minus = gradAt(xx - ux, yy - uy).mag;
      if (g.mag + 1e-6 < plus || g.mag + 1e-6 < minus) continue;
      let left = 0;
      let right = 0;
      for (let t = 1; t <= 14; t++) {
        if (gradAt(xx - t * ux, yy - t * uy).mag < g.mag * 0.5) break;
        left = t;
      }
      for (let t = 1; t <= 14; t++) {
        if (gradAt(xx + t * ux, yy + t * uy).mag < g.mag * 0.5) break;
        right = t;
      }
      widths.push(left + right);
    }
  }
  if (widths.length < 8) return 1.15;
  widths.sort((a, b) => a - b);
  const median = widths[Math.floor(widths.length / 2)];
  const sigma = (median + 1) / 2.35;
  return clamp(sigma, 0.45, 3.3);
}

function rgbToYCbCr(r: Float32Array, g: Float32Array, b: Float32Array) {
  const n = r.length;
  const Y = new Float32Array(n);
  const Cb = new Float32Array(n);
  const Cr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const rv = r[i];
    const gv = g[i];
    const bv = b[i];
    const y = 0.2126 * rv + 0.7152 * gv + 0.0722 * bv;
    Y[i] = y;
    Cb[i] = (bv - y) / 1.8556;
    Cr[i] = (rv - y) / 1.5748;
  }
  return { Y, Cb, Cr };
}

function ycbcrToRgb(Y: Float32Array, Cb: Float32Array, Cr: Float32Array) {
  const n = Y.length;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const y = Y[i];
    const cb = Cb[i];
    const cr = Cr[i];
    r[i] = y + 1.5748 * cr;
    g[i] = y - 0.1873 * cb - 0.4681 * cr;
    b[i] = y + 1.8556 * cb;
  }
  return { r, g, b };
}

function bilateral(src: Float32Array, w: number, h: number, radius: number, spatialSigma: number, rangeSigma: number) {
  const out = new Float32Array(src.length);
  const s2 = 2 * spatialSigma * spatialSigma;
  const r2 = 2 * rangeSigma * rangeSigma;
  const spatial: number[] = [];
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) spatial.push(Math.exp(-(dx * dx + dy * dy) / s2));
  }
  const reflect = (i: number, n: number) => {
    if (n <= 1) return 0;
    const period = (n - 1) * 2;
    let m = i % period;
    if (m < 0) m += period;
    if (m >= n) m = period - m;
    return m;
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const center = src[y * w + x];
      let acc = 0;
      let norm = 0;
      let k = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = reflect(y + dy, h);
        for (let dx = -radius; dx <= radius; dx++, k++) {
          const xx = reflect(x + dx, w);
          const v = src[yy * w + xx];
          const wt = spatial[k] * Math.exp(-((v - center) * (v - center)) / r2);
          acc += v * wt;
          norm += wt;
        }
      }
      out[y * w + x] = norm > 0 ? acc / norm : center;
    }
  }
  return out;
}

function clahe(src: Float32Array, w: number, h: number, blend: number) {
  if (blend <= 0.001) return new Float32Array(src);
  const tilesX = clamp(Math.round(w / 64), 2, 8);
  const tilesY = clamp(Math.round(h / 64), 2, 8);
  const bins = 256;
  const maps: Float32Array[] = [];
  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const hist = new Float32Array(bins);
      const x0 = Math.floor((tx * w) / tilesX);
      const x1 = Math.floor(((tx + 1) * w) / tilesX);
      const y0 = Math.floor((ty * h) / tilesY);
      const y1 = Math.floor(((ty + 1) * h) / tilesY);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const bin = clamp(Math.round(clamp(src[y * w + x], 0, 1) * 255), 0, 255);
          hist[bin]++;
          count++;
        }
      }
      if (count < 1) {
        const cdf = new Float32Array(bins);
        for (let i = 0; i < bins; i++) cdf[i] = i / 255;
        maps.push(cdf);
        continue;
      }
      const limit = Math.max(1, (2.8 * count) / bins);
      let excess = 0;
      for (let i = 0; i < bins; i++) {
        if (hist[i] > limit) {
          excess += hist[i] - limit;
          hist[i] = limit;
        }
      }
      const bonus = excess / bins;
      let acc = 0;
      const cdf = new Float32Array(bins);
      for (let i = 0; i < bins; i++) {
        hist[i] += bonus;
        acc += hist[i];
        cdf[i] = acc / count;
      }
      maps.push(cdf);
    }
  }
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    const tyf = (y / h) * tilesY - 0.5;
    const yT0 = Math.floor(tyf);
    const ay = tyf - yT0;
    const yy0 = clamp(yT0, 0, tilesY - 1);
    const yy1 = clamp(yT0 + 1, 0, tilesY - 1);
    for (let x = 0; x < w; x++) {
      const txf = (x / w) * tilesX - 0.5;
      const xT0 = Math.floor(txf);
      const ax = txf - xT0;
      const xx0 = clamp(xT0, 0, tilesX - 1);
      const xx1 = clamp(xT0 + 1, 0, tilesX - 1);
      const bin = clamp(Math.round(clamp(src[y * w + x], 0, 1) * 255), 0, 255);
      const v00 = maps[yy0 * tilesX + xx0][bin];
      const v10 = maps[yy0 * tilesX + xx1][bin];
      const v01 = maps[yy1 * tilesX + xx0][bin];
      const v11 = maps[yy1 * tilesX + xx1][bin];
      const mapped = (v00 + (v10 - v00) * ax) * (1 - ay) + (v01 + (v11 - v01) * ax) * ay;
      const base = src[y * w + x];
      out[y * w + x] = base * (1 - blend) + mapped * blend;
    }
  }
  return out;
}

function lanczos(x: number, a: number) {
  const ax = Math.abs(x);
  if (ax < 1e-6) return 1;
  if (ax >= a) return 0;
  const pi = Math.PI;
  const pix = pi * x;
  return (a * Math.sin(pix) * Math.sin(pix / a)) / (pix * pix);
}

export function resizeLanczos(src: Float32Array, sw: number, sh: number, dw: number, dh: number) {
  if (sw === dw && sh === dh) return new Float32Array(src);
  const a = 3;
  const tmp = new Float32Array(dw * sh);
  const xRatio = sw / dw;
  const xSupport = a * Math.max(1, xRatio);
  for (let y = 0; y < sh; y++) {
    const row = y * sw;
    for (let dx = 0; dx < dw; dx++) {
      const center = (dx + 0.5) * xRatio - 0.5;
      const start = Math.max(0, Math.ceil(center - xSupport));
      const end = Math.min(sw - 1, Math.floor(center + xSupport));
      let acc = 0;
      let norm = 0;
      const scale = Math.max(1, xRatio);
      for (let sx = start; sx <= end; sx++) {
        const wt = lanczos((center - sx) / scale, a);
        acc += src[row + sx] * wt;
        norm += wt;
      }
      tmp[y * dw + dx] = norm > 0 ? acc / norm : 0;
    }
  }
  const out = new Float32Array(dw * dh);
  const yRatio = sh / dh;
  const ySupport = a * Math.max(1, yRatio);
  for (let dy = 0; dy < dh; dy++) {
    const center = (dy + 0.5) * yRatio - 0.5;
    const start = Math.max(0, Math.ceil(center - ySupport));
    const end = Math.min(sh - 1, Math.floor(center + ySupport));
    const scale = Math.max(1, yRatio);
    for (let x = 0; x < dw; x++) {
      let acc = 0;
      let norm = 0;
      for (let sy = start; sy <= end; sy++) {
        const wt = lanczos((center - sy) / scale, a);
        acc += tmp[sy * dw + x] * wt;
        norm += wt;
      }
      out[dy * dw + x] = norm > 0 ? acc / norm : 0;
    }
  }
  return out;
}

function unsharp(r: Float32Array, g: Float32Array, b: Float32Array, w: number, h: number, amount: number) {
  if (amount <= 0.001) return;
  const n = r.length;
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = 0.2126 * r[i] + 0.7152 * g[i] + 0.0722 * b[i];
  const blurred = convolveSeparable(y, w, h, gaussianKernel(1.15));
  for (let i = 0; i < n; i++) {
    const high = y[i] - blurred[i];
    const mag = Math.abs(high);
    const t = clamp((mag - 0.004) / 0.04, 0, 1);
    const mask = t * t * (3 - 2 * t);
    const delta = amount * high * (0.35 + 0.65 * mask);
    r[i] = clamp(r[i] + delta, 0, 1);
    g[i] = clamp(g[i] + delta, 0, 1);
    b[i] = clamp(b[i] + delta, 0, 1);
  }
}

function rgbaToLinear(data: Uint8ClampedArray, n: number) {
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0, p = 0; p < n; i += 4, p++) {
    const a = data[i + 3] / 255;
    r[p] = srgbToLinear(data[i] / 255) * a;
    g[p] = srgbToLinear(data[i + 1] / 255) * a;
    b[p] = srgbToLinear(data[i + 2] / 255) * a;
  }
  return { r, g, b };
}

function planesToRgba(r: Float32Array, g: Float32Array, b: Float32Array) {
  const out = new Uint8ClampedArray(r.length * 4);
  for (let p = 0, i = 0; p < r.length; p++, i += 4) {
    out[i] = clamp(Math.round(linearToSrgb(r[p]) * 255), 0, 255);
    out[i + 1] = clamp(Math.round(linearToSrgb(g[p]) * 255), 0, 255);
    out[i + 2] = clamp(Math.round(linearToSrgb(b[p]) * 255), 0, 255);
    out[i + 3] = 255;
  }
  return out;
}

/** Gaussian defocus in linear light, matching the deconvolution model. */
export function defocusBlurRgba(data: Uint8ClampedArray, w: number, h: number, sigma: number) {
  const { r, g, b } = rgbaToLinear(data, w * h);
  const k = gaussianKernel(sigma);
  const rb = convolveSeparable(r, w, h, k);
  const gb = convolveSeparable(g, w, h, k);
  const bb = convolveSeparable(b, w, h, k);
  return planesToRgba(rb, gb, bb);
}

export function restoreImage(input: RgbaImage, options: RestoreOptions, onProgress?: ProgressFn): { image: RgbaImage; report: RestoreReport } {
  const t0 = Date.now();
  const w = input.width;
  const h = input.height;
  if (w < 2 || h < 2) throw new Error("Image is too small to restore.");
  onProgress?.("Reading pixels", 0.06);
  const planes = rgbaToLinear(input.data, w * h);
  let { Y, Cb, Cr } = rgbToYCbCr(planes.r, planes.g, planes.b);

  const denoise = clamp(options.denoise, 0, 1);
  if (denoise > 0.02) {
    onProgress?.("Denoising color", 0.14);
    const chromaSigma = 0.35 + denoise * 1.5;
    const ck = gaussianKernel(chromaSigma);
    Cb = convolveSeparable(Cb, w, h, ck);
    Cr = convolveSeparable(Cr, w, h, ck);
  }
  if (denoise > 0.62) {
    onProgress?.("Smoothing grain", 0.22);
    const blend = clamp((denoise - 0.62) / 0.38, 0, 1) * 0.85;
    const smooth = bilateral(Y, w, h, 2, 1.1, 0.08);
    for (let i = 0; i < Y.length; i++) Y[i] = Y[i] * (1 - blend) + smooth[i] * blend;
  }

  const model = options.model;
  let sigma = clamp(options.sigma, 0.4, 3.4);
  let depth = clamp(Math.round(options.iterations), 2, 10);
  const motionLength = clamp(Math.round(options.motionLength), 3, 31);
  const motionAngle = options.motionAngle;

  if (model === "auto") {
    onProgress?.("Estimating blur", 0.3);
    sigma = estimateDefocusSigma(Y, w, h);
    if (sigma < 0.85) depth = Math.min(depth, 3);
  }

  onProgress?.("Inverting blur", 0.4);
  const K = regularizationFor(depth);
  if (model === "motion") {
    Y = wienerDeblur(Y, w, h, K, paintMotion(motionLength, motionAngle));
  } else {
    Y = wienerDeblur(Y, w, h, K, paintGaussian(sigma));
  }
  onProgress?.("Inverting blur", 0.72);

  for (let i = 0; i < Y.length; i++) Y[i] = clamp(Y[i], 0, 1.2);

  const detail = clamp(options.detail, 0, 1);
  if (detail > 0.02) {
    onProgress?.("Recovering local contrast", 0.8);
    Y = clahe(Y, w, h, detail * 0.38);
  }

  let { r, g, b } = ycbcrToRgb(Y, Cb, Cr);

  const ow = Math.max(2, options.outWidth);
  const oh = Math.max(2, options.outHeight);
  if (ow !== w || oh !== h) {
    onProgress?.("Upscaling", 0.86);
    r = resizeLanczos(r, w, h, ow, oh);
    g = resizeLanczos(g, w, h, ow, oh);
    b = resizeLanczos(b, w, h, ow, oh);
  }

  if (detail > 0.02) {
    onProgress?.("Sharpening edges", 0.94);
    const sr = new Float32Array(r.length);
    const sg = new Float32Array(g.length);
    const sb = new Float32Array(b.length);
    for (let i = 0; i < r.length; i++) {
      sr[i] = linearToSrgb(r[i]);
      sg[i] = linearToSrgb(g[i]);
      sb[i] = linearToSrgb(b[i]);
    }
    unsharp(sr, sg, sb, ow, oh, detail * 0.85);
    for (let i = 0; i < r.length; i++) {
      r[i] = srgbToLinear(sr[i]);
      g[i] = srgbToLinear(sg[i]);
      b[i] = srgbToLinear(sb[i]);
    }
  }

  onProgress?.("Finishing", 0.98);
  const data = planesToRgba(r, g, b);
  return {
    image: { data, width: ow, height: oh },
    report: {
      model,
      sigma: Math.round(sigma * 100) / 100,
      motionLength,
      motionAngle,
      iterations: depth,
      stack: "frequency",
      width: ow,
      height: oh,
      ms: Date.now() - t0,
    },
  };
}
