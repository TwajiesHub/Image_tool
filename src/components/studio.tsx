import { useEffect, useMemo, useRef, useState } from "react";
import { Aperture, Camera, ChevronsLeftRight, Download, ImagePlus, Upload } from "lucide-react";
import {
  defocusBlurRgba,
  planSizes,
  type BlurModel,
  type RestoreOptions,
  type RestoreReport,
  type ScaleFactor,
} from "@/lib/restore";

type PresetId = "soft" | "standard" | "strong" | "motion";

type Settings = {
  scale: ScaleFactor;
  model: BlurModel;
  sigma: number;
  motionLength: number;
  motionAngle: number;
  iterations: number;
  denoise: number;
  detail: number;
};

type Loaded = {
  url: string;
  srcW: number;
  srcH: number;
  pixels: Uint8ClampedArray;
  pw: number;
  ph: number;
  note: string | null;
};

type ResultView = {
  url: string;
  width: number;
  height: number;
  report: RestoreReport;
};

const PRESETS: Record<PresetId, { label: string; hint: string; settings: Settings }> = {
  soft: {
    label: "Soft",
    hint: "Light pass. Safer on faces and sky.",
    settings: { scale: 2, model: "auto", sigma: 1.2, motionLength: 9, motionAngle: 0, iterations: 3, denoise: 0.45, detail: 0.32 },
  },
  standard: {
    label: "Standard",
    hint: "Balanced deconvolution and detail.",
    settings: { scale: 2, model: "auto", sigma: 1.6, motionLength: 13, motionAngle: 8, iterations: 5, denoise: 0.4, detail: 0.55 },
  },
  strong: {
    label: "Strong",
    hint: "Heavier blur. Watch for bright rings.",
    settings: { scale: 2, model: "auto", sigma: 2.4, motionLength: 17, motionAngle: 0, iterations: 8, denoise: 0.28, detail: 0.72 },
  },
  motion: {
    label: "Motion",
    hint: "Smear along a direction, not a soft lens.",
    settings: { scale: 2, model: "motion", sigma: 1.6, motionLength: 13, motionAngle: 0, iterations: 6, denoise: 0.48, detail: 0.48 },
  },
};

function Segment<T extends string | number>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div>
      <div className="mb-2 text-sm text-muted">{label}</div>
      <div role="radiogroup" aria-label={label} className="grid grid-cols-3 gap-1 rounded-full bg-bg p-1">
        {options.map((opt) => {
          const on = opt.value === value;
          return (
            <button
              key={String(opt.value)}
              type="button"
              role="radio"
              aria-checked={on}
              onClick={() => onChange(opt.value)}
              className={
                "h-11 rounded-full text-sm font-medium transition-colors " +
                (on ? "bg-primary text-primary-fg" : "text-muted hover:text-fg")
              }
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Field({
  id,
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-sm text-fg">
          {label}
        </label>
        <span className="font-serif text-xl leading-none tabular-nums text-primary">{display}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

async function readFile(file: File): Promise<Loaded> {
  if (file.type && !file.type.startsWith("image/")) {
    throw new Error("Use a JPG, PNG, or WebP photo.");
  }
  if (file.size > 40 * 1024 * 1024) {
    throw new Error("That file is over 40 MB. Export a smaller copy and try again.");
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const srcW = img.naturalWidth;
    const srcH = img.naturalHeight;
    if (srcW < 16 || srcH < 16) throw new Error("That image is too small to restore.");
    const plan = planSizes(srcW, srcH, 1);
    const canvas = document.createElement("canvas");
    canvas.width = plan.pw;
    canvas.height = plan.ph;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("This browser could not read the photo.");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, plan.pw, plan.ph);
    const pixels = new Uint8ClampedArray(ctx.getImageData(0, 0, plan.pw, plan.ph).data);
    return {
      url,
      srcW,
      srcH,
      pixels,
      pw: plan.pw,
      ph: plan.ph,
      note: plan.scaledDown
        ? "Large photo. Restoration runs on a 960px working image, then scales to the export size."
        : null,
    };
  } catch (error) {
    URL.revokeObjectURL(url);
    if (error instanceof Error && error.message.startsWith("That")) throw error;
    if (error instanceof Error && error.message.startsWith("This")) throw error;
    if (error instanceof Error && error.message.startsWith("Large")) throw error;
    throw new Error("Could not read that image. Try JPG or PNG.");
  }
}

function drawPlate(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = "#17140f";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#2c261e";
  ctx.fillRect(24, 24, w - 48, h - 48);

  const cx = 176;
  const cy = h / 2;
  const spokes = 40;
  const radius = 132;
  for (let i = 0; i < spokes; i++) {
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius, (i * 2 * Math.PI) / spokes, ((i + 0.48) * 2 * Math.PI) / spokes);
    ctx.closePath();
    ctx.fillStyle = i % 2 === 0 ? "#f4efe6" : "#14110d";
    ctx.fill();
  }
  ctx.beginPath();
  ctx.arc(cx, cy, 16, 0, Math.PI * 2);
  ctx.fillStyle = "#e4a04a";
  ctx.fill();

  const groups = [2, 3, 4, 6, 8];
  groups.forEach((pitch, index) => {
    const y = 52 + index * 58;
    for (let k = 0; k < 28; k++) {
      ctx.fillStyle = k % 2 === 0 ? "#f6f1e8" : "#14110d";
      ctx.fillRect(340 + k * pitch, y, pitch, 40);
    }
  });

  ctx.fillStyle = "#e4a04a";
  ctx.font = "italic 64px Georgia, serif";
  ctx.fillText("Lucid", 338, 362);
  ctx.fillStyle = "#f4efe6";
  ctx.font = "18px Georgia, serif";
  ctx.fillText("defocus plate", 508, 362);
}

async function makeSample(): Promise<Loaded> {
  const w = 640;
  const h = 400;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Could not build the sample plate.");
  drawPlate(ctx, w, h);
  const sharp = ctx.getImageData(0, 0, w, h);
  const blurred = defocusBlurRgba(sharp.data, w, h, 1.8);
  ctx.putImageData(new ImageData(blurred, w, h), 0, 0);
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((value) => (value ? resolve(value) : reject(new Error("Could not build the sample plate."))), "image/png");
  });
  return {
    url: URL.createObjectURL(blob),
    srcW: w,
    srcH: h,
    pixels: blurred,
    pw: w,
    ph: h,
    note: "Sample plate, defocused on purpose at σ 1.8. Defocus mode is set to match.",
  };
}

function formatMs(ms: number) {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export function Studio() {
  const [preset, setPreset] = useState<PresetId | "custom">("standard");
  const [settings, setSettings] = useState<Settings>(PRESETS.standard.settings);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [result, setResult] = useState<ResultView | null>(null);
  const [phase, setPhase] = useState<"idle" | "working" | "ready" | "done">("idle");
  const [stage, setStage] = useState("Preparing");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [split, setSplit] = useState(0.5);
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const cameraRef = useRef<HTMLInputElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const workerRef = useRef<Worker | null>(null);
  const jobRef = useRef(0);
  const loadedRef = useRef<Loaded | null>(null);

  useEffect(() => {
    loadedRef.current = loaded;
  }, [loaded]);

  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      if (loadedRef.current) URL.revokeObjectURL(loadedRef.current.url);
    };
  }, []);

  const plan = useMemo(
    () => (loaded ? planSizes(loaded.srcW, loaded.srcH, settings.scale) : null),
    [loaded, settings.scale],
  );

  function patch(partial: Partial<Settings>, nextPreset: PresetId | "custom" = "custom") {
    setPreset(nextPreset);
    setSettings((current) => ({ ...current, ...partial }));
  }

  function applyPreset(id: PresetId) {
    setPreset(id);
    setSettings(PRESETS[id].settings);
  }

  function adopt(next: Loaded) {
    jobRef.current += 1;
    workerRef.current?.terminate();
    setLoaded((current) => {
      if (current) URL.revokeObjectURL(current.url);
      return next;
    });
    setResult((current) => {
      if (current) URL.revokeObjectURL(current.url);
      return null;
    });
    setSplit(0.5);
    setError(null);
    setPhase("ready");
  }

  async function onFiles(list: FileList | null) {
    const file = list?.[0];
    if (!file) return;
    try {
      adopt(await readFile(file));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not read that image.");
    }
  }

  async function onSample() {
    try {
      adopt(await makeSample());
      setPreset("custom");
      setSettings({
        scale: 2,
        model: "defocus",
        sigma: 1.8,
        motionLength: 13,
        motionAngle: 0,
        iterations: 6,
        denoise: 0.15,
        detail: 0.6,
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not build the sample plate.");
    }
  }

  function restore() {
    if (!loaded || phase === "working") return;
    const id = ++jobRef.current;
    workerRef.current?.terminate();
    const worker = new Worker(new URL("../lib/restore.worker.ts", import.meta.url), { type: "module" });
    workerRef.current = worker;
    setPhase("working");
    setStage("Preparing");
    setProgress(0.02);
    setError(null);

    const sizes = planSizes(loaded.srcW, loaded.srcH, settings.scale);
    const options: RestoreOptions = {
      ...settings,
      outWidth: sizes.ow,
      outHeight: sizes.oh,
    };
    const copy = new Uint8ClampedArray(loaded.pixels);

    worker.onmessage = (event: MessageEvent) => {
      if (jobRef.current !== id) return;
      const msg = event.data as
        | { type: "progress"; stage: string; value: number }
        | { type: "error"; message: string }
        | { type: "done"; buffer: ArrayBuffer; width: number; height: number; report: RestoreReport };
      if (msg.type === "progress") {
        setStage(msg.stage);
        setProgress(msg.value);
        return;
      }
      if (msg.type === "error") {
        worker.terminate();
        setError(msg.message);
        setPhase(result ? "done" : "ready");
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = msg.width;
      canvas.height = msg.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        setError("Could not draw the restored image.");
        setPhase("ready");
        worker.terminate();
        return;
      }
      ctx.putImageData(new ImageData(new Uint8ClampedArray(msg.buffer), msg.width, msg.height), 0, 0);
      canvas.toBlob((blob) => {
        if (jobRef.current !== id) return;
        worker.terminate();
        if (!blob) {
          setError("Could not export the restored image.");
          setPhase("ready");
          return;
        }
        const url = URL.createObjectURL(blob);
        setResult((current) => {
          if (current) URL.revokeObjectURL(current.url);
          return { url, width: msg.width, height: msg.height, report: msg.report };
        });
        setProgress(1);
        setStage("Done");
        setPhase("done");
      }, "image/png");
    };

    worker.onerror = () => {
      if (jobRef.current !== id) return;
      worker.terminate();
      setError("The restore stopped unexpectedly. Try fewer passes or a smaller photo.");
      setPhase(result ? "done" : "ready");
    };

    worker.postMessage(
      { buffer: copy.buffer, width: loaded.pw, height: loaded.ph, options },
      [copy.buffer],
    );
  }

  function moveSplit(clientX: number) {
    const rect = frameRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return;
    setSplit(Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)));
  }

  function download() {
    if (!result) return;
    const link = document.createElement("a");
    link.href = result.url;
    link.download = `lucid-${settings.scale}x.png`;
    link.click();
  }

  const showCompare = Boolean(loaded && result && phase !== "working");
  const aspectW = result?.width ?? loaded?.srcW ?? 4;
  const aspectH = result?.height ?? loaded?.srcH ?? 3;

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 px-4 py-6 lg:px-8 lg:py-8">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <span className="flex size-11 items-center justify-center rounded-full bg-primary text-primary-fg">
            <Aperture className="size-5" aria-hidden="true" />
          </span>
          <div>
            <p className="text-xs font-medium tracking-widest text-primary">ANTI-BLUR</p>
            <h1 className="font-serif text-5xl leading-none text-fg">Lucid</h1>
          </div>
        </div>
        <p className="max-w-xl text-pretty leading-relaxed text-muted">
          Deep anti-blur for soft photos. A Wiener inverse lifts blur in frequency space, then Lanczos
          builds the larger file. Everything stays in this browser.
        </p>
      </header>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <section className="relative flex h-full flex-col overflow-hidden rounded-panel bg-surface ring-1 ring-line">
          <span className="pointer-events-none absolute top-3 left-3 z-10 h-4 w-4 border-t-2 border-l-2 border-primary" />
          <span className="pointer-events-none absolute top-3 right-3 z-10 h-4 w-4 border-t-2 border-r-2 border-primary" />
          <span className="pointer-events-none absolute bottom-3 left-3 z-10 h-4 w-4 border-b-2 border-l-2 border-primary" />
          <span className="pointer-events-none absolute right-3 bottom-3 z-10 h-4 w-4 border-r-2 border-b-2 border-primary" />

          {!loaded ? (
            <div
              className={
                "stage-frame flex flex-col items-center justify-center gap-4 px-6 py-16 text-center " +
                (dragOver ? "bg-raised" : "")
              }
              onDragOver={(event) => {
                event.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragOver(false);
                void onFiles(event.dataTransfer.files);
              }}
            >
              <ImagePlus className="size-8 text-primary" aria-hidden="true" />
              <div>
                <h2 className="font-serif text-3xl text-balance text-fg">Drop a blurry photo</h2>
                <p className="mt-2 text-sm text-pretty text-muted">JPG, PNG, or WebP. Nothing is uploaded.</p>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2">
                <button type="button" className="h-12 rounded-full bg-primary px-5 font-semibold text-primary-fg" onClick={() => fileRef.current?.click()}>
                  <span className="inline-flex items-center gap-2">
                    <Upload className="size-4" aria-hidden="true" />
                    Upload
                  </span>
                </button>
                <button type="button" className="h-12 rounded-full bg-raised px-5 font-medium text-fg ring-1 ring-line" onClick={() => cameraRef.current?.click()}>
                  <span className="inline-flex items-center gap-2">
                    <Camera className="size-4" aria-hidden="true" />
                    Camera
                  </span>
                </button>
                <button type="button" className="h-12 rounded-full px-5 font-medium text-muted" onClick={() => void onSample()}>
                  Try the plate
                </button>
              </div>
            </div>
          ) : (
            <div
              ref={frameRef}
              className="stage-frame relative w-full touch-none bg-bg"
              style={{ aspectRatio: `${aspectW} / ${aspectH}` }}
              onPointerDown={(event) => {
                if (!result) return;
                event.currentTarget.setPointerCapture(event.pointerId);
                moveSplit(event.clientX);
              }}
              onPointerMove={(event) => {
                if (!result || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
                moveSplit(event.clientX);
              }}
              onDragOver={(event) => {
                event.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragOver(false);
                void onFiles(event.dataTransfer.files);
              }}
            >
              <img
                src={result?.url ?? loaded.url}
                alt={result ? "Restored photograph" : "Original photograph"}
                className="absolute inset-0 h-full w-full object-contain"
                draggable={false}
              />
              {showCompare && (
                <img
                  src={loaded.url}
                  alt=""
                  className="absolute inset-0 h-full w-full object-contain"
                  style={{ clipPath: `inset(0 ${((1 - split) * 100).toFixed(2)}% 0 0)` }}
                  draggable={false}
                />
              )}
              {showCompare && (
                <>
                  <div className="pointer-events-none absolute inset-y-0 z-10" style={{ left: `${split * 100}%` }}>
                    <div className="absolute inset-y-0 -ml-px w-0.5 bg-primary" />
                    <div className="absolute top-1/2 -ml-5 flex size-11 -translate-y-1/2 items-center justify-center rounded-full bg-primary text-primary-fg shadow-none">
                      <ChevronsLeftRight className="size-5" aria-hidden="true" />
                    </div>
                  </div>
                  <span className="pointer-events-none absolute top-4 left-4 z-10 rounded-full bg-bg px-3 py-1 text-xs font-medium tracking-widest text-muted">
                    BEFORE
                  </span>
                  <span className="pointer-events-none absolute top-4 right-4 z-10 rounded-full bg-bg px-3 py-1 text-xs font-medium tracking-widest text-primary">
                    AFTER
                  </span>
                </>
              )}
              {phase === "working" && (
                <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-3 bg-bg/80 px-6 text-center">
                  <p className="font-serif text-3xl text-fg">{stage}</p>
                  <div className="h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-raised" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
                    <div className="bar h-full bg-primary" style={{ width: `${Math.round(progress * 100)}%` }} />
                  </div>
                  <p className="text-sm text-muted">Frequency inverse. This usually takes a few seconds.</p>
                </div>
              )}
            </div>
          )}
          {loaded && (
            <p className="px-4 py-3 text-sm text-pretty text-muted">
              {result && phase !== "working"
                ? "Drag the handle. Left is the original, right is restored."
                : "Ready. Set the scale, then restore."}
            </p>
          )}
        </section>

        <section className="flex flex-col gap-5 rounded-panel bg-surface p-4 ring-1 ring-line lg:p-5">
          <div>
            <h2 className="font-serif text-2xl text-fg">Restore</h2>
            <p className="mt-1 text-sm text-pretty text-muted">
              {preset === "custom" ? "Custom stack." : PRESETS[preset].hint}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {(Object.keys(PRESETS) as PresetId[]).map((id) => {
              const on = preset === id;
              return (
                <button
                  key={id}
                  type="button"
                  aria-pressed={on}
                  onClick={() => applyPreset(id)}
                  className={
                    "h-11 rounded-full text-sm font-medium transition-colors " +
                    (on ? "bg-primary text-primary-fg" : "bg-raised text-muted hover:text-fg")
                  }
                >
                  {PRESETS[id].label}
                </button>
              );
            })}
          </div>

          <Segment
            label="Export scale"
            value={settings.scale}
            options={[
              { value: 1 as ScaleFactor, label: "1×" },
              { value: 2 as ScaleFactor, label: "2×" },
              { value: 4 as ScaleFactor, label: "4×" },
            ]}
            onChange={(scale) => patch({ scale })}
          />
          {plan && (
            <p className="text-sm tabular-nums text-muted">
              {loaded?.srcW}×{loaded?.srcH} source → {plan.ow}×{plan.oh} export
              {plan.capped ? " · capped at 3200px" : ""}
            </p>
          )}

          <Segment
            label="Blur model"
            value={settings.model}
            options={[
              { value: "auto" as BlurModel, label: "Auto" },
              { value: "defocus" as BlurModel, label: "Defocus" },
              { value: "motion" as BlurModel, label: "Motion" },
            ]}
            onChange={(model) => patch({ model })}
          />

          {settings.model === "defocus" && (
            <Field id="sigma" label="Defocus radius" min={0.4} max={3.4} step={0.1} value={settings.sigma} display={settings.sigma.toFixed(1)} onChange={(sigma) => patch({ sigma })} />
          )}
          {settings.model === "motion" && (
            <>
              <Field id="motion-length" label="Smear length" min={3} max={31} step={2} value={settings.motionLength} display={`${Math.round(settings.motionLength)} px`} onChange={(motionLength) => patch({ motionLength })} />
              <Field id="motion-angle" label="Smear angle" min={0} max={180} step={1} value={settings.motionAngle} display={`${Math.round(settings.motionAngle)}°`} onChange={(motionAngle) => patch({ motionAngle })} />
            </>
          )}
          {settings.model === "auto" && (
            <p className="text-sm text-pretty text-muted">Radius is measured from edge width before the passes run.</p>
          )}

          <Field id="depth" label="Deconvolution depth" min={2} max={10} step={1} value={settings.iterations} display={String(Math.round(settings.iterations))} onChange={(iterations) => patch({ iterations })} />
          <Field id="denoise" label="Denoise" min={0} max={1} step={0.01} value={settings.denoise} display={String(Math.round(settings.denoise * 100))} onChange={(denoise) => patch({ denoise })} />
          <Field id="detail" label="Detail" min={0} max={1} step={0.01} value={settings.detail} display={String(Math.round(settings.detail * 100))} onChange={(detail) => patch({ detail })} />

          {loaded?.note && <p className="text-sm text-pretty text-muted">{loaded.note}</p>}
          {error && <p className="text-sm text-pretty text-fg">{error}</p>}
          {result && phase !== "working" && (
            <p className="text-sm text-pretty tabular-nums text-muted">
              σ {result.report.sigma.toFixed(2)} · depth {result.report.iterations} · frequency inverse · {formatMs(result.report.ms)}
              {result.report.model === "motion" ? ` · ${result.report.motionLength}px @ ${Math.round(result.report.motionAngle)}°` : ""}
            </p>
          )}

          <div className="flex flex-col gap-2">
            <button
              type="button"
              disabled={!loaded || phase === "working"}
              onClick={restore}
              className="h-12 rounded-full bg-primary font-semibold text-primary-fg disabled:opacity-40"
            >
              {phase === "working" ? "Restoring…" : result ? "Restore again" : "Restore and upscale"}
            </button>
            <div className="grid grid-cols-2 gap-2">
              <button type="button" className="h-12 rounded-full bg-raised font-medium text-fg ring-1 ring-line" onClick={() => fileRef.current?.click()}>
                New photo
              </button>
              <button type="button" disabled={!result} onClick={download} className="h-12 rounded-full bg-raised font-medium text-fg ring-1 ring-line disabled:opacity-40">
                <span className="inline-flex items-center justify-center gap-2">
                  <Download className="size-4" aria-hidden="true" />
                  PNG
                </span>
              </button>
            </div>
          </div>
        </section>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/*"
        className="sr-only"
        onChange={(event) => {
          void onFiles(event.target.files);
          event.target.value = "";
        }}
      />
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        onChange={(event) => {
          void onFiles(event.target.files);
          event.target.value = "";
        }}
      />
    </main>
  );
}
