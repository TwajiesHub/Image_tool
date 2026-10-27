import { restoreImage, type RestoreOptions } from "./restore";

type RequestMessage = {
  buffer: ArrayBuffer;
  width: number;
  height: number;
  options: RestoreOptions;
};

type WorkerHost = {
  onmessage: ((event: MessageEvent<RequestMessage>) => void) | null;
  postMessage: (data: unknown, transfer?: Transferable[]) => void;
};

const ctx = self as unknown as WorkerHost;

ctx.onmessage = (event: MessageEvent<RequestMessage>) => {
  try {
    const { buffer, width, height, options } = event.data;
    const data = new Uint8ClampedArray(buffer);
    const result = restoreImage({ data, width, height }, options, (stage, value) => {
      ctx.postMessage({ type: "progress", stage, value });
    });
    const out = result.image.data.buffer;
    ctx.postMessage(
      {
        type: "done",
        buffer: out,
        width: result.image.width,
        height: result.image.height,
        report: result.report,
      },
      [out],
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : "Restore failed.";
    ctx.postMessage({ type: "error", message });
  }
};
