import { pipeline, env, type DepthEstimationPipeline } from '@huggingface/transformers';

// Fetch model files from the Hugging Face Hub (no local model copies).
env.allowLocalModels = false;
// NOTE: The ONNX Runtime Web WASM binary is loaded from the jsdelivr CDN at
// runtime via Transformers.js' default `wasmPaths` (see its src/backends/onnx.js).
// We intentionally do NOT override `env.backends.onnx.wasm.wasmPaths` here, so
// the library resolves the correct CDN paths/versions for both the WebGPU
// (asyncify) and WASM backends. The Vite config strips the unused bundled
// `.wasm` asset so the production bundle stays within deploy limits.

export type Backend = 'webgpu' | 'wasm';

export type Stage =
  | 'idle'
  | 'initializing'
  | 'downloading'
  | 'running'
  | 'done'
  | 'error';

export interface ProgressInfo {
  stage: Stage;
  /** Overall download progress across all model files, 0..1. null when unknown. */
  progress: number | null;
  /** Human-readable status message. */
  message: string;
  /** Current backend once determined. */
  backend: Backend | null;
  /** Error message when stage === 'error'. */
  error?: string;
}

export interface DepthResult {
  /** Grayscale depth map as RGBA pixels. */
  data: Uint8ClampedArray;
  width: number;
  height: number;
  backend: Backend;
  /** Normalized depth values (0..1), kept for a future 3D geometry stage. */
  depthValues: Float32Array;
}

type ProgressCallback = (info: ProgressInfo) => void;

interface FileProgress {
  progress: number;
  total: number;
  loaded: number;
}

// Accumulate per-file progress so the bar reflects overall download, not one file.
function makeProgressTracker(onProgress: (overall: number | null) => void) {
  const files = new Map<string, FileProgress>();

  return (data: unknown) => {
    if (!data || typeof data !== 'object') return;
    const d = data as Record<string, unknown>;
    const status = d.status as string | undefined;
    const file = (d.file as string | undefined) ?? 'unknown';

    if (status === 'initiate' || status === 'download') {
      const p = typeof d.progress === 'number' ? d.progress : 0;
      const total = typeof d.total === 'number' ? d.total : 0;
      const loaded = typeof d.loaded === 'number' ? d.loaded : 0;
      files.set(file, { progress: p, total, loaded });
    } else if (status === 'done' || status === 'ready') {
      const prev = files.get(file);
      files.set(file, { progress: 1, total: prev?.total ?? 0, loaded: prev?.total ?? 0 });
    }

    let weighted = 0;
    let totalBytes = 0;
    for (const f of files.values()) {
      if (f.total > 0) {
        weighted += f.progress * f.total;
        totalBytes += f.total;
      } else if (f.progress > 0) {
        weighted += f.progress;
        totalBytes += 1;
      }
    }
    onProgress(totalBytes > 0 ? Math.min(1, weighted / totalBytes) : null);
  };
}

let cached: { backend: Backend; pipe: DepthEstimationPipeline } | null = null;

async function createPipeline(
  backend: Backend,
  onProgress: ProgressCallback,
): Promise<DepthEstimationPipeline> {
  const tracker = makeProgressTracker((overall) => {
    onProgress({
      stage: 'downloading',
      progress: overall,
      message: overall !== null
        ? `Downloading model weights (${Math.round(overall * 100)}%)`
        : 'Downloading model weights',
      backend,
    });
  });

  onProgress({
    stage: 'initializing',
    progress: null,
    message: 'Initializing AI model',
    backend,
  });

  const pipe = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small', {
    device: backend,
    progress_callback: tracker,
  });

  return pipe as unknown as DepthEstimationPipeline;
}

/**
 * Initialize the depth-estimation pipeline, attempting WebGPU first and
 * falling back to WASM on any initialization or inference failure.
 */
export async function getDepthPipeline(onProgress: ProgressCallback): Promise<{
  backend: Backend;
  pipe: DepthEstimationPipeline;
}> {
  if (cached) {
    onProgress({
      stage: 'initializing',
      progress: null,
      message: 'Model already loaded',
      backend: cached.backend,
    });
    return cached;
  }

  // Attempt WebGPU first.
  try {
    const pipe = await createPipeline('webgpu', onProgress);
    cached = { backend: 'webgpu', pipe };
    return cached;
  } catch {
    onProgress({
      stage: 'initializing',
      progress: null,
      message: 'WebGPU unavailable, falling back to WASM',
      backend: 'wasm',
    });
  }

  // Fallback to WASM.
  try {
    const pipe = await createPipeline('wasm', onProgress);
    cached = { backend: 'wasm', pipe };
    return cached;
  } catch (err) {
    cached = null;
    onProgress({
      stage: 'error',
      progress: null,
      message: 'Model loading or inference failed',
      backend: null,
      error: err instanceof Error ? err.message : String(err),
    });
    throw err;
  }
}

function normalizeToGrayscale(depth: Float32Array): {
  data: Uint8ClampedArray;
  normalized: Float32Array;
} {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < depth.length; i++) {
    const v = depth[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const normalized = new Float32Array(depth.length);
  const rgba = new Uint8ClampedArray(depth.length * 4);
  for (let i = 0; i < depth.length; i++) {
    const g = (depth[i] - min) / range; // 0..1
    normalized[i] = g;
    const v = Math.round(g * 255);
    rgba[i * 4] = v;
    rgba[i * 4 + 1] = v;
    rgba[i * 4 + 2] = v;
    rgba[i * 4 + 3] = 255;
  }
  return { data: rgba, normalized };
}

function extractDepth(output: unknown): {
  depthValues: Float32Array;
  width: number;
  height: number;
} {
  const result = output as {
    depth?: { data: ArrayLike<number>; width: number; height: number; channels: number };
    predicted_depth?: { data: ArrayLike<number> | number[]; dims: number[] };
  };

  if (!result?.depth) {
    throw new Error('Model did not return a depth map');
  }

  const pd = result.predicted_depth;
  if (pd && pd.data && pd.dims && pd.dims.length >= 2) {
    const dims = pd.dims;
    const h = dims[dims.length - 2];
    const w = dims[dims.length - 1];
    const src = pd.data;
    return {
      depthValues: src instanceof Float32Array ? src : new Float32Array(src),
      width: w,
      height: h,
    };
  }

  const d = result.depth;
  const depthValues = new Float32Array(d.data.length / d.channels);
  for (let i = 0; i < depthValues.length; i++) {
    let sum = 0;
    for (let c = 0; c < d.channels; c++) sum += d.data[i * d.channels + c];
    depthValues[i] = sum / d.channels;
  }
  return { depthValues, width: d.width, height: d.height };
}

/**
 * Run depth estimation on an image source (URL, Blob, or HTMLImageElement).
 * Attempts WebGPU first, falls back to WASM on inference failure.
 */
export async function runDepthEstimation(
  image: string | Blob,
  onProgress: ProgressCallback,
): Promise<DepthResult> {
  const { backend, pipe } = await getDepthPipeline(onProgress);

  onProgress({
    stage: 'running',
    progress: null,
    message: 'Running depth estimation',
    backend,
  });

  let output;
  try {
    output = await pipe(image);
  } catch (err) {
    // If WebGPU inference fails, retry the whole flow with WASM.
    if (backend === 'webgpu') {
      cached = null;
      onProgress({
        stage: 'initializing',
        progress: null,
        message: 'WebGPU inference failed, falling back to WASM',
        backend: 'wasm',
      });
      const { backend: wasmBackend, pipe: wasmPipe } = await getDepthPipeline(onProgress);
      onProgress({
        stage: 'running',
        progress: null,
        message: 'Running depth estimation',
        backend: wasmBackend,
      });
      output = await wasmPipe(image);
      const r = extractDepth(output);
      const { data, normalized } = normalizeToGrayscale(r.depthValues);
      onProgress({
        stage: 'done',
        progress: 1,
        message: 'Depth map generated',
        backend: wasmBackend,
      });
      return { data, width: r.width, height: r.height, backend: wasmBackend, depthValues: normalized };
    }
    throw err;
  }

  const r = extractDepth(output);
  const { data, normalized } = normalizeToGrayscale(r.depthValues);
  onProgress({
    stage: 'done',
    progress: 1,
    message: 'Depth map generated',
    backend,
  });
  return { data, width: r.width, height: r.height, backend, depthValues: normalized };
}

/** Reset cached pipeline (used on explicit reset). */
export function resetPipeline(): void {
  cached = null;
}
