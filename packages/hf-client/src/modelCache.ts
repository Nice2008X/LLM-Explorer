const DB_NAME = "Tensorium-model-cache";
const DB_VERSION = 1;
const STORE_NAME = "files";

/**
 * Files larger than this are never written to the cache — fetched fresh
 * from Hugging Face every time instead. Keeps one oversized checkpoint from
 * burning through a large share of the browser's per-origin storage quota,
 * and avoids a slow structured-clone IndexedDB write on the main thread for
 * something that isn't one of this app's small "tiny-random" presets.
 */
export const MAX_CACHEABLE_BYTES = 50 * 1024 * 1024; // 50 MB

interface CachedFile {
  url: string;
  bytes: ArrayBuffer;
  size: number;
  cachedAt: number;
}

let dbPromise: Promise<IDBDatabase | null> | null = null;

/** Opens (or creates) the cache database. Resolves to null rather than rejecting on any failure — IndexedDB can be unavailable or disabled (private browsing in some browsers, storage restrictions), and that should just mean "no cache", never a load failure. */
function openDb(): Promise<IDBDatabase | null> {
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      if (typeof indexedDB === "undefined") {
        resolve(null);
        return;
      }
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE_NAME, { keyPath: "url" });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
  }
  return dbPromise;
}

async function getCached(url: string): Promise<ArrayBuffer | undefined> {
  const db = await openDb();
  if (!db) return undefined;
  return new Promise((resolve) => {
    const req = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(url);
    req.onsuccess = () => resolve((req.result as CachedFile | undefined)?.bytes);
    req.onerror = () => resolve(undefined);
  });
}

async function putCached(url: string, bytes: ArrayBuffer): Promise<void> {
  const db = await openDb();
  if (!db) return;
  return new Promise((resolve) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const entry: CachedFile = { url, bytes, size: bytes.byteLength, cachedAt: Date.now() };
    tx.objectStore(STORE_NAME).put(entry);
    // A write failure (quota exceeded despite the size check above, private
    // mode restrictions) should never break loading the model that's
    // already sitting in memory — just means it won't be cached this time.
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
  });
}

/** `loadedBytes` counts up as the response streams in; `totalBytes` is undefined when the server doesn't send Content-Length. */
export type ByteProgressCallback = (loadedBytes: number, totalBytes: number | undefined) => void;

/**
 * Fetches `url` as raw bytes, transparently caching the result in
 * IndexedDB (skipping anything over MAX_CACHEABLE_BYTES) so a re-click of
 * the same preset loads instantly with no network request at all —
 * `fetchJson` and `fetchArrayBuffer` both route through this, so
 * config.json/tokenizer.json/model.safetensors are all covered uniformly.
 *
 * A cache hit reports `onProgress` once, immediately, at 100% — there's no
 * network transfer to time, but callers shouldn't have to special-case that.
 */
export async function fetchCachedArrayBuffer(url: string, onProgress?: ByteProgressCallback): Promise<ArrayBuffer> {
  const cached = await getCached(url);
  if (cached) {
    onProgress?.(cached.byteLength, cached.byteLength);
    return cached;
  }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}`);

  // Streaming reads are only worth the extra bookkeeping when someone's
  // actually watching — otherwise a plain arrayBuffer() is simpler and just
  // as fast.
  if (!onProgress || !res.body) {
    const bytes = await res.arrayBuffer();
    onProgress?.(bytes.byteLength, bytes.byteLength);
    if (bytes.byteLength <= MAX_CACHEABLE_BYTES) await putCached(url, bytes);
    return bytes;
  }

  const totalHeader = res.headers.get("content-length");
  const totalBytes = totalHeader ? Number(totalHeader) : undefined;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loadedBytes += value.byteLength;
    onProgress(loadedBytes, totalBytes);
  }
  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (bytes.byteLength <= MAX_CACHEABLE_BYTES) await putCached(url, bytes.buffer);
  return bytes.buffer;
}
