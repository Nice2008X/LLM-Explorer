import type { ParameterRef, Tensor, TensorSlice, WeightProvider } from "@llm-explorer/model-ir";
import { dtypeSize, numElements } from "@llm-explorer/model-ir";
import { parseSafetensorsHeader, readTensor, type SafetensorsFile } from "./safetensors.js";

/**
 * WeightProvider backed by an in-memory safetensors buffer.
 *
 * The whole file is downloaded once (fine for tiny models — a few hundred KB
 * to a few MB), but tensor *decoding* stays lazy: listParameters()/
 * getParameterInfo() only ever read the safetensors header, and loadTensor()
 * decodes just the requested slice. For multi-GB checkpoints the same
 * interface would be backed by a server that does true byte-range HTTP reads
 * instead of an in-memory buffer — nothing above this provider would change.
 */
export class SafetensorsWeightProvider implements WeightProvider {
  id: string;
  private file: SafetensorsFile;

  constructor(id: string, buffer: ArrayBuffer) {
    this.id = id;
    this.file = parseSafetensorsHeader(buffer);
  }

  async listParameters(): Promise<ParameterRef[]> {
    return Object.entries(this.file.header).map(([name, entry]) => this.toParameterRef(name, entry));
  }

  async getParameterInfo(parameterId: string): Promise<ParameterRef> {
    const entry = this.file.header[parameterId];
    if (!entry) throw new Error(`Unknown parameter: ${parameterId}`);
    return this.toParameterRef(parameterId, entry);
  }

  async loadTensor(parameterId: string, options?: TensorSlice): Promise<Tensor> {
    return readTensor(this.file, parameterId, options);
  }

  private toParameterRef(name: string, entry: { shape: number[]; dtype: string }): ParameterRef {
    const n = numElements(entry.shape);
    return {
      name,
      shape: entry.shape,
      dtype: entry.dtype,
      numElements: n,
      bytes: n * dtypeSize(entry.dtype),
      providerId: this.id,
      logicalShape: entry.shape,
    };
  }
}
