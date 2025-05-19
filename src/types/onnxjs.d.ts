declare module 'onnxjs' {
  export interface InferenceSession {
    loadModel(modelPath: string): Promise<void>;
    run(inputs: Tensor[]): Promise<Map<string, Tensor>>;
  }

  export interface Tensor {
    data: Float32Array;
    dims: number[];
    type: string;
  }

  export interface InferenceSessionOptions {
    backendHint?: string;
  }

  export class InferenceSession {
    constructor(options?: InferenceSessionOptions);
  }

  export class Tensor {
    constructor(
      data: Float32Array,
      type: string,
      dims: number[]
    );
  }

  export const env: {
    wasm: {
      wasmPaths: string;
      simd: boolean;
      proxy: boolean;
    };
  };
} 