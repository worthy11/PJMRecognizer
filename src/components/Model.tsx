import { Tensor, InferenceSession } from "onnxjs";

const extractor = new InferenceSession();
const lstm = new InferenceSession();
const source = "./assets/pjmrecognizer.onnx";

await extractor.loadModel(source);
await lstm.loadModel(source);

const inputs = [
  new Tensor(new Float32Array([1.0, 2.0, 3.0, 4.0]), "float32", [2, 2]),
];

const extract_landmarks = async (frame: ImageBitmap) => {
  const input = new Tensor(new Float32Array(frame), "float32", [
    frame.height,
    frame.width,
  ]);
  const landmarks = await extractor.run(frame);

  // Process landmarks

  return new Tensor(new Float32Array(landmarks), "float32", [
    landmarks.size,
    2,
  ]);
};

export const process_frame = async (frame: ImageBitmap) => {
  const landmarks = extract_landmarks(frame);
  const prediction = await lstm.run(landmarks);
};
