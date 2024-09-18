import { Tensor, InferenceSession } from "onnxjs";
import {
  PoseLandmarker,
  HandLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

const vision = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
);
const handLandmarker = await HandLandmarker.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
    delegate: "GPU",
  },
  numHands: 2,
});
const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
  baseOptions: {
    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
    delegate: "GPU",
  },
  numPoses: 2,
});

const lstm = new InferenceSession();
const modelUrl = "./assets/pjmrecognizer.onnx";
await lstm.loadModel(modelUrl);

function extractLandmarks(frame: ImageData) {
  const hands = handLandmarker.detect(frame);
  const body = poseLandmarker.detect(frame);

  const numHands = hands.landmarks.length;
  let landmarks = Array(266).fill(0);

  switch (numHands) {
    case 0:
      break;

    case 1:
      if (hands.handedness[0][0].index) {
        landmarks.fill(hands.landmarks[0], 112 * 2, 133 * 2);
      } else {
        landmarks.fill(hands.landmarks[0], 91 * 2, 112 * 2);
      }
      break;

    case 2:
      if (hands.handedness[0][0].index) {
        landmarks.fill(hands.landmarks[0], 112 * 2, 133 * 2);
        landmarks.fill(hands.landmarks[1], 91 * 2, 112 * 2);
      } else {
        landmarks.fill(hands.landmarks[0], 91 * 2, 112 * 2);
        landmarks.fill(hands.landmarks[1], 112 * 2, 133 * 2);
      }
  }
  landmarks.fill(body.landmarks[0][0], 0, 2);
  landmarks.fill(body.landmarks[0][11], 5 * 2, 6 * 2);
  landmarks.fill(body.landmarks[0][12], 6 * 2, 7 * 2);

  return landmarks;
}

export const getPrediction = async (imageData: ImageData) => {
  const landmarks = [
    new Tensor(
      new Float32Array(extractLandmarks(imageData)),
      "float32",
      [133, 2]
    ),
  ];
  const prediction = await lstm.run(landmarks);
  return prediction;
};
