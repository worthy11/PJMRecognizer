import React, { useEffect, useRef } from "react";
import * as onnx from "onnxjs";

interface SignLanguageClassifierProps {
  onPrediction: (prediction: string) => void;
  landmarks?: number[];
  targetLetter?: string;
}

const SignLanguageClassifier: React.FC<SignLanguageClassifierProps> = ({
  onPrediction,
  landmarks,
  targetLetter,
}) => {
  const sessionRef = useRef<onnx.InferenceSession | null>(null);

  // Polish sign language letters (excluding J and Z as they map to I and D)
  const POLISH_LETTERS = "ABCDEFGHIKLMNOPRSTUWY";

  // Check if prediction is correct, including special cases
  const isCorrectPrediction = (
    prediction: string,
    target?: string
  ): boolean => {
    if (!target) return false;
    if (prediction === target) return true;
    if (target === "J" && prediction === "I") return true;
    if (target === "Z" && prediction === "D") return true;
    return false;
  };

  // Calculate Euclidean distance between two points
  const calculateDistance = (
    x1: number,
    y1: number,
    x2: number,
    y2: number
  ): number => {
    return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
  };

  // Normalize landmarks to [0,1] range
  const normalizeLandmarks = (landmarks: number[]): number[] => {
    const numLandmarks = landmarks.length / 2;
    const xCoords: number[] = [];
    const yCoords: number[] = [];

    // Separate x and y coordinates
    for (let i = 0; i < numLandmarks; i++) {
      xCoords.push(landmarks[i * 2]);
      yCoords.push(landmarks[i * 2 + 1]);
    }

    // Find min and max for both dimensions
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);

    // Normalize coordinates
    const normalizedLandmarks: number[] = [];
    for (let i = 0; i < numLandmarks; i++) {
      const x = (landmarks[i * 2] - minX) / (maxX - minX);
      const y = (landmarks[i * 2 + 1] - minY) / (maxY - minY);
      normalizedLandmarks.push(x, y);
    }

    return normalizedLandmarks;
  };

  // Convert landmarks to distance matrix
  const landmarksToDistances = (landmarks: number[]): number[] => {
    // First normalize the landmarks
    const normalizedLandmarks = normalizeLandmarks(landmarks);
    const distances: number[] = [];
    const numLandmarks = normalizedLandmarks.length / 2;

    // Calculate distances between all pairs of landmarks
    for (let i = 0; i < numLandmarks; i++) {
      for (let j = 0; j < numLandmarks; j++) {
        const x1 = normalizedLandmarks[i * 2];
        const y1 = normalizedLandmarks[i * 2 + 1];
        const x2 = normalizedLandmarks[j * 2];
        const y2 = normalizedLandmarks[j * 2 + 1];
        distances.push(calculateDistance(x1, y1, x2, y2));
      }
    }

    return distances;
  };

  useEffect(() => {
    const loadModel = async () => {
      try {
        // Initialize ONNX.js session
        const session = new onnx.InferenceSession({
          backendHint: "webgl",
        });

        // Load the model
        await session.loadModel("/model.onnx");
        sessionRef.current = session;
        console.log("Model loaded successfully");
      } catch (error) {
        console.error("Error loading model:", error);
      }
    };

    loadModel();

    return () => {
      sessionRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (landmarks && landmarks.length > 0) {
      classifyLandmarks(landmarks);
    }
  }, [landmarks]);

  const classifyLandmarks = async (landmarks: number[]) => {
    if (!sessionRef.current) {
      console.log("Model not loaded yet");
      return;
    }

    try {
      // Convert landmarks to distance matrix
      const distances = landmarksToDistances(landmarks);

      // Create tensor from distances
      const tensor = new onnx.Tensor(
        new Float32Array(distances),
        "float32",
        [1, 441, 1] // Shape: [batch_size, num_distances, 1]
      );

      // Run inference
      const results = await sessionRef.current.run([tensor]);
      const output = results.values().next().value.data;

      // Get the index of the highest probability
      const maxIndex = output.indexOf(Math.max(...output));

      // Convert index to Polish letter
      const prediction = POLISH_LETTERS[maxIndex];

      // Check if prediction is correct
      const isCorrect = isCorrectPrediction(prediction, targetLetter);

      onPrediction(prediction);
    } catch (error) {
      console.error("Error during inference:", error);
      console.error("Landmarks shape:", landmarks.length);
      if (error instanceof Error) {
        console.error("Error details:", error.message);
        console.error("Error stack:", error.stack);
      }
    }
  };

  return null;
};

export default SignLanguageClassifier;
