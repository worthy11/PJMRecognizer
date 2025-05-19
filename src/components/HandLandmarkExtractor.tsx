import React, { useEffect, useRef } from "react";
import { FilesetResolver, GestureRecognizer } from "@mediapipe/tasks-vision";

interface HandLandmarkExtractorProps {
  onLandmarks: (landmarks: number[]) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

const HandLandmarkExtractor: React.FC<HandLandmarkExtractorProps> = ({
  onLandmarks,
  videoRef,
}) => {
  const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);

  useEffect(() => {
    const initializeGestureRecognizer = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
      );

      const gestureRecognizer = await GestureRecognizer.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        }
      );

      gestureRecognizerRef.current = gestureRecognizer;
    };

    initializeGestureRecognizer();

    return () => {
      if (gestureRecognizerRef.current) {
        gestureRecognizerRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    let animationFrameId: number;

    const processFrame = async () => {
      if (gestureRecognizerRef.current && videoRef.current) {
        const video = videoRef.current;

        // Check if video is ready and has valid dimensions
        if (
          video.readyState !== 4 ||
          video.videoWidth === 0 ||
          video.videoHeight === 0
        ) {
          animationFrameId = requestAnimationFrame(processFrame);
          return;
        }

        const nowInMs = performance.now();

        try {
          const results = gestureRecognizerRef.current.recognizeForVideo(
            video,
            nowInMs
          );

          if (results.landmarks && results.landmarks.length > 0) {
            // Take the first detected hand
            const landmarks = results.landmarks[0];
            // Extract only x and y coordinates
            const flattenedLandmarks = landmarks.flatMap((landmark) => [
              landmark.x,
              landmark.y,
            ]);
            onLandmarks(flattenedLandmarks);
          }
        } catch (error) {
          console.error("Error processing frame:", error);
        }
      }
      animationFrameId = requestAnimationFrame(processFrame);
    };

    processFrame();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [videoRef, onLandmarks]);

  return null;
};

export default HandLandmarkExtractor;
