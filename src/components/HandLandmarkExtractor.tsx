import React, { useEffect, useRef, useState } from "react";
import { FilesetResolver, GestureRecognizer } from "@mediapipe/tasks-vision";

interface HandLandmarkExtractorProps {
  onLandmarks: (landmarks: number[]) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  showLandmarks?: boolean;
}

// Define the hand connections for drawing lines between landmarks
const HAND_CONNECTIONS = [
  // Thumb connections
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger connections
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger connections
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger connections
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky finger connections
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm connections
  [5, 9], [9, 13], [13, 17]
];

const HandLandmarkExtractor: React.FC<HandLandmarkExtractorProps> = ({
  onLandmarks,
  videoRef,
  showLandmarks = false,
}) => {
  const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const [landmarks, setLandmarks] = useState<any[]>([]);

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
            const handLandmarks = results.landmarks[0];
            // Set landmarks for visualization
            setLandmarks(handLandmarks);
            // Extract only x and y coordinates
            const flattenedLandmarks = handLandmarks.flatMap((landmark) => [
              landmark.x,
              landmark.y,
            ]);
            onLandmarks(flattenedLandmarks);
          } else {
            setLandmarks([]);
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

  // Add useEffect for drawing landmarks on canvas
  useEffect(() => {
    if (!showLandmarks || !videoRef.current || !canvasRef.current || landmarks.length === 0) {
      // Clear canvas if landmarks should not be shown
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    // Set canvas dimensions to match video dimensions (not client dimensions)
    // This is important for correct scaling
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear previous drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw connections between landmarks
    ctx.strokeStyle = '#00FF00'; // Green color for lines
    ctx.lineWidth = 3;
    
    HAND_CONNECTIONS.forEach(([start, end]) => {
      if (landmarks[start] && landmarks[end]) {
        // Use the actual video dimensions for scaling
        const startX = landmarks[start].x * canvas.width;
        const startY = landmarks[start].y * canvas.height;
        const endX = landmarks[end].x * canvas.width;
        const endY = landmarks[end].y * canvas.height;
        
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      }
    });
    
    // Draw each landmark point
    ctx.fillStyle = '#FF0000'; // Red color for points
    
    landmarks.forEach((landmark) => {
      // Use the actual video dimensions for scaling
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      
      // Draw circle for each point
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();
    });
    
  }, [landmarks, showLandmarks, videoRef]);

  // Add a resize observer to handle window resizing
  useEffect(() => {
    if (!canvasContainerRef.current || !canvasRef.current || !videoRef.current) return;
    
    const resizeObserver = new ResizeObserver(() => {
      if (canvasRef.current && videoRef.current) {
        // Update canvas style dimensions to match the video element's display size
        const videoRect = videoRef.current.getBoundingClientRect();
        canvasContainerRef.current!.style.width = `${videoRect.width}px`;
        canvasContainerRef.current!.style.height = `${videoRect.height}px`;
        canvasContainerRef.current!.style.top = `${videoRect.top}px`;
        canvasContainerRef.current!.style.left = `${videoRect.left}px`;
      }
    });
    
    // Observe the video element
    if (videoRef.current) {
      resizeObserver.observe(videoRef.current);
    }
    
    // Initial positioning
    if (videoRef.current) {
      const videoRect = videoRef.current.getBoundingClientRect();
      canvasContainerRef.current.style.width = `${videoRect.width}px`;
      canvasContainerRef.current.style.height = `${videoRect.height}px`;
      canvasContainerRef.current.style.top = `${videoRect.top}px`;
      canvasContainerRef.current.style.left = `${videoRect.left}px`;
    }
    
    return () => {
      resizeObserver.disconnect();
    };
  }, [videoRef.current, canvasRef.current, showLandmarks]);
  
  return (
    <>
      {showLandmarks && (
        <div 
          ref={canvasContainerRef}
          style={{
            position: 'fixed',
            pointerEvents: 'none',
            zIndex: 10,
          }}
        >
          <canvas
            ref={canvasRef}
            style={{
              width: '100%',
              height: '100%',
              display: 'block',
              transform: 'scaleX(-1)', // Mirror the canvas to match mirrored webcam
            }}
          />
        </div>
      )}
    </>
  );
};

export default HandLandmarkExtractor;
