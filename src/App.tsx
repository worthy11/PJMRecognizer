import React, { useState, useCallback, useRef } from "react";
import WebcamComponent from "./components/Webcam";
import HandLandmarkExtractor from "./components/HandLandmarkExtractor";
import SignLanguageClassifier from "./components/SignLanguageClassifier";
import WordGenerator from "./components/WordGenerator";
import "./App.css";

function App() {
  const [landmarks, setLandmarks] = useState<number[]>([]);
  const [prediction, setPrediction] = useState<string>("");
  const [targetLetter, setTargetLetter] = useState<string>("");
  const [isCorrect, setIsCorrect] = useState<boolean>(false);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const handleFrame = useCallback((frame: ImageData) => {
    // This will be handled by the HandLandmarkExtractor
  }, []);

  const handleVideoRef = useCallback((video: HTMLVideoElement | null) => {
    videoRef.current = video;
  }, []);

  const handleLandmarks = useCallback((newLandmarks: number[]) => {
    setLandmarks(newLandmarks);
  }, []);

  const handlePrediction = (pred: string) => {
    setPrediction(pred);
    // Check if prediction is correct
    const correct =
      pred === targetLetter ||
      (targetLetter === "J" && pred === "I") ||
      (targetLetter === "Z" && pred === "D");
    setIsCorrect(correct);
  };

  const handleLetterComplete = useCallback((letter: string) => {
    setTargetLetter(letter);
    setIsCorrect(false);
  }, []);

  const toggleLandmarks = () => {
    setShowLandmarks(prev => !prev);
  };

  return (
    <div className="app">
      <h1>Polish Sign Language Recognizer</h1>

      <div className="main-content">
        <div className="webcam-section">
          <div className="webcam-controls">
            <button 
              onClick={toggleLandmarks}
              className={`landmark-toggle ${showLandmarks ? 'active' : ''}`}
            >
              {showLandmarks ? 'Hide Landmarks' : 'Show Landmarks'}
            </button>
          </div>
          <WebcamComponent onFrame={handleFrame} onVideoRef={handleVideoRef} />
          <HandLandmarkExtractor
            onLandmarks={handleLandmarks}
            videoRef={videoRef}
            showLandmarks={showLandmarks}
          />
          <SignLanguageClassifier
            onPrediction={handlePrediction}
            landmarks={landmarks}
            targetLetter={targetLetter}
          />
        </div>

        <div className="word-section">
          <div className={`prediction ${isCorrect ? "correct" : ""}`}>
            {prediction}
          </div>
          <WordGenerator onLetterComplete={handleLetterComplete} />
        </div>
      </div>
    </div>
  );
}

export default App;
