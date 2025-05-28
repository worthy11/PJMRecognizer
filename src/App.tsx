import React, { useState, useCallback, useRef } from "react";
import WebcamComponent from "./components/Webcam";
import HandLandmarkExtractor from "./components/HandLandmarkExtractor";
import SignLanguageClassifier from "./components/SignLanguageClassifier";
import WordGenerator from "./components/WordGenerator";
import "./App.css";

// Translations for UI elements
const translations = {
  en: {
    showLandmarks: "Show Landmarks",
    hideLandmarks: "Hide Landmarks",
    currentPrediction: "Current Prediction",
    alphabetGuide: "Polish Sign Language Alphabet",
    language: "PL",
    generateNewWord: "Generate New Word"
  },
  pl: {
    showLandmarks: "Pokaż punkty charakterystyczne",
    hideLandmarks: "Ukryj punkty charakterystyczne",
    currentPrediction: "Rozpoznana litera",
    alphabetGuide: "Polski Alfabet Migowy",
    language: "EN",
    generateNewWord: "Wygeneruj nowe słowo"
  }
};

// Define interchangeable letters in Polish sign language
const interchangeableLetters: Record<string, string[]> = {
  'I': ['J'],
  'J': ['I'],
  'D': ['Z'],
  'Z': ['D'],
  'F': ['T'],
  'T': ['F']
};

function App() {
  const [landmarks, setLandmarks] = useState<number[]>([]);
  const [prediction, setPrediction] = useState<string>("");
  const [targetLetter, setTargetLetter] = useState<string>("");
  const [isCorrect, setIsCorrect] = useState<boolean>(false);
  const [showLandmarks, setShowLandmarks] = useState<boolean>(false);
  const [language, setLanguage] = useState<"en" | "pl">("en");
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const t = translations[language];

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
    
    // Check if prediction is correct using interchangeable letters logic
    let correct = pred === targetLetter;
    
    // Check interchangeable letters (works bidirectionally)
    if (!correct) {
      // Check if the target letter has interchangeable alternatives
      if (interchangeableLetters[targetLetter]?.includes(pred)) {
        correct = true;
      }
      // Check if the prediction has interchangeable alternatives that match the target
      else if (interchangeableLetters[pred]?.includes(targetLetter)) {
        correct = true;
      }
    }
    
    setIsCorrect(correct);
  };

  const handleLetterComplete = useCallback((letter: string) => {
    setTargetLetter(letter);
    setIsCorrect(false);
  }, []);

  const toggleLandmarks = () => {
    setShowLandmarks(prev => !prev);
  };

  const toggleLanguage = () => {
    setLanguage(prev => prev === "en" ? "pl" : "en");
  };

  return (
    <div className="app">
      <div className="app-header">
        <h1>Polish Sign Language Alphabet Recognizer</h1>
        <button onClick={toggleLanguage} className="language-toggle">
          {t.language}
        </button>
      </div>

      <div className="main-content">
        <div className="webcam-section">
          <div className="webcam-controls">
            <button 
              onClick={toggleLandmarks}
              className={`landmark-toggle ${showLandmarks ? 'active' : ''}`}
            >
              {showLandmarks ? t.hideLandmarks : t.showLandmarks}
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
          <div className="current-prediction">
            <div className="prediction-label">{t.currentPrediction}</div>
            <div className={`prediction ${isCorrect ? "correct" : ""}`}>
              {prediction}
            </div>
          </div>
        </div>

        <div className="word-section">
          <WordGenerator 
            onLetterComplete={handleLetterComplete} 
            language={language}
          />
        </div>

        <div className="instruction-section">
          <div className="alphabet-guide">
            <h3>{t.alphabetGuide}</h3>
            <img 
              src="/polski-alfabet-palcowy.jpg" 
              alt="Polish Sign Language Alphabet Guide" 
              className="alphabet-image"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
