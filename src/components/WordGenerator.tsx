import React, { useState, useCallback, useEffect } from "react";

// Sample Polish words (you can expand this list)
const POLISH_WORDS = [
  "kot",
  "pies",
  "foka",
  "fala",
  "dom",
  "las",
  "woda",
  "niebo",
  "kwiat",
  "drzewo",
  "ptak",
  "ryba",
  "serce",
  "most",
  "Piotr",
  "Maks",
  "Lidia",
  "Adam"
];

// Translations for UI elements
const translations = {
  en: {
    currentWord: "Current Word",
    signTheLetter: "Sign the letter:",
    wordCompleted: "Word completed! 🎉",
    generateNewWord: "Generate New Word"
  },
  pl: {
    currentWord: "Aktualne słowo",
    signTheLetter: "Pokaż literę:",
    wordCompleted: "Słowo ukończone! 🎉",
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

interface WordGeneratorProps {
  onLetterComplete: (letter: string) => void;
  language: "en" | "pl";
}

const WordGenerator: React.FC<WordGeneratorProps> = ({ 
  onLetterComplete, 
  language 
}) => {
  const [currentWord, setCurrentWord] = useState<string>("");
  const [currentPosition, setCurrentPosition] = useState<number>(0);
  const [correctlyGuessedLetters, setCorrectlyGuessedLetters] = useState<boolean[]>([]);
  
  const t = translations[language];

  const generateWord = useCallback(() => {
    // Select a random word from the list
    const randomWord = POLISH_WORDS[Math.floor(Math.random() * POLISH_WORDS.length)];
    const upperCaseWord = randomWord.toUpperCase();
    
    setCurrentWord(upperCaseWord);
    setCurrentPosition(0);
    setCorrectlyGuessedLetters(new Array(upperCaseWord.length).fill(false));
    
    // Set the first letter as the target
    onLetterComplete(upperCaseWord[0]);
  }, [onLetterComplete]);

  // Handle prediction changes
  useEffect(() => {
    // Set up a listener for letter recognition
    const handlePredictionChange = (event: CustomEvent) => {
      const prediction = event.detail.prediction;
      
      if (currentWord && currentPosition < currentWord.length) {
        const targetLetter = currentWord[currentPosition];
        
        // Check if prediction is correct, including interchangeable letters
        let isCorrect = prediction === targetLetter;
        
        // Check interchangeable letters (works bidirectionally)
        if (!isCorrect) {
          // Check if the target letter has interchangeable alternatives
          if (interchangeableLetters[targetLetter]?.includes(prediction)) {
            isCorrect = true;
          }
          // Check if the prediction has interchangeable alternatives that match the target
          else if (interchangeableLetters[prediction]?.includes(targetLetter)) {
            isCorrect = true;
          }
        }
        
        if (isCorrect) {
          // Update correctly guessed letters
          const newCorrectlyGuessed = [...correctlyGuessedLetters];
          newCorrectlyGuessed[currentPosition] = true;
          setCorrectlyGuessedLetters(newCorrectlyGuessed);
          
          // Move to next letter
          const nextPosition = currentPosition + 1;
          setCurrentPosition(nextPosition);
          
          // If there are more letters, set the next target letter
          if (nextPosition < currentWord.length) {
            onLetterComplete(currentWord[nextPosition]);
          }
          // Word is completed - no alert popup
        }
      }
    };
    
    // Create and register the custom event listener
    window.addEventListener("predictionChange", handlePredictionChange as EventListener);
    
    return () => {
      window.removeEventListener("predictionChange", handlePredictionChange as EventListener);
    };
  }, [currentWord, currentPosition, correctlyGuessedLetters, onLetterComplete]);

  // Generate initial word on mount
  useEffect(() => {
    generateWord();
  }, [generateWord]);

  return (
    <div className="word-generator">
      <h2>{t.currentWord}</h2>
      <div className="word-display">
        {currentWord.split("").map((letter, index) => (
          <span 
            key={index} 
            className={`letter ${correctlyGuessedLetters[index] ? "correct" : ""} ${index === currentPosition ? "current" : ""}`}
          >
            {letter}
          </span>
        ))}
      </div>
      <div className="progress-info">
        {currentPosition < currentWord.length ? (
          <p>{t.signTheLetter} <strong>{currentWord[currentPosition]}</strong></p>
        ) : (
          <p>{t.wordCompleted}</p>
        )}
      </div>
      <button onClick={generateWord} className="new-word-button">
        {t.generateNewWord}
      </button>
    </div>
  );
};

export default WordGenerator;
