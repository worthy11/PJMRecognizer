import React, { useState, useCallback, useEffect } from "react";

// Sample Polish words (you can expand this list)
const POLISH_WORDS = [
  "kot",
  "pies",
  "dom",
  "las",
  "woda",
  "niebo",
  "słońce",
  "księżyc",
  "kwiat",
  "drzewo",
  "ptak",
  "ryba",
  "serce",
  "miłość",
  "radość",
  "smutek",
];

interface WordGeneratorProps {
  onLetterComplete: (letter: string) => void;
}

const WordGenerator: React.FC<WordGeneratorProps> = ({ onLetterComplete }) => {
  const [currentWord, setCurrentWord] = useState<string>("");

  const generateWord = useCallback(() => {
    // Updated to match the corrected letter order in the classifier
    const letters = "ABCDEFGHIKLMNOPRSUWY";
    const randomLetter = letters[Math.floor(Math.random() * letters.length)];
    setCurrentWord(randomLetter);
    onLetterComplete(randomLetter);
  }, [onLetterComplete]);

  // Generate initial letter on mount
  useEffect(() => {
    generateWord();
  }, [generateWord]);

  return (
    <div className="word-generator">
      <h2>Current Letter: {currentWord}</h2>
      <button onClick={generateWord} className="new-word-button">
        Generate New Letter
      </button>
    </div>
  );
};

export default WordGenerator;
