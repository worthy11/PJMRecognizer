import React, { useCallback, useRef, useEffect } from "react";
import Webcam from "react-webcam";

interface WebcamComponentProps {
  onFrame: (frame: ImageData) => void;
  onVideoRef: (video: HTMLVideoElement | null) => void;
}

const WebcamComponent: React.FC<WebcamComponentProps> = ({
  onFrame,
  onVideoRef,
}) => {
  const webcamRef = useRef<Webcam>(null);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  useEffect(() => {
    if (webcamRef.current?.video) {
      onVideoRef(webcamRef.current.video);
    }
    return () => onVideoRef(null);
  }, [onVideoRef]);

  const handleUserMedia = useCallback(() => {
    console.log("Webcam stream started");
  }, []);

  return (
    <div className="webcam-container">
      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        onUserMedia={handleUserMedia}
        mirrored={true}
      />
    </div>
  );
};

export default WebcamComponent;
