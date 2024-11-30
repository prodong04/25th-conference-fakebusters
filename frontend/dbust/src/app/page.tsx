'use client'

import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';

function simulateLongProcess(file: File): Promise<boolean> {
  return new Promise((resolve) => {
    console.log(`Processing file: ${file.name}, Size: ${file.size} bytes`);
    const processingTime = 5000;

    setTimeout(() => {
      console.log(`Finished processing: ${file.name}`);
      resolve(true);
    }, processingTime);
  });
}

const MainPage: React.FC = () => {
  const [result, setResult] = useState<boolean | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [originalVideoSrc, setOriginalVideoSrc] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setIsProcessing(true);
    setResult(null);
    setOriginalVideoSrc(URL.createObjectURL(file));

    try {
      const processedResult = await simulateLongProcess(file);
      setResult(processedResult);
    } catch (error) {
      console.error("Error processing file:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mb-4">Deepfake Detector</h1>
      <FileUpload onFileUpload={handleFileUpload} />
      <p className="mt-4 text-sm text-gray-600">Upload an image or video to check for deepfakes.</p>
      {originalVideoSrc && (
        <div className="mt-6">
          <h2 className="text-2xl font-bold text-gray-900">Processing Results</h2>
          <div className="mb-6">
            <video
              controls
              autoPlay
              loop
              muted
              className="w-full max-w-2xl"
              src={originalVideoSrc}
            />
          </div>
        </div>
      )}
    </main>
  );
};

export default MainPage;