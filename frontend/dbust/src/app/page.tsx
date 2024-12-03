'use client'

import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import ProcessingPage from './ProcessingPage';

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

  const roiVideos = {
    leftEye: '/002/002_left_eye_roi.mp4',
    mouth: '/002/002_mouth_roi.mp4',
    nose: '/002/002_nose_roi.mp4',
  };

  return (
    <main className="flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mt-24 mb-4">Deepfake Detector</h1>
      <FileUpload onFileUpload={handleFileUpload} />
      <p className="mt-2 mb-64 text-sm text-gray-600">Upload an image or video to check for deepfakes.</p>
      {originalVideoSrc && result !== null ? (
        <ProcessingPage originalVideoSrc={originalVideoSrc} roiVideos={roiVideos} />
      ) : null}
    </main>
  );
};

export default MainPage;