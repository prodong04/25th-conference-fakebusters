'use client'

import React, { useState } from 'react';
import FileUpload from '../components/FileUpload';
import Result from '../components/Result';
import ProcessingOverlay from '../components/ProcessingOverlay';

function simulateLongProcess(file: File): Promise<boolean> {
  return new Promise((resolve) => {
    console.log(`Processing file: ${file.name}, Size: ${file.size} bytes`);
    const processingTime = 5000;

    setTimeout(() => {
      console.log(`Finished processing: ${file.name}`);
      const result = file.size < 1000000; // True if file is smaller than 1MB
      resolve(result);
    }, processingTime);
  });
}

export default function Home() {
  const [result, setResult] = useState<boolean | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileUpload = async (file: File) => {
    setIsProcessing(true);
    setResult(null);

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
      {result !== null && <Result isTrue={result} />}
      {isProcessing && <ProcessingOverlay />}
      <p className="mt-4 text-sm text-gray-600">Upload an image or video to check for deepfakes.</p>
    </main>
  );
}