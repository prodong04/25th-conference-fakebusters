'use client'

import React, { useState, useEffect } from 'react';
import FileUpload from '../components/FileUpload';
import ProcessingPage from './ProcessingPage';
import AugmentedVideo from './AugmentedVideo';


function simulateLongProcess(file: File): Promise<boolean> {
  return new Promise((resolve) => {
    console.log(`Processing file: ${file.name}, Size: ${file.size} bytes`);
    const processingTime = 0;

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
  const [augmentedImages, setAugmentedImages] = useState<string[]>([]);


  const handleFileUpload = async (file: File) => {
    setIsProcessing(true);
    setResult(null);
    setOriginalVideoSrc(URL.createObjectURL(file));

    try {
      const processedResult = await simulateLongProcess(file);
      setResult(processedResult);
      // Load augmented images from the public directory
      const images: string[] = loadAugmentedImages();
      setAugmentedImages(images);
    } catch (error) {
      console.error("Error processing file:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  const loadAugmentedImages = (): string[] => {
    const imageCount = 1200; // Total number of images
    const images: string[] = [];
    for (let i = 1; i <= imageCount; i++) {
      const paddedNumber = String(i).padStart(4, '0');
      images.push(`/data/chim/frames/frame${paddedNumber}.jpg`);
    }
    return images;
  };


  useEffect(() => {
    if (originalVideoSrc && result !== null) {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }
  }, [originalVideoSrc, result]);

  const roiVideos = {
    leftEye: '/data/002/002_left_eye_roi.mp4',
    mouth: '/data/002/002_mouth_roi.mp4',
    nose: '/data/002/002_nose_roi.mp4',
  };

  return (
    <main className="flex flex-col items-center justify-center p-6">
      <h1 className="text-3xl font-bold mt-24 mb-4">Deepfake Detector</h1>
      <FileUpload onFileUpload={handleFileUpload} />
      <p className="mt-2 mb-64 text-sm text-gray-600">Upload an image or video to check for deepfakes.</p>
      {originalVideoSrc && result !== null ? (
        <>
          <ProcessingPage originalVideoSrc={originalVideoSrc} roiVideos={roiVideos} />
          <AugmentedVideo images={augmentedImages} />
        </>
      ) : null}
    </main>
  );
};

export default MainPage;