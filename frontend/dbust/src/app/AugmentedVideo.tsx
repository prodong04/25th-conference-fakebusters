import React, { useState, useEffect } from 'react';

interface AugmentedVideoProps {
  images: string[];
}

const AugmentedVideo: React.FC<AugmentedVideoProps> = ({ images }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % images.length);
    }, 1000 / 60); // Change image every 16.67 milliseconds (60 images per second)

    return () => clearInterval(interval);
  }, [images.length]);

  return (
    <div className="bg-gray-100">
      <div className="mx-auto max-w-2xl py-8 sm:py-12 lg:py-4">
        <div className="mx-auto py-8 sm:py-12 lg:py-4">
          <h1 className="text-3xl font-bold mb-4">Augmented Video</h1>

          <div className="mt-6 space-y-12 lg:grid lg:grid-cols-1 lg:gap-x-6 lg:space-y-0">
            <div className="group relative">
              <img
                className="w-full rounded-lg bg-white object-cover group-hover:opacity-75"
                src={images[currentIndex]}
                alt={`Augmented frame ${currentIndex + 1}`}
                style={{ aspectRatio: '16/9' }} // Adjust aspect ratio to match original dimensions
              />
              <h3 className="mt-6 text-sm text-gray-500">Frame {currentIndex + 1}</h3>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AugmentedVideo;