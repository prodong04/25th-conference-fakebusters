import React from 'react';

interface ResultProps {
  isTrue: boolean;
}

const Result: React.FC<ResultProps> = ({ isTrue }) => {
  return (
    <div className={`text-2xl font-bold p-4 rounded ${isTrue ? 'bg-green-500' : 'bg-red-500'} text-white`}>
      {isTrue ? 'True' : 'False'}
    </div>
  );
};

export default Result;