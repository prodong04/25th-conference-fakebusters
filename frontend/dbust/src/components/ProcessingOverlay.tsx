import React from 'react';
import styled, { keyframes } from 'styled-components';

const fadeIn = keyframes`
  from { opacity: 0; }
  to { opacity: 1; }
`;

const Overlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.8);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  animation: ${fadeIn} 0.3s ease-in-out;
`;

const Text = styled.h2`
  color: white;
  font-size: 2rem;
  margin-bottom: 1rem;
`;

const dotAnimation = keyframes`
  0% { opacity: 0.2; }
  20% { opacity: 1; }
  100% { opacity: 0.2; }
`;

const DotsContainer = styled.div`
  display: flex;
`;

const Dot = styled.span`
  width: 10px;
  height: 10px;
  background-color: white;
  border-radius: 50%;
  margin: 0 5px;
  animation: ${dotAnimation} 1.4s infinite ease-in-out;

  &:nth-child(2) {
    animation-delay: 0.2s;
  }

  &:nth-child(3) {
    animation-delay: 0.4s;
  }
`;

const ProcessingOverlay: React.FC = () => (
  <Overlay>
    <Text>PROCESSING</Text>
    <DotsContainer>
      <Dot />
      <Dot />
      <Dot />
    </DotsContainer>
  </Overlay>
);

export default ProcessingOverlay;