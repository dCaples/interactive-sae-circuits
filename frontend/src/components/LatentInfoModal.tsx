/**
 * frontend/src/components/LatentInfoModal.tsx
 */
import React from "react";

interface LatentInfoProps {
  latentInfo: {
    layerId: number;
    latentId: number;
    conceptName?: string;
  };
  onClose: () => void;
}

export const LatentInfoModal: React.FC<LatentInfoProps> = ({ latentInfo, onClose }) => {
  const { layerId, latentId, conceptName } = latentInfo;

  return (
    <div
      style={{
        position: "fixed",
        top: "20%",
        left: "30%",
        width: "40%",
        backgroundColor: "#fff",
        border: "2px solid #000",
        padding: "1rem",
        zIndex: 999,
      }}
    >
      <h2>Latent Details</h2>
      <p>Layer ID: {layerId}</p>
      <p>Latent ID: {latentId}</p>
      <p>Concept Name: {conceptName}</p>
      <p>
        Here you could show additional details:
        <ul>
          <li>Activation across tokens</li>
          <li>Task importance scores</li>
          <li>Interpretation notes, etc.</li>
        </ul>
      </p>

      <button onClick={onClose}>Close</button>
    </div>
  );
};
