/**
 * frontend/src/components/Controls.tsx
 */
import React from "react";

interface ControlsProps {
  onRun: () => void;
  onReset: () => void;
  modelResult: any; // shape depends on your fake backend
}

export const Controls: React.FC<ControlsProps> = ({ onRun, onReset, modelResult }) => {
  return (
    <div style={{ margin: "1rem" }}>
      <button onClick={onRun} style={{ marginRight: "1rem" }}>
        Run
      </button>
      <button onClick={onReset}>Reset</button>

      {modelResult && (
        <div style={{ marginTop: "1rem", padding: "1rem", border: "1px dashed #aaa" }}>
          <h3>Result from Model</h3>
          <pre>{JSON.stringify(modelResult, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};
