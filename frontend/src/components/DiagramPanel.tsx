/**
 * frontend/src/components/DiagramPanel.tsx
 */
import React, { useState } from "react";
import { LatentToggleMap } from "../App";
import { LatentInfoModal } from "./LatentInfoModal";

interface CircuitData {
  id: string;
  name: string;
  layers: {
    layerId: number;
    saeName: string;
    latents: {
      latentId: number;
      conceptName: string;
      activationSample?: number;
    }[];
  }[];
}

interface DiagramPanelProps {
  circuit: CircuitData;
  toggledLatents: LatentToggleMap;
  onToggleLatent: (layerId: number, latentId: number, isOff: boolean) => void;
}

export const DiagramPanel: React.FC<DiagramPanelProps> = ({
  circuit,
  toggledLatents,
  onToggleLatent,
}) => {
  const [selectedLatent, setSelectedLatent] = useState<{
    layerId: number;
    latentId: number;
    conceptName?: string;
  } | null>(null);

  const handleLatentClick = (layerId: number, latentId: number, conceptName?: string) => {
    setSelectedLatent({ layerId, latentId, conceptName });
  };

  const handleToggleChange = (
    layerId: number,
    latentId: number,
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const isOff = e.target.checked; // If checked => we interpret as "OFF" for clarity
    onToggleLatent(layerId, latentId, isOff);
  };

  return (
    <div style={{ display: "flex", gap: "1rem", margin: "1rem" }}>
      {circuit.layers.map((layer) => {
        const offLatents = toggledLatents[layer.layerId] || [];
        return (
          <div
            key={layer.layerId}
            style={{
              border: "1px solid #ccc",
              padding: "0.5rem",
              flex: "0 0 auto",
              width: "200px",
            }}
          >
            <h3>Layer {layer.layerId}</h3>
            {layer.latents.map((latent) => {
              const isOff = offLatents.includes(latent.latentId);
              return (
                <div key={latent.latentId}
                     style={{ 
                       margin: "0.25rem", 
                       border: "1px solid #aaa", 
                       backgroundColor: isOff ? "#fdd" : "#dfd" 
                     }}
                >
                  <label>
                    <input
                      type="checkbox"
                      checked={isOff}
                      onChange={(e) => handleToggleChange(layer.layerId, latent.latentId, e)}
                    />
                    <span 
                      style={{ cursor: "pointer", marginLeft: "6px" }}
                      onClick={() => handleLatentClick(layer.layerId, latent.latentId, latent.conceptName)}
                    >
                      {latent.conceptName} (#{latent.latentId})
                    </span>
                  </label>
                </div>
              );
            })}
          </div>
        );
      })}
      {selectedLatent && (
        <LatentInfoModal 
          latentInfo={selectedLatent} 
          onClose={() => setSelectedLatent(null)} 
        />
      )}
    </div>
  );
};
