// /**
//  * frontend/src/App.tsx
//  */

// import React, { useEffect, useState } from "react";
// import "./App.css";
// import { TaskSelector } from "./components/TaskSelector";
// import { DiagramPanel } from "./components/DiagramPanel";
// import { Controls } from "./components/Controls";

// // Sample circuit data
// import exampleCircuit from "./circuits/exampleCircuitData.json";

// export interface LatentToggleMap {
//   [layerId: string]: number[]; // array of latentIds that are "off"
// }

// function App() {
//   const [selectedTask, setSelectedTask] = useState("dictionaryKeyTask");
//   // Because we only have one circuit in the example, we’ll just store it:
//   const [circuit, setCircuit] = useState(exampleCircuit);
//   // Keep track of toggled latents:
//   const [toggledLatents, setToggledLatents] = useState<LatentToggleMap>({});

//   // For the final model output from the backend:
//   const [modelResult, setModelResult] = useState<any>(null);

//   // Resets toggles
//   const handleReset = () => {
//     setToggledLatents({});
//     setModelResult(null);
//   };

//   // Called on "Run" button
//   const handleRun = async () => {
//     const payload = {
//       promptId: selectedTask,
//       toggles: toggledLatents,
//     };
//     try {
//       const res = await fetch("http://localhost:4000/runWithLatentMask", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify(payload),
//       });
//       const data = await res.json();
//       setModelResult(data.result);
//     } catch (err) {
//       console.error("Error running with mask:", err);
//     }
//   };

//   // Called when user toggles a single latent on/off:
//   const onToggleLatent = (layerId: number, latentId: number, isOff: boolean) => {
//     // We store a list of latents that are "off." 
//     // If isOff=true, we add that latent to toggledLatents for that layer.
//     // If isOff=false, we remove it.
//     setToggledLatents((prev) => {
//       const layerKey = String(layerId);
//       const existing = prev[layerKey] || [];
//       let updated: number[];
//       if (isOff) {
//         // Add latent ID if not present
//         if (!existing.includes(latentId)) {
//           updated = [...existing, latentId];
//         } else {
//           updated = existing;
//         }
//       } else {
//         // Remove latent ID
//         updated = existing.filter((id) => id !== latentId);
//       }
//       return { ...prev, [layerKey]: updated };
//     });
//   };

//   return (
//     <div className="App">
//       <header>
//         <h1>Scalable Sparse Circuit UI</h1>
//         <TaskSelector 
//           selectedTask={selectedTask} 
//           setSelectedTask={setSelectedTask} 
//         />
//       </header>

//       <h1 className="bg-blue">
//         Hello, world
//       </h1>

//       <DiagramPanel 
//         circuit={circuit} 
//         toggledLatents={toggledLatents} 
//         onToggleLatent={onToggleLatent}
//       />

//       <Controls 
//         onRun={handleRun} 
//         onReset={handleReset} 
//         modelResult={modelResult}
//       />
//     </div>
//   );
// }

// export default App;

// frontend/src/App.tsx

import React from "react";
import CircuitVisualization from "./CircuitVisualization"; // or wherever you placed it

function App() {
  return (
    <div className="App">
      <CircuitVisualization />
    </div>
  );
}

export default App;
