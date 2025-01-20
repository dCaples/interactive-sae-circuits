import React, { useState, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { ChevronLeft, ChevronRight, Play } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Checkbox } from '@/components/ui/checkbox';

// ----------------------------------------------------------------------
// Example prompts
// ----------------------------------------------------------------------
const EXAMPLES = [
  `predict: ages["Bob"]`,
  `predict: prices["Apples"]`,
  `predict: dictionary["Hello"]`,
];

// ----------------------------------------------------------------------
// We assume the same layers in the front end as in the backend
// ----------------------------------------------------------------------
const LAYERS = [7, 21, 40];

// ----------------------------------------------------------------------
// In the backend, these layers map to array indices [0,1,2] for
// latentActivations. We'll define an index map to help interpret them.
// ----------------------------------------------------------------------
const layerIndexMap: Record<number, number> = {};
LAYERS.forEach((layerId, idx) => {
  layerIndexMap[layerId] = idx;
});

// ----------------------------------------------------------------------
// Single "global" cluster definitions.
// ----------------------------------------------------------------------
const CLUSTERS: Record<
  number,
  Record<
    number,
    Record<string, number[]>
  >
> = {
  7: {
    1: {},
    2: { bracket_magic: [77, 78] },
    3: { bob_clusters: [300, 301] },
    4: { cluster_x: [50, 51, 52] },
  },
  21: {
    1: {},
    2: {},
    3: { bob_clusters: [345] },
    4: { cluster_x: [122, 343, 6225, 23] },
  },
  40: {
    1: {},
    2: {},
    3: {},
    4: {
      traceback: [200, 201, 202],
      output_1: [30, 101, 43],
    },
  },
};

// ----------------------------------------------------------------------
// Build an initial clusterStates object for a given token count
// ----------------------------------------------------------------------
function initClusterStatesForTokens(tokenCount: number) {
  const result: Record<
    number, // layerId
    Record<
      number, // tokenIdx
      {
        [clusterName: string]: {
          enabled: boolean;
          latents: Record<number, boolean>;
        }
      }
    >
  > = {};

  for (const layerId of LAYERS) {
    result[layerId] = {};
    for (let t = 0; t < tokenCount; t++) {
      // If no definition => ablated => no clusters
      const def = CLUSTERS[layerId]?.[t] || {};
      const clusterMap: any = {};
      for (const [clusterName, latentIds] of Object.entries(def)) {
        // By default, all latents are "on"
        const latObj: Record<number, boolean> = {};
        latentIds.forEach((lid) => {
          latObj[lid] = true;
        });
        clusterMap[clusterName] = {
          enabled: true,
          latents: latObj,
        };
      }
      result[layerId][t] = clusterMap;
    }
  }
  return result;
}

// ----------------------------------------------------------------------
// Check if token index is ablated across all layers => no clusters
// ----------------------------------------------------------------------
function isTokenAblatedAcrossAllLayers(
  clusterStates: ReturnType<typeof initClusterStatesForTokens>,
  tokenIdx: number
) {
  for (const layerId of LAYERS) {
    const clusterMap = clusterStates[layerId]?.[tokenIdx];
    const clusterNames = Object.keys(clusterMap || {});
    if (clusterNames.length > 0) {
      return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------
// Merge consecutive ablated tokens into a single "column"
// ----------------------------------------------------------------------
function buildColumns(
  clusterStates: ReturnType<typeof initClusterStatesForTokens>,
  tokens: string[]
) {
  const columns: {
    ablated: boolean;
    tokenIdxs: number[];
    label: string;
  }[] = [];

  let i = 0;
  while (i < tokens.length) {
    const ablated = isTokenAblatedAcrossAllLayers(clusterStates, i);
    if (!ablated) {
      columns.push({
        ablated: false,
        tokenIdxs: [i],
        label: tokens[i],
      });
      i++;
    } else {
      // gather run of consecutive ablated
      const start = i;
      let end = i;
      while (
        end < tokens.length &&
        isTokenAblatedAcrossAllLayers(clusterStates, end)
      ) {
        end++;
      }
      const chunk = tokens.slice(start, end).join(" + ");
      columns.push({
        ablated: true,
        tokenIdxs: Array.from({ length: end - start }, (_, k) => start + k),
        label: chunk,
      });
      i = end;
    }
  }

  return columns;
}

const CircuitVisualization = () => {
  // Which example index
  const [exampleIndex, setExampleIndex] = useState(0);
  const currentExample = EXAMPLES[exampleIndex];

  // We'll store the currently displayed tokens in a single array
  const [tokens, setTokens] = useState<string[]>([]);

  // Single clusterStates for the entire app (not per prompt)
  const [clusterStates, setClusterStates] = useState<ReturnType<
    typeof initClusterStatesForTokens
  > | null>(null);

  // "needRerun" indicates toggles changed or user switched examples
  // so the displayed results are out of date
  const [needRerun, setNeedRerun] = useState(false);

  // The results from the last run
  const [topProbs, setTopProbs] = useState<Record<string, number> | null>(null);
  const [latentActivations, setLatentActivations] = useState<any>(null);

  // We'll store computed activation fraction for each cluster
  const [activationValues, setActivationValues] = useState<any>({});

  // For cluster detail dialog
  const [selectedCell, setSelectedCell] = useState<null | {
    layerId: number;
    tokenIdx: number;
  }>(null);

  // Hover logic for arrows
  const [hoveredCell, setHoveredCell] = useState<null | {
    layerId: number;
    colIndex: number;
  }>(null);
  const [arrowLines, setArrowLines] = useState<any[]>([]);
  const tableRef = useRef<HTMLDivElement | null>(null);
  const cellRefs = useRef<Record<string, HTMLTableCellElement | null>>({});

  // Build columns if we have tokens + clusterStates
  const columns = clusterStates
    ? buildColumns(clusterStates, tokens)
    : [];

  // ----------------------------------------------------------------
  // Example navigation
  // ----------------------------------------------------------------
  function handlePrevExample() {
    if (exampleIndex > 0) {
      setExampleIndex(exampleIndex - 1);
      setNeedRerun(true);
      // Clear old display results
      setTopProbs(null);
      setLatentActivations(null);
      setActivationValues({});
    }
  }
  function handleNextExample() {
    if (exampleIndex < EXAMPLES.length - 1) {
      setExampleIndex(exampleIndex + 1);
      setNeedRerun(true);
      // Clear old display results
      setTopProbs(null);
      setLatentActivations(null);
      setActivationValues({});
    }
  }

  // ----------------------------------------------------------------
  // Toggling clusters
  // ----------------------------------------------------------------
  function toggleCluster(
    layerId: number,
    tokenIdx: number,
    clusterName: string,
    enabled: boolean
  ) {
    if (!clusterStates) return;
    setClusterStates((prev) => {
      if (!prev) return prev;
      const newLayerObj = { ...prev[layerId] };
      const newTokenObj = { ...newLayerObj[tokenIdx] };
      const oldCluster = newTokenObj[clusterName];
      newTokenObj[clusterName] = { ...oldCluster, enabled };
      newLayerObj[tokenIdx] = newTokenObj;

      return {
        ...prev,
        [layerId]: newLayerObj,
      };
    });
    setNeedRerun(true);
  }

  // ----------------------------------------------------------------
  // Toggling an individual latent
  // ----------------------------------------------------------------
  function toggleLatent(
    layerId: number,
    tokenIdx: number,
    clusterName: string,
    latentId: number,
    checked: boolean
  ) {
    if (!clusterStates) return;
    setClusterStates((prev) => {
      if (!prev) return prev;
      const newLayerObj = { ...prev[layerId] };
      const newTokenObj = { ...newLayerObj[tokenIdx] };
      const oldCluster = newTokenObj[clusterName];
      newTokenObj[clusterName] = {
        ...oldCluster,
        latents: {
          ...oldCluster.latents,
          [latentId]: checked,
        },
      };
      newLayerObj[tokenIdx] = newTokenObj;

      return {
        ...prev,
        [layerId]: newLayerObj,
      };
    });
    setNeedRerun(true);
  }

  // ----------------------------------------------------------------
  // Build toggles for the backend
  // toggles[layerId][tokenIndex] = array of latents we want "on"
  // ----------------------------------------------------------------
  function buildToggles() {
    const toggles: Record<number, Record<number, number[]>> = {};
    if (!clusterStates) return toggles;

    for (const layerId of LAYERS) {
      toggles[layerId] = {};
      for (let t = 0; t < tokens.length; t++) {
        const clusterMap = clusterStates[layerId][t];
        if (!clusterMap) {
          toggles[layerId][t] = [];
          continue;
        }
        const latentsOn: number[] = [];
        for (const info of Object.values(clusterMap)) {
          const cInfo = info as {
            enabled: boolean;
            latents: Record<number, boolean>;
          };
          if (cInfo.enabled) {
            for (const [latentIdStr, isOn] of Object.entries(cInfo.latents)) {
              if (isOn) latentsOn.push(parseInt(latentIdStr));
            }
          }
        }
        toggles[layerId][t] = latentsOn;
      }
    }
    return toggles;
  }

  // ----------------------------------------------------------------
  // Run circuit => call backend => get tokens & latentActivations
  // ----------------------------------------------------------------
  async function runCircuit() {
    const payload = {
      prompt: currentExample,
      toggles: buildToggles(),
    };

    try {
      const res = await fetch('http://localhost:4000/runWithLatentMask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      // data should have data.prompt, data.tokens, data.result.topProbs, data.result.latentActivations
      const { tokens: newTokens, result } = data;
      const { topProbs, latentActivations } = result || {};

      // If we have no clusterStates yet, initialize them
      // Otherwise, check that the new tokens match in length
      if (!clusterStates) {
        const newCS = initClusterStatesForTokens(newTokens.length);
        setClusterStates(newCS);
        setTokens(newTokens);
      } else {
        if (newTokens.length !== tokens.length) {
          // Show an error or do something if lengths differ
          alert(
            `Error: new tokens length (${newTokens.length}) differs from existing token length (${tokens.length}).`
          );
          return;
        }
        // If length is the same, we can keep our existing clusterStates
        setTokens(newTokens);
      }

      // 2) Save topProbs & latentActivations
      setTopProbs(topProbs || {});
      setLatentActivations(latentActivations || null);

      // 3) Recompute cluster activation fractions
      if (latentActivations) {
        computeClusterActivations(latentActivations, newTokens.length);
      } else {
        setActivationValues({});
      }

      setNeedRerun(false);
    } catch (err) {
      console.error('Backend error:', err);
    }
  }

  // ----------------------------------------------------------------
  // Compute cluster activation fraction
  // shape: latentActivations[tokenIdx][layerIndex][latentId]
  // ----------------------------------------------------------------
  function computeClusterActivations(activations: any, tokenCount: number) {
    if (!clusterStates) return;
    const newVals: any = {};

    for (let t = 0; t < tokenCount; t++) {
      for (const layerId of LAYERS) {
        const saeIndex = layerIndexMap[layerId];
        const latArray = activations[t][saeIndex];
        // If latArray is missing, skip
        if (!latArray) continue;

        if (!newVals[layerId]) newVals[layerId] = {};
        if (!newVals[layerId][t]) newVals[layerId][t] = {};

        // For each cluster in clusterStates
        const clusterMap = clusterStates[layerId][t];
        for (const [clusterName, clusterObj] of Object.entries(clusterMap)) {
          const cObj = clusterObj as {
            enabled: boolean;
            latents: Record<number, boolean>;
          };
          const enabledLatents = Object.entries(cObj.latents)
            .filter(([_, on]) => on)
            .map(([lid]) => parseInt(lid));

          if (enabledLatents.length === 0) {
            newVals[layerId][t][clusterName] = 0;
            continue;
          }

          let onCount = 0;
          for (const latentId of enabledLatents) {
            if (latentId < latArray.length && latArray[latentId] > 0) {
              onCount++;
            }
          }
          const fraction = onCount / enabledLatents.length;
          newVals[layerId][t][clusterName] = fraction;
        }
      }
    }
    setActivationValues(newVals);
  }

  // ----------------------------------------------------------------
  // Hover arrows logic
  // ----------------------------------------------------------------
  function handleMouseEnter(layerId: number, colIndex: number) {
    const col = columns[colIndex];
    if (col.ablated) {
      setHoveredCell(null);
      setArrowLines([]);
      return;
    }
    setHoveredCell({ layerId, colIndex });
    setArrowLines(computeArrowLines(layerId, colIndex));
  }

  function handleMouseLeave() {
    setHoveredCell(null);
    setArrowLines([]);
  }

  function computeArrowLines(hoverLayerId: number, hoverColIndex: number) {
    const lines: any[] = [];
    const rowIndex = LAYERS.indexOf(hoverLayerId);
    if (rowIndex <= 0) return lines;
    const aboveLayerId = LAYERS[rowIndex - 1];

    const tableRect = tableRef.current?.getBoundingClientRect();
    if (!tableRect) return lines;

    const hoveredEl = cellRefs.current[`${hoverLayerId}-${hoverColIndex}`];
    if (!hoveredEl) return lines;
    const hoveredCenter = getCellCenter(hoveredEl, tableRect);

    for (let c = 0; c <= hoverColIndex; c++) {
      if (columns[c].ablated) continue;
      const aboveEl = cellRefs.current[`${aboveLayerId}-${c}`];
      if (!aboveEl) continue;
      const aboveCenter = getCellCenter(aboveEl, tableRect);
      lines.push({
        x1: aboveCenter.x,
        y1: aboveCenter.y,
        x2: hoveredCenter.x,
        y2: hoveredCenter.y,
      });
    }
    return lines;
  }

  function getCellCenter(cellEl: HTMLTableCellElement, tableRect: DOMRect) {
    const rect = cellEl.getBoundingClientRect();
    return {
      x: rect.left + rect.width / 2 - tableRect.left,
      y: rect.top + rect.height / 2 - tableRect.top,
    };
  }

  // ----------------------------------------------------------------
  // Render cluster cell
  // ----------------------------------------------------------------
  function ClusterCell({ layerId, col }: { layerId: number; col: any }) {
    if (!clusterStates) {
      return null;
    }
    if (col.ablated) {
      return (
        <div className="h-16 bg-gray-50 rounded-lg flex items-center justify-center">
          (ablated)
        </div>
      );
    }
    // We assume col.tokenIdxs has exactly 1 in normal usage
    const tokenIdx = col.tokenIdxs[0];
    const clusterMap = clusterStates[layerId][tokenIdx] || {};
    const clusterNames = Object.keys(clusterMap);

    if (!clusterNames.length) {
      // treat as ablated
      return (
        <div className="h-16 bg-gray-50 rounded-lg flex items-center justify-center">
          (ablated)
        </div>
      );
    }

    return (
      <TooltipProvider>
        <div className="h-16 p-1 flex flex-wrap gap-1 items-center justify-center flex-col">
          {clusterNames.map((clusterName) => {
            const info = clusterMap[clusterName];
            if (!info) return null;
            const fraction = activationValues[layerId]?.[tokenIdx]?.[clusterName] ?? 0;
            const percent = Math.round(fraction * 100);
            const latentsOn = Object.values(info.latents).filter(Boolean).length;

            return (
              <Tooltip key={clusterName}>
                <TooltipTrigger asChild>
                  <button
                    onClick={() =>
                      setSelectedCell(
                        selectedCell &&
                        selectedCell.layerId === layerId &&
                        selectedCell.tokenIdx === tokenIdx
                          ? null
                          : { layerId, tokenIdx }
                      )
                    }
                    className={`relative px-2 py-1 rounded text-xs ${
                      info.enabled
                        ? 'bg-blue-100 hover:bg-blue-200 border-blue-300'
                        : 'bg-gray-100 hover:bg-gray-200 border-gray-300'
                    } border-l-4 transition-colors min-w-[40px] text-center`}
                  >
                    <div className="font-medium">
                      {clusterName} ({latentsOn})
                    </div>
                    {info.enabled && (
                      <div
                        className="absolute bottom-0 left-0 h-0.5 bg-blue-500"
                        style={{ width: `${percent}%` }}
                      />
                    )}
                  </button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{clusterName}</p>
                  {info.enabled ? (
                    <p>Activation: {percent}%</p>
                  ) : (
                    <p className="text-gray-500">Disabled</p>
                  )}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      </TooltipProvider>
    );
  }

  // ----------------------------------------------------------------
  // Dialog: cluster details
  // ----------------------------------------------------------------
  function ClusterDetails() {
    if (!selectedCell || !clusterStates) return null;
    const { layerId, tokenIdx } = selectedCell;
    const clusterMap = clusterStates[layerId][tokenIdx] || {};
    const clusterNames = Object.keys(clusterMap);

    const saeIndex = layerIndexMap[layerId];

    return (
      <Dialog
        open={!!selectedCell}
        onOpenChange={(open) => {
          if (!open) setSelectedCell(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              SAE {layerId}, Token: “{tokens[tokenIdx] || ''}”
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            {!clusterNames.length && (
              <div className="text-sm italic text-gray-600">
                No clusters for this (layer, token).
              </div>
            )}
            {clusterNames.map((clusterName) => {
              const info = clusterMap[clusterName];
              const latents = Object.keys(info.latents).map(Number);
              return (
                <div key={clusterName} className="border rounded p-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-medium">{clusterName}</span>
                    <Switch
                      checked={info.enabled}
                      onCheckedChange={(checked) =>
                        toggleCluster(layerId, tokenIdx, clusterName, checked)
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    {latents.map((latentId) => {
                      // If we have latentActivations, show the value
                      let activationVal: number | null = null;
                      if (latentActivations) {
                        const latArray = latentActivations[tokenIdx]?.[saeIndex];
                        if (latArray && latentId < latArray.length) {
                          activationVal = latArray[latentId];
                        }
                      }
                      return (
                        <div
                          key={latentId}
                          className="flex justify-between items-center bg-gray-50 p-2 rounded"
                        >
                          <div>
                            <span className="font-mono">#{latentId}</span>
                            {activationVal !== null ? (
                              <span className="ml-2 text-gray-600">
                                Activation: {activationVal.toFixed(2)}
                              </span>
                            ) : (
                              <span className="ml-2 text-gray-400">
                                Activation: --
                              </span>
                            )}
                          </div>
                          <Checkbox
                            disabled={!info.enabled}
                            checked={info.latents[latentId] || false}
                            onCheckedChange={(checked) =>
                              toggleLatent(
                                layerId,
                                tokenIdx,
                                clusterName,
                                latentId,
                                checked
                              )
                            }
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  // ----------------------------------------------------------------
  // Render
  // ----------------------------------------------------------------
  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      {/* Example selection and "run" */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex justify-between items-center">
            <span>Python Code (Backend Tokenization)</span>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={handlePrevExample}
                disabled={exampleIndex === 0}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={handleNextExample}
                disabled={exampleIndex === EXAMPLES.length - 1}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {needRerun && (
            <div className="mb-2 p-2 text-sm font-medium text-yellow-800 bg-yellow-50 border-l-4 border-yellow-400 rounded">
              You have changed the example or toggled latents. Click “Run” to see updated results.
            </div>
          )}
          <div className="flex items-center gap-4">
            <code className="flex-grow p-2 border rounded bg-gray-50">
              {currentExample}
            </code>
            <Button onClick={runCircuit}>
              <Play className="h-4 w-4 mr-2" />
              Run
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Circuit Visualization */}
      <Card>
        <CardHeader>
          <CardTitle>
            Circuit Visualization (Rows = SAEs, Columns = Tokens)
          </CardTitle>
        </CardHeader>
        <CardContent>
          {!latentActivations || !tokens.length ? (
            <div className="text-center p-4 text-gray-600 italic border rounded">
              Click “Run” to fetch tokens & activations from the backend
            </div>
          ) : (
            <div ref={tableRef} className="relative overflow-x-auto">
              <table className="min-w-max border-collapse border">
                <thead>
                  <tr>
                    <th className="border p-2">SAE / Token</th>
                    {columns.map((col, cIdx) => (
                      <th key={cIdx} className="border p-2 text-center">
                        {col.ablated
                          ? `Ablated: ${col.label}`
                          : col.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {LAYERS.map((layerId) => (
                    <tr key={layerId}>
                      <td className="border p-2 font-semibold text-center">
                        SAE {layerId}
                      </td>
                      {columns.map((col, cIdx) => (
                        <td
                          key={`${layerId}-${cIdx}`}
                          className="border p-2"
                          ref={(el) =>
                            (cellRefs.current[`${layerId}-${cIdx}`] = el)
                          }
                          onMouseEnter={() => handleMouseEnter(layerId, cIdx)}
                          onMouseLeave={handleMouseLeave}
                        >
                          <ClusterCell layerId={layerId} col={col} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Arrows for hovered cell */}
              {hoveredCell && arrowLines.length > 0 && (
                <svg
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    pointerEvents: 'none',
                  }}
                >
                  <defs>
                    <marker
                      id="arrowhead"
                      markerWidth="6"
                      markerHeight="6"
                      refX="5"
                      refY="3"
                      orient="auto"
                      fill="gray"
                    >
                      <path d="M0,0 L0,6 L6,3 z" />
                    </marker>
                  </defs>
                  {arrowLines.map((line, i) => (
                    <line
                      key={i}
                      x1={line.x1}
                      y1={line.y1}
                      x2={line.x2}
                      y2={line.y2}
                      stroke="gray"
                      strokeWidth="2"
                      markerEnd="url(#arrowhead)"
                    />
                  ))}
                </svg>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Output probabilities */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Output Token Probabilities</CardTitle>
        </CardHeader>
        <CardContent>
          {topProbs === null ? (
            <div className="p-2 italic text-gray-600">
              No output yet. Click “Run” to see probabilities.
            </div>
          ) : Object.keys(topProbs).length === 0 ? (
            <div className="p-2 italic text-gray-600">
              No probabilities from backend.
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(topProbs).map(([tok, prob]) => (
                <div
                  key={tok}
                  className="flex justify-between items-center p-2 border rounded"
                >
                  <span className="font-medium">{tok}:</span>
                  <span className="text-blue-600">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Cluster Detail Dialog */}
      <ClusterDetails />
    </div>
  );
};

export default CircuitVisualization;
