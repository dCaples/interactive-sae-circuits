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
import circuit from './circuit.json';

//
// ----------------------------------------------------------------------
// Configuration & Constants
// ----------------------------------------------------------------------
const BACKEND_URL = 'http://194.68.245.78:22119';

// Example prompts
const EXAMPLES = [
  `Type "help", "copyright", "credits" or "license" for more information.\n>>> age = {'Sophia': 19, 'Chris': 14, 'Alice': 16, 'Sally': 15, 'Emily': 17}\n>>> age["Chris"]\n`,
  `Type "help", "copyright", "credits" or "license" for more information.\n>>> age = {'Sophia': 19, 'Grace': 14, 'Alice': 16, 'Sally': 15, 'Emily': 17}\n>>> age["Chris"]\n`,
];

// We assume these are the actual layers in ascending order
const LAYERS = [7, 14, 21, 40];

const CLUSTERS: Record<number, Record<number, Record<string, number[]>>> = circuit as any;

function initClusterStatesForTokens(tokenCount: number) {
  const result: Record<
    number,
    Record<
      number,
      {
        [clusterName: string]: {
          enabled: boolean;
          latents: Record<number, boolean>;
        };
      }
    >
  > = {};

  for (const layerId of LAYERS) {
    result[layerId] = {};
    for (let t = 0; t < tokenCount; t++) {
      const def = CLUSTERS[layerId]?.[t] || {};
      const clusterMap: any = {};
      for (const [clusterName, latentIds] of Object.entries(def)) {
        // By default, all latents in that cluster are "on"
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

function isTokenAblatedAcrossAllLayers(
  clusterStates: ReturnType<typeof initClusterStatesForTokens>,
  tokenIdx: number
) {
  for (const layerId of LAYERS) {
    const clusterMap = clusterStates[layerId]?.[tokenIdx];
    if (clusterMap && Object.keys(clusterMap).length > 0) {
      // There is at least one cluster => not fully ablated
      return false;
    }
  }
  return true;
}

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
      // Single token or non-ablated chunk
      const displayedToken = tokens[i] === '\n' ? '\\n' : tokens[i];
      columns.push({
        ablated: false,
        tokenIdxs: [i],
        label: displayedToken,
      });
      i++;
    } else {
      // This chunk is ablated across all layers
      const start = i;
      let end = i;
      while (
        end < tokens.length &&
        isTokenAblatedAcrossAllLayers(clusterStates, end)
      ) {
        end++;
      }

      const chunk = tokens
        .slice(start, end)
        .map((t) => (t === '\n' ? '\\n' : t))
        .join(' + ');

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

/**
 * Build two dictionaries for the backend:
 *  1) toggles  -> latents that are "on":     toggles=1 => keep
 *  2) zeroToggles -> latents that are "off": zeroToggles=1 => forcibly zero
 */
function buildToggleDictionaries(
  clusterStates: ReturnType<typeof initClusterStatesForTokens>,
  tokens: string[]
) {
  const toggles: Record<string, Record<string, Record<string, number>>> = {};
  const zeroToggles: Record<string, Record<string, Record<string, number>>> = {};

  for (const layerId of LAYERS) {
    toggles[String(layerId)] = {};
    zeroToggles[String(layerId)] = {};

    for (let t = 0; t < tokens.length; t++) {
      const clusterMap = clusterStates[layerId][t];
      const togglesForToken: Record<string, number> = {};
      const zeroTogglesForToken: Record<string, number> = {};

      if (clusterMap) {
        // For each cluster at (layer, token)
        for (const [clusterName, cInfo] of Object.entries(clusterMap)) {
          const { enabled, latents } = cInfo;
          for (const [latentIdStr, isOn] of Object.entries(latents)) {
            const latentId = parseInt(latentIdStr);
            if (enabled && isOn) {
              // "on" => keep => toggles=1
              togglesForToken[latentIdStr] = 1.0;
            } else {
              // "off" => forcibly zero => zeroToggles=1
              zeroTogglesForToken[latentIdStr] = 1.0;
            }
          }
        }
      }

      toggles[String(layerId)][String(t)] = togglesForToken;
      zeroToggles[String(layerId)][String(t)] = zeroTogglesForToken;
    }
  }

  return { toggles, zeroToggles };
}

function buildRequestedLatents(tokenCount: number) {
  const requested: Record<number, Record<number, Set<number>>> = {};

  for (const layerId of LAYERS) {
    requested[layerId] = {};
    for (let t = 0; t < tokenCount; t++) {
      requested[layerId][t] = new Set<number>();
    }
  }

  for (const layerId of LAYERS) {
    const definitionForLayer = CLUSTERS[layerId] || {};
    for (const [tokenIdxStr, clusterMap] of Object.entries(definitionForLayer)) {
      const tokenIdx = parseInt(tokenIdxStr);
      for (const latentIds of Object.values(clusterMap)) {
        latentIds.forEach((lid) => {
          requested[layerId][tokenIdx].add(lid);
        });
      }
    }
  }

  const requestedLatents: Record<number, Record<number, number[]>> = {};
  for (const layerId of LAYERS) {
    requestedLatents[layerId] = {};
    for (let t = 0; t < tokenCount; t++) {
      requestedLatents[layerId][t] = Array.from(requested[layerId][t]);
    }
  }

  return requestedLatents;
}

type LatentActivationsMap = {
  [layerId: number]: {
    [tokenIdx: number]: {
      [latentId: number]: number; // some float or activation
    };
  };
};

const CircuitVisualization = () => {
  const [exampleIndex, setExampleIndex] = useState(0);
  const currentExample = EXAMPLES[exampleIndex];

  const [tokens, setTokens] = useState<string[]>([]);
  const [clusterStates, setClusterStates] = useState<
    ReturnType<typeof initClusterStatesForTokens> | null
  >(null);

  const [needRerun, setNeedRerun] = useState(false);
  const [topProbs, setTopProbs] = useState<Record<string, number> | null>(null);

  const [latentActivations, setLatentActivations] =
    useState<LatentActivationsMap | null>(null);

  const [activationValues, setActivationValues] = useState<any>({});

  const [selectedCell, setSelectedCell] = useState<null | {
    layerId: number;
    tokenIdx: number;
  }>(null);

  const [hoveredCell, setHoveredCell] = useState<null | {
    layerId: number;
    colIndex: number;
  }>(null);
  const [arrowLines, setArrowLines] = useState<any[]>([]);
  const tableRef = useRef<HTMLDivElement | null>(null);
  const cellRefs = useRef<Record<string, HTMLTableCellElement | null>>({});

  // Build the final columns (tokens) for display
  const columns = clusterStates ? buildColumns(clusterStates, tokens) : [];

  // On page load, tokenize the first example
  React.useEffect(() => {
    fetchTokensForPrompt(currentExample);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function fetchTokensForPrompt(prompt: string) {
    try {
      const res = await fetch(`${BACKEND_URL}/tokenize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      const newTokens = data.tokens || [];

      setTokens(newTokens);
      const newCS = initClusterStatesForTokens(newTokens.length);
      setClusterStates(newCS);

      // Reset states
      setTopProbs(null);
      setLatentActivations(null);
      setActivationValues({});
      setNeedRerun(true);
    } catch (err) {
      console.error('Error in /tokenize:', err);
    }
  }

  function handlePrevExample() {
    if (exampleIndex > 0) {
      const newIndex = exampleIndex - 1;
      setExampleIndex(newIndex);
      fetchTokensForPrompt(EXAMPLES[newIndex]);
    }
  }

  function handleNextExample() {
    if (exampleIndex < EXAMPLES.length - 1) {
      const newIndex = exampleIndex + 1;
      setExampleIndex(newIndex);
      fetchTokensForPrompt(EXAMPLES[newIndex]);
    }
  }

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
      return { ...prev, [layerId]: newLayerObj };
    });
    setNeedRerun(true);
  }

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
      return { ...prev, [layerId]: newLayerObj };
    });
    setNeedRerun(true);
  }

  /**
   * Build the toggles & zeroToggles from clusterStates,
   * then POST to the backend to run the circuit.
   */
  async function runCircuit() {
    if (!clusterStates) {
      alert('No cluster states found. Try retokenizing first.');
      return;
    }

    // A) Convert clusterStates -> toggles + zeroToggles
    const { toggles, zeroToggles } = buildToggleDictionaries(clusterStates, tokens);

    // B) Also build a "requestedLatents" dict, so we retrieve final latents
    const requestedLatents = buildRequestedLatents(tokens.length);

    try {
      const payload = {
        prompt: currentExample,
        toggles,
        zeroToggles,       // <--- Pass the new zeroToggles
        requestedLatents,
      };
      const res = await fetch(`${BACKEND_URL}/runWithLatentMask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      const { tokens: newTokens, result } = data;
      const { topProbs, latentActivations } = result || {};

      // If tokens changed in length for some reason
      if (newTokens.length !== tokens.length) {
        console.warn(
          `Token mismatch: got ${newTokens.length} from backend, local is ${tokens.length}`
        );
      }

      setTopProbs(topProbs || {});
      setLatentActivations(latentActivations || null);

      if (latentActivations) {
        computeClusterActivations(latentActivations, tokens.length);
      } else {
        setActivationValues({});
      }

      setNeedRerun(false);
    } catch (err) {
      console.error('Error in /runWithLatentMask:', err);
    }
  }

  function computeClusterActivations(
    activations: LatentActivationsMap,
    tokenCount: number
  ) {
    if (!clusterStates) return;
    const newVals: any = {};

    for (const layerId of LAYERS) {
      for (let t = 0; t < tokenCount; t++) {
        if (!newVals[layerId]) newVals[layerId] = {};
        if (!newVals[layerId][t]) newVals[layerId][t] = {};

        const latMap = activations?.[layerId]?.[t] || {};
        const clusterMap = clusterStates[layerId][t];
        if (!clusterMap) continue;

        for (const [clusterName, clusterObj] of Object.entries(clusterMap)) {
          const { enabled, latents } = clusterObj;
          // If cluster is disabled, we might show 0 or skip; your choice
          if (!enabled) {
            newVals[layerId][t][clusterName] = 0;
            continue;
          }
          // Otherwise compute fraction of latents > 0
          const latentIdsEnabled = Object.entries(latents)
            .filter(([_, isOn]) => isOn)
            .map(([lidStr]) => parseInt(lidStr));
          let onCount = 0;
          latentIdsEnabled.forEach((lid) => {
            if (latMap[lid] && latMap[lid] > 0) {
              onCount++;
            }
          });
          const fraction =
            latentIdsEnabled.length > 0
              ? onCount / latentIdsEnabled.length
              : 0;
          newVals[layerId][t][clusterName] = fraction;
        }
      }
    }
    setActivationValues(newVals);
  }

  function handleMouseEnter(layerId: number, colIndex: number) {
    if (!columns[colIndex] || columns[colIndex].ablated) {
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

    // For demonstration, link from all columns <= colIndex
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

  function ClusterCell({ layerId, col }: { layerId: number; col: any }) {
    if (!clusterStates) return null;
    if (col.ablated) {
      return (
        <div className="h-16 bg-gray-50 rounded-lg flex items-center justify-center">
          (ablated)
        </div>
      );
    }
    const tokenIdx = col.tokenIdxs[0];
    const clusterMap = clusterStates[layerId][tokenIdx] || {};
    const clusterNames = Object.keys(clusterMap);
    if (!clusterNames.length) {
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
            const fraction =
              activationValues?.[layerId]?.[tokenIdx]?.[clusterName] ?? 0;
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

  /* =======================
     Dialog: cluster details
     ======================= */
  function ClusterDetails() {
    if (!selectedCell || !clusterStates) return null;
    const { layerId, tokenIdx } = selectedCell;
    const clusterMap = clusterStates[layerId]?.[tokenIdx] || {};
    const latMap = latentActivations?.[layerId]?.[tokenIdx] || {};
    const clusterNames = Object.keys(clusterMap);

    return (
      <Dialog
        open={!!selectedCell}
        onOpenChange={(open) => {
          if (!open) setSelectedCell(null);
        }}
      >
        <DialogContent className="max-h-[80vh] overflow-y-auto">
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
                      const activationVal = latMap[latentId] ?? null;
                      const isLatentOn = info.latents[latentId] || false;

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

                          <div className="flex items-center gap-2">
                            {/* CHECKBOX for enabling/disabling latent */}
                            <Checkbox
                              disabled={!info.enabled}
                              checked={isLatentOn}
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
                            {/* Example: link to a "Neuronpedia" page */}
                            <Button variant="link" size="sm" asChild>
                              <a
                                href={`https://www.neuronpedia.org/gemma-2-9b/${layerId}-gemmascope-res-16k/${latentId}`}
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                Neuronpedia
                              </a>
                            </Button>
                          </div>
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

  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      {/* Example selection + Run */}
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
              You have changed tokens or toggled latents. Click “Run” to see updated results.
            </div>
          )}
          <div className="flex flex-col items-left gap-4 overflow-x-auto">
            <code className="flex-grow p-2 border rounded bg-gray-50 whitespace-pre">
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
          <CardTitle>Circuit Visualization (Rows = SAEs, Columns = Tokens)</CardTitle>
        </CardHeader>
        <CardContent>
          {!clusterStates || !tokens.length ? (
            <div className="text-center p-4 text-gray-600 italic border rounded">
              No tokens yet. Please pick an example or wait for tokenization.
            </div>
          ) : (
            <div ref={tableRef} className="relative overflow-x-auto">
              <table className="min-w-max border-collapse border">
                <thead>
                  <tr>
                    <th className="border p-2">SAE / Token</th>
                    {columns.map((col, cIdx) => (
                      <th key={cIdx} className="border p-2 text-center">
                        {col.ablated ? 'Ablated' : col.label}
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
                          ref={(el) => (cellRefs.current[`${layerId}-${cIdx}`] = el)}
                          // onMouseEnter={() => handleMouseEnter(layerId, cIdx)}
                          // onMouseLeave={handleMouseLeave}
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
