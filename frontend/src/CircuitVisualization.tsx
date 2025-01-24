import React, { useState, useEffect } from 'react';
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
import { ChevronLeft, ChevronRight, Play, Loader2 } from 'lucide-react';
import { Switch } from '@/components/ui/switch';
import { Checkbox } from '@/components/ui/checkbox';
import Xarrow, { Xwrapper } from 'react-xarrows';
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

const CLUSTERS: Record<number, Record<number, Record<string, number[]>>> =
  circuit as any;

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
 *  1) toggles     -> latents that are "on": toggles=1 => keep
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

export default function CircuitVisualization() {
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

  // For arrow animation
  // -----------------------------------------
  // We store connections row-by-row (an array of arrays).
  const [connectionsByRow, setConnectionsByRow] = useState<any[]>([]);
  // currentStage=0 => show no arrows, else show the row connections for currentStage-1
  const [currentStage, setCurrentStage] = useState(0);

  // Track loading state
  const [isRunning, setIsRunning] = useState(false);

  // On page load, tokenize the first example
  useEffect(() => {
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

      setConnectionsByRow([]);
      setCurrentStage(0);
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
   * Main "run" function that:
   * 1) Sends toggles + zeroToggles to the backend
   * 2) Receives new topProbs + latentActivations
   * 3) Computes clusterActivations
   * 4) Builds the arrow connections
   * 5) Animates arrow drawing, only one row of arrows at a time
   */
  async function runCircuit() {
    if (!clusterStates) {
      alert('No cluster states found. Try retokenizing first.');
      return;
    }
    // Show loading spinner
    setIsRunning(true);

    // 1) Convert clusterStates -> toggles + zeroToggles
    const { toggles, zeroToggles } = buildToggleDictionaries(clusterStates, tokens);

    // 2) Also build a "requestedLatents" dict, so we retrieve final latents
    const requestedLatents = buildRequestedLatents(tokens.length);

    try {
      const payload = {
        prompt: currentExample,
        toggles,
        zeroToggles,
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

      // 3) If we got new latentActivations, compute clusterActivations
      if (latentActivations) {
        computeClusterActivations(latentActivations, tokens.length);
      } else {
        setActivationValues({});
      }

      setNeedRerun(false);

      // 4) Build arrow connections + animate them
      await buildAndAnimateArrows();
    } catch (err) {
      console.error('Error in /runWithLatentMask:', err);
    } finally {
      // Done loading
      setIsRunning(false);
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
          if (!enabled) {
            // If cluster is disabled, we show 0 (or skip)
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

  /** 
   * We treat each table cell with clusters as a "box". 
   * Return 2D array tableLayout[row][col] = "box" if there's a cluster, "" otherwise.
   */
  function buildTableLayout(columns: any[]) {
    if (!clusterStates) return [];
    // rows = LAYERS.length, columns = columns.length
    const layout = LAYERS.map(() => [] as string[]);
    for (let r = 0; r < LAYERS.length; r++) {
      for (let c = 0; c < columns.length; c++) {
        const isAblated = columns[c].ablated;
        if (!isAblated) {
          // There's a single token index for this column
          const tokenIdx = columns[c].tokenIdxs[0];
          const clusterMap = clusterStates[LAYERS[r]]?.[tokenIdx];
          if (clusterMap && Object.keys(clusterMap).length > 0) {
            layout[r].push('box');
          } else {
            layout[r].push('');
          }
        } else {
          // ablated => no clusters => ""
          layout[r].push('');
        }
      }
    }
    return layout;
  }

  /**
   * Build connections row-by-row, from row i => row i+1.
   * For each "box" in row i col c, connect it to row i+1 col c2 for all c2 >= c
   * if row i+1 col c2 is also "box".
   */
  function buildConnections(layout: string[][]) {
    const connections: any[] = [];

    for (let row = 0; row < layout.length - 1; row++) {
      const nextRow = row + 1;
      const rowConnections = [];

      for (let col = 0; col < layout[row].length; col++) {
        if (layout[row][col] === 'box') {
          // Look for boxes in the next row from col..end
          for (let c = col; c < layout[nextRow].length; c++) {
            if (layout[nextRow][c] === 'box') {
              rowConnections.push({
                start: `cell-r${row}-c${col}`,
                end: `cell-r${nextRow}-c${c}`,
              });
            }
          }
        }
      }
      connections.push(rowConnections);
    }
    return connections;
  }

  /**
   * Build connections, store them in "connectionsByRow",
   * then animate row by row, showing only a single row at a time.
   */
  async function buildAndAnimateArrows() {
    if (!clusterStates) return;
    // 1) Build table layout from cluster data
    const cols = clusterStates ? buildColumns(clusterStates, tokens) : [];
    const layout = buildTableLayout(cols);

    // 2) Build connections from layout
    const conByRow = buildConnections(layout);
    setConnectionsByRow(conByRow);

    // 3) Animate them. For each row i:
    //   1) setCurrentStage = i+1 => show row i
    //   2) wait some time for it to fully animate
    //   3) hide them => setCurrentStage(0)
    //   4) short pause
    for (let i = 0; i < conByRow.length; i++) {
      setCurrentStage(i + 1);   // reveal row i
      await delay(500);        // let the arrow draw
      setCurrentStage(0);       // hide them
      await delay(200);         // short gap before next
    }
  }

  function delay(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function ClusterCell({ layerId, col }: { layerId: number; col: any }) {
    if (!clusterStates) return null;

    // We'll give each cell a stable ID, used for Xarrow references
    // layerId => row index, but let's find row index in LAYERS array
    const rowIndex = LAYERS.indexOf(layerId);
    const colIndex = col._colIndex; // We'll store _colIndex in the columns array below

    if (col.ablated) {
      // Just an empty space
      return (
        <div
          id={`cell-r${rowIndex}-c${colIndex}`}
          className="h-16 bg-gray-50 rounded-lg flex items-center justify-center"
        >
          &nbsp;
        </div>
      );
    }

    const tokenIdx = col.tokenIdxs[0];
    const clusterMap = clusterStates[layerId][tokenIdx] || {};
    const clusterNames = Object.keys(clusterMap);
    if (!clusterNames.length) {
      // Also empty if no cluster names
      return (
        <div
          id={`cell-r${rowIndex}-c${colIndex}`}
          className="h-16 bg-gray-50 rounded-lg flex items-center justify-center"
        >
          &nbsp;
        </div>
      );
    }

    return (
      <TooltipProvider>
        <div
          id={`cell-r${rowIndex}-c${colIndex}`}
          className="h-16 p-1 flex flex-wrap gap-1 items-center justify-center flex-col bg-white border border-gray-200 rounded relative"
        >
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
                    // UPDATED: fixed width at 160px
                    className={`relative w-[160px] px-2 py-1 rounded text-xs ${
                      info.enabled
                        ? 'bg-blue-100 hover:bg-blue-200 border-blue-300'
                        : 'bg-gray-100 hover:bg-gray-200 border-gray-300'
                    } border-l-4 transition-colors text-center`}
                  >
                    <div className="font-medium">
                      {clusterName} ({latentsOn})
                    </div>

                    {/* UPDATED: Grey track behind the blue-filled bar */}
                    {info.enabled && (
                      <div className="absolute bottom-0 left-0 w-full h-0.5 bg-gray-300">
                        <div
                          className="h-0.5 bg-red-500"
                          style={{ width: `${percent}%` }}
                        />
                      </div>
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

  // Build the final columns (tokens) for display
  const columns = clusterStates ? buildColumns(clusterStates, tokens) : [];
  columns.forEach((col, idx) => {
    (col as any)._colIndex = idx;
  });

  // Only show connections for the currentStage’s row (or none if stage=0)
  const visibleConnections =
    currentStage === 0 ? [] : connectionsByRow[currentStage - 1] || [];

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
                disabled={exampleIndex === 0 || isRunning}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={handleNextExample}
                disabled={exampleIndex === EXAMPLES.length - 1 || isRunning}
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
          <div className="flex flex-col  gap-4">
            <code className="flex-grow p-2 border rounded bg-gray-50 whitespace-pre overflow-x-auto">
              {currentExample}
            </code>
            <Button onClick={runCircuit} disabled={isRunning}>
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-2" />
                  Run
                </>
              )}
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
            // Wrap the table and arrows in Xwrapper
            <Xwrapper>
              <div style={{ position: 'relative', overflowX: 'auto' }}>
                <table
                  className="min-w-max border-collapse border"
                  style={{ position: 'relative', zIndex: 1 }}
                >
                  <thead>
                    <tr>
                      <th className="border p-2">SAE / Token</th>
                      {columns.map((col, cIdx) => (
                        <th key={cIdx} className="border p-2 text-center">
                          {col.ablated ? '' : col.label}
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
                          <td key={`${layerId}-${cIdx}`} className="border p-2">
                            <ClusterCell layerId={layerId} col={col} />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* Render the "visible" arrows for the currently active row */}
                {visibleConnections.map(({ start, end }, i) => (
                  <Xarrow
                    key={i}
                    start={start}
                    end={end}
                    color="red"
                    strokeWidth={2}
                    animateDrawing={0.3}
                    headSize={3}
                    // anchor points at the center of each cell
                    startAnchor="middle"
                    endAnchor="middle"
                    // raise zIndex so the arrow is on top
                    zIndex={9999}
                  />
                ))}
              </div>
            </Xwrapper>
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
}
