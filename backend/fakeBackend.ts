// backend/fakeBackend.ts

import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(bodyParser.json());

// -----------------------------------------
// 1) Mock Tokenize
// -----------------------------------------
function mockTokenize(prompt: string): string[] {
  // For demonstration: memorized tokenizations for certain strings.
  // If not in memorized, just do a naive split.
  const memorized: Record<string, string[]> = {
    'predict: ages["Bob"]': ['predict:', 'ages', '[', '"Bob"', ']'],
    'predict: prices["Apples"]': ['predict:', 'prices', '[', '"Apples"', ']'],
    'predict: dictionary["Hello"]': ['predict:', 'dictionary', '[', '"Hello"', ']'],
  };

  if (memorized[prompt]) {
    return memorized[prompt];
  }
  // fallback
  return prompt.split(/\s+/);
}

// -----------------------------------------
// 2) Random/zero generation for latents
// -----------------------------------------
function randomValue() {
  // half the time 0, half the time a random float in [0..10]
  return Math.random() < 0.5 ? 0 : Math.random() * 10;
}

/**
 * Generate a baseline activation tensor of shape:
 * [ numTokens, numSAEs, numLatents ]
 *
 * For this example, we assume LAYERS = 3 => numSAEs=3,
 * and numLatents=16384 (arbitrary large dimension).
 */
function generateLatentActivations(
  numTokens: number,
  numSAEs = 3,
  numLatents = 16384
): number[][][] {
  const outer: number[][][] = [];
  for (let t = 0; t < numTokens; t++) {
    const saeList: number[][] = [];
    for (let s = 0; s < numSAEs; s++) {
      // fill with random 0..10 or 0
      const latents = Array.from({ length: numLatents }, () => randomValue());
      saeList.push(latents);
    }
    outer.push(saeList);
  }
  return outer;
}

// -----------------------------------------
// 3) Layers array (must match your frontend)
// -----------------------------------------
const LAYERS = [7, 21, 40];
const layerIndexMap: Record<number, number> = {};
LAYERS.forEach((layerId, i) => {
  layerIndexMap[layerId] = i;
});

// -----------------------------------------
// POST /runWithLatentMask
// Body: { prompt: string, toggles: { [layerId]: { [tokenIndex]: number[] } } }
// -----------------------------------------
app.post('/runWithLatentMask', (req, res) => {
  const { prompt, toggles } = req.body;

  // 1) Tokenize the prompt
  const tokens = mockTokenize(prompt);
  const numTokens = tokens.length;
  const numSAEs = LAYERS.length;

  // 2) Generate baseline [numTokens, numSAEs, 16384]
  const baselineActivations = generateLatentActivations(numTokens, numSAEs, 16384);

  // 3) Zero out any latents that are *not* in toggles
  for (let t = 0; t < numTokens; t++) {
    for (let s = 0; s < LAYERS.length; s++) {
      const layerId = LAYERS[s];
      // toggles[layerId][t] = array of latents we want to keep "on"
      const latentsToEnable: number[] = toggles?.[layerId]?.[t] || [];
      const enableSet = new Set(latentsToEnable);

      // baselineActivations[t][s] is the array of length 16384
      const latArray = baselineActivations[t][s];
      for (let latentId = 0; latentId < latArray.length; latentId++) {
        if (!enableSet.has(latentId)) {
          latArray[latentId] = 0;
        }
      }
    }
  }

  // 4) Generate random topProbs
  const topProbs = {
    Traceback: Math.random(),
    '1': Math.random(),
    '>>>': Math.random(),
  };

  // 5) Send final JSON
  res.json({
    prompt,
    tokens,
    toggles,
    result: {
      topProbs,
      latentActivations: baselineActivations,
    },
  });
});

// -----------------------------------------
// Start server
// -----------------------------------------
const PORT = 4000;
app.listen(PORT, () => {
  console.log(`Fake backend is running on http://localhost:${PORT}`);
});
