/**
 * Pure helpers for registering source sets by hand (scripts/source-set.ts, `make source-set-add`).
 * No imports, so the ts-node script can load this without the `@/` path alias.
 */

/** The model's raw MLP neurons, which every inference pod serves. Same name as lib/utils/source.ts. */
export const NEURONS_SOURCESET = 'neurons';

/** "0-3,7" -> [0, 1, 2, 3, 7] */
export function parseLayers(spec: string): number[] {
  const layers = new Set<number>();
  spec.split(',').forEach((part) => {
    const [start, end] = part.trim().split('-').map(Number);
    if (!Number.isInteger(start) || (end !== undefined && !Number.isInteger(end))) {
      throw new Error(`Invalid --layers "${spec}"`);
    }
    for (let layer = start; layer <= (end ?? start); layer += 1) {
      layers.add(layer);
    }
  });
  return [...layers].sort((a, b) => a - b);
}

/** Source ids follow the inference server's: bare layer numbers for neurons, `<layer>-<set>` otherwise. */
export function sourceId(layer: number, setName: string): string {
  return setName === NEURONS_SOURCESET ? `${layer}` : `${layer}-${setName}`;
}
