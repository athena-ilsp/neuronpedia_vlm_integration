/**
 * Display helpers for image (vision-language) prompts.
 *
 * An image expands into many placeholder tokens (256 on Gemma 3), one per patch, which would bury
 * the text in any per-token view. These collapse the patches into one token for text-shaped views,
 * and lay them out on the patch grid for image-shaped ones. Pure functions over the inference
 * server's `image` block (`ActivationImage`), so they know no model's token literals.
 */

/** Label shown in place of the collapsed image patches. */
export const IMAGE_TOKEN_LABEL = '[image]';

export type CollapsedImageTokens = {
  tokens: string[];
  values: number[];
  /** For each collapsed index, the original token index (the first patch for the image token). */
  originalIndex: number[];
  /** Collapsed index of the image token, or -1 when there are no image positions. */
  imageIndex: number;
};

/**
 * Replace the image placeholder tokens with one `IMAGE_TOKEN_LABEL` token at the first patch's
 * place, valued at the strongest patch (largest absolute value, sign kept).
 */
export function collapseImageTokens(
  tokens: string[],
  values: number[],
  imagePositions: number[],
): CollapsedImageTokens {
  const image = new Set(imagePositions);
  const out: CollapsedImageTokens = { tokens: [], values: [], originalIndex: [], imageIndex: -1 };
  tokens.forEach((token, i) => {
    if (!image.has(i)) {
      out.tokens.push(token);
      out.values.push(values[i] ?? 0);
      out.originalIndex.push(i);
      return;
    }
    if (out.imageIndex === -1) {
      out.imageIndex = out.tokens.length;
      out.tokens.push(IMAGE_TOKEN_LABEL);
      out.values.push(strongestPatch(values, imagePositions));
      out.originalIndex.push(i);
    }
  });
  return out;
}

/** The patch value with the largest magnitude, sign kept; 0 for no patches. */
export function strongestPatch(values: number[], imagePositions: number[]): number {
  let best = 0;
  imagePositions.forEach((p) => {
    const v = values[p] ?? 0;
    if (Math.abs(v) > Math.abs(best)) {
      best = v;
    }
  });
  return best;
}

/** Row and column (0-based) of the k-th image position on a row-major grid. */
export function patchCell(k: number, cols: number): { row: number; col: number } {
  return { row: Math.floor(k / cols), col: k % cols };
}

/** "patch r,c" for the token at `tokenIndex`, or undefined when it is not an image position. */
export function patchLabel(tokenIndex: number, imagePositions: number[], cols: number | null | undefined) {
  const k = imagePositions.indexOf(tokenIndex);
  if (k === -1) {
    return undefined;
  }
  if (!cols) {
    return `patch ${k + 1}`;
  }
  const { row, col } = patchCell(k, cols);
  return `patch ${row + 1},${col + 1}`;
}
