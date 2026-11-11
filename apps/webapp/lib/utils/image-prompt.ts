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

/**
 * Image placeholder token strings, for views that have only a prompt's token strings (a saved
 * graph's `promptTokens`) and no image positions. Gemma 3, then Qwen-VL.
 */
export const IMAGE_PLACEHOLDER_TOKENS = new Set(['<image_soft_token>', '<|image_pad|>']);

/** Positions of image placeholder tokens in `tokens`, by their token string. */
export function imagePositionsFromTokens(tokens: string[]): number[] {
  return tokens.flatMap((token, i) => (IMAGE_PLACEHOLDER_TOKENS.has(token) ? [i] : []));
}

export type TokenChip = {
  /** The token, or `IMAGE_TOKEN_LABEL` for a run of image patches. */
  token: string;
  /** How many image patches the chip stands for; 0 for a text token. */
  imagePatches: number;
};

/**
 * Tokens for a chip-per-token view, with each contiguous run of image positions as one chip.
 * `imagePositions` defaults to the placeholder tokens found in `tokens`.
 */
export function tokenChips(tokens: string[], imagePositions?: number[]): TokenChip[] {
  const image = new Set(imagePositions ?? imagePositionsFromTokens(tokens));
  const chips: TokenChip[] = [];
  tokens.forEach((token, i) => {
    if (!image.has(i)) {
      chips.push({ token, imagePatches: 0 });
    } else if (i > 0 && image.has(i - 1)) {
      chips[chips.length - 1].imagePatches += 1;
    } else {
      chips.push({ token: IMAGE_TOKEN_LABEL, imagePatches: 1 });
    }
  });
  return chips;
}

/** `image` as a URL an <img> can show: a data URL as-is, bare base64 as a PNG/JPEG data URL. */
export function imageDataUrl(image: string): string {
  return image.startsWith('data:') ? image : `data:image/png;base64,${image}`;
}

/** Tokens that are not image placeholders: what a text token limit counts. */
export function textTokenCount(tokens: string[]): number {
  return tokens.length - imagePositionsFromTokens(tokens).length;
}

/** A chip's text: the token, or "[image] ×N" for N image patches. */
export function chipLabel(chip: TokenChip): string {
  return chip.imagePatches > 0 ? `${chip.token} ×${chip.imagePatches}` : chip.token;
}
