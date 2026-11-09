import { describe, expect, it } from 'vitest';
import { collapseImageTokens, IMAGE_TOKEN_LABEL, patchCell, patchLabel, strongestPatch } from './image-prompt';

// <bos> <turn> <img> <img> <img> <img> text <eot>
const tokens = ['<bos>', '<turn>', '<i>', '<i>', '<i>', '<i>', 'cat', '<eot>'];
const values = [0, 9, 1, -5, 2, 3, 4, 0];
const image = [2, 3, 4, 5];

describe('collapseImageTokens', () => {
  it('replaces the patches with one image token at the first patch, valued at the strongest patch', () => {
    const out = collapseImageTokens(tokens, values, image);
    expect(out.tokens).toEqual(['<bos>', '<turn>', IMAGE_TOKEN_LABEL, 'cat', '<eot>']);
    expect(out.values).toEqual([0, 9, -5, 4, 0]);
    expect(out.originalIndex).toEqual([0, 1, 2, 6, 7]);
    expect(out.imageIndex).toBe(2);
  });

  it('leaves a text-only prompt unchanged', () => {
    const out = collapseImageTokens(['a', 'b'], [1, 2], []);
    expect(out.tokens).toEqual(['a', 'b']);
    expect(out.values).toEqual([1, 2]);
    expect(out.originalIndex).toEqual([0, 1]);
    expect(out.imageIndex).toBe(-1);
  });
});

describe('strongestPatch', () => {
  it('keeps the sign of the largest-magnitude patch', () => {
    expect(strongestPatch(values, image)).toBe(-5);
    expect(strongestPatch(values, [])).toBe(0);
  });
});

describe('patch positions', () => {
  it('lays patches out row-major', () => {
    expect(patchCell(0, 16)).toEqual({ row: 0, col: 0 });
    expect(patchCell(17, 16)).toEqual({ row: 1, col: 1 });
  });

  it('labels image positions by grid cell, and nothing else', () => {
    expect(patchLabel(3, image, 2)).toBe('patch 1,2');
    expect(patchLabel(5, image, 2)).toBe('patch 2,2');
    expect(patchLabel(5, image, null)).toBe('patch 4');
    expect(patchLabel(6, image, 2)).toBeUndefined();
  });
});
