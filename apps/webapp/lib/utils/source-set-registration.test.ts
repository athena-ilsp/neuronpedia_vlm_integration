import { describe, expect, it } from 'vitest';
import { parseLayers, sourceId } from './source-set-registration';

describe('parseLayers', () => {
  it('expands ranges and lists, sorted and deduplicated', () => {
    expect(parseLayers('0-3')).toEqual([0, 1, 2, 3]);
    expect(parseLayers('5,0-1,1')).toEqual([0, 1, 5]);
  });

  it('refuses anything that is not a layer number', () => {
    expect(() => parseLayers('a-3')).toThrow('Invalid --layers');
  });
});

describe('sourceId', () => {
  it('matches the inference server: bare layer numbers for neurons, layer-set otherwise', () => {
    expect(sourceId(10, 'neurons')).toBe('10');
    expect(sourceId(10, 'xai-tc-200m-16x')).toBe('10-xai-tc-200m-16x');
  });
});
