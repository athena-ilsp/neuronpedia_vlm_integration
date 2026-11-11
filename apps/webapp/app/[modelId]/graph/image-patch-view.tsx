'use client';

import { useGraphContext } from '@/components/provider/graph-provider';
import { useGraphStateContext } from '@/components/provider/graph-state-provider';
import { imageDataUrl, patchCell } from '@/lib/utils/image-prompt';
import { useEffect, useState } from 'react';

/**
 * For a graph generated from an image prompt: the prompt's image, beside the graph. Hovering or
 * clicking a node at an image position highlights that node's patch, so which part of the image a
 * feature sits on can be read off directly. Renders nothing for a text-only graph.
 */
export default function ImagePatchView() {
  const { selectedGraph } = useGraphContext();
  const { hoveredCtxIdxRef, clickedCtxIdxRef, registerHoverCallback, registerClickedCallback } = useGraphStateContext();
  const [focusedCtxIdx, setFocusedCtxIdx] = useState<number | null>(null);

  // The graph's hover and click state live in refs; follow them through its callbacks. A hovered
  // node wins, and the clicked one shows once the pointer leaves.
  useEffect(() => {
    const follow = () => setFocusedCtxIdx(hoveredCtxIdxRef.current ?? clickedCtxIdxRef.current);
    const unregisterHover = registerHoverCallback(follow);
    const unregisterClicked = registerClickedCallback(follow);
    return () => {
      unregisterHover();
      unregisterClicked();
    };
  }, [hoveredCtxIdxRef, clickedCtxIdxRef, registerHoverCallback, registerClickedCallback]);

  const image = selectedGraph?.metadata.image_input;
  if (!image) {
    return null;
  }

  const k = focusedCtxIdx === null ? -1 : image.image_positions.indexOf(focusedCtxIdx);
  const focused = k === -1 ? null : patchCell(k, image.grid_cols);

  return (
    <div className="flex w-[200px] min-w-[200px] flex-col items-center justify-start gap-y-1 pr-2 pt-1">
      <div className="relative aspect-square w-full">
        {/* eslint-disable-next-line @next/next/no-img-element -- the graph's own embedded image */}
        <img
          src={imageDataUrl(image.image_base64)}
          alt="Prompt image"
          className="block h-full w-full rounded-sm border border-slate-300"
        />
        {focused && (
          <div
            className="pointer-events-none absolute border-2 border-emerald-600 bg-emerald-400/40"
            style={{
              left: `${(focused.col / image.grid_cols) * 100}%`,
              top: `${(focused.row / image.grid_rows) * 100}%`,
              width: `${100 / image.grid_cols}%`,
              height: `${100 / image.grid_rows}%`,
            }}
          />
        )}
      </div>
      <div className="text-center text-[10px] leading-tight text-slate-500">
        {focused
          ? `Patch ${focused.row + 1},${focused.col + 1} (row, column)`
          : 'Hover a node at an image position to see its patch'}
      </div>
    </div>
  );
}
