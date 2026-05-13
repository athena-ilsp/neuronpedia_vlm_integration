'use client';

/**
 * VLM extension: shows the original input image plus the strip of image patches that
 * survived into the graph (i.e. patches with at least one informative feature node).
 * Each kept patch is highlighted on the full image with an overlay rectangle, and the
 * strip below mirrors the column order in the graph above so it's easy to scan
 * patch -> graph column visually.
 *
 * Hover/click on a patch tile sets visState.hoveredCtxIdx so the graph highlights the
 * matching column (existing behaviour from the prompt token ticks).
 */

import { useGraphContext } from '@/components/provider/graph-provider';
import { useMemo } from 'react';

const FULL_IMG_PX = 260; // displayed size of the full image preview
const STRIP_TILE_PX = 28; // size of each patch tile in the strip
const STRIP_GAP_PX = 4;

export default function ImagePatchStrip() {
  const { selectedGraph, visState, updateVisStateField } = useGraphContext();

  // Derive: list of (ctx_idx, row, col, patchIndex) for patches kept in the graph,
  // in ctx_idx order. We consider a patch "kept" if at least one node at that ctx_idx
  // has feature_type !== 'embedding'.
  const data = useMemo(() => {
    const image = selectedGraph?.metadata?.image_input;
    if (!image || !selectedGraph) return null;
    const informativeCtxIdx = new Set<number>(
      selectedGraph.nodes
        .filter((n: any) => n.feature_type !== undefined && n.feature_type !== 'embedding')
        .map((n: any) => n.ctx_idx)
        .filter((c: number | undefined) => c !== undefined),
    );
    const kept = image.image_positions
      .map((ctxIdx, k) => ({
        ctxIdx,
        index: k,
        row: Math.floor(k / image.grid_cols),
        col: k % image.grid_cols,
      }))
      .filter((p) => informativeCtxIdx.has(p.ctxIdx));

    const imgSrc = image.image_base64.startsWith('data:')
      ? image.image_base64
      : `data:image/png;base64,${image.image_base64}`;
    return { kept, imgSrc, gridRows: image.grid_rows, gridCols: image.grid_cols };
  }, [selectedGraph]);

  if (!data || data.kept.length === 0) return null;

  const { kept, imgSrc, gridRows, gridCols } = data;
  const cellPx = FULL_IMG_PX / gridCols;
  const hoveredCtx = visState.hoveredCtxIdx;

  return (
    <div className="flex flex-row items-start gap-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
      {/* Full image with per-patch overlay rectangles for kept patches */}
      <div className="relative shrink-0" style={{ width: FULL_IMG_PX, height: FULL_IMG_PX }}>
        <img
          alt="input"
          src={imgSrc}
          width={FULL_IMG_PX}
          height={FULL_IMG_PX}
          className="block rounded-sm border border-slate-300"
          style={{ width: FULL_IMG_PX, height: FULL_IMG_PX }}
        />
        <svg
          className="pointer-events-none absolute inset-0"
          width={FULL_IMG_PX}
          height={FULL_IMG_PX}
        >
          {kept.map((p) => {
            const isHovered = hoveredCtx === p.ctxIdx;
            return (
              <rect
                key={p.ctxIdx}
                x={p.col * cellPx}
                y={p.row * cellPx}
                width={cellPx}
                height={cellPx}
                fill={isHovered ? 'rgba(56,189,248,0.35)' : 'rgba(56,189,248,0.12)'}
                stroke={isHovered ? '#0284c7' : '#38bdf8'}
                strokeWidth={isHovered ? 1.5 : 0.75}
              />
            );
          })}
        </svg>
      </div>

      {/* Strip of kept patches in graph-column order */}
      <div className="flex flex-1 flex-col">
        <div className="mb-1 text-[11px] font-medium text-slate-500">
          Kept image patches ({kept.length}/{gridRows * gridCols})
        </div>
        <div
          className="flex flex-row flex-wrap"
          style={{ gap: STRIP_GAP_PX }}
        >
          {kept.map((p) => {
            const isHovered = hoveredCtx === p.ctxIdx;
            return (
              <button
                key={p.ctxIdx}
                type="button"
                title={`patch (${p.row}, ${p.col}) — ctx_idx ${p.ctxIdx}`}
                onMouseEnter={() => updateVisStateField('hoveredCtxIdx', p.ctxIdx)}
                onMouseLeave={() => updateVisStateField('hoveredCtxIdx', null)}
                className="relative overflow-hidden rounded-sm border"
                style={{
                  width: STRIP_TILE_PX,
                  height: STRIP_TILE_PX,
                  borderColor: isHovered ? '#0284c7' : '#cbd5e1',
                  borderWidth: isHovered ? 2 : 1,
                }}
              >
                <img
                  alt={`patch ${p.row},${p.col}`}
                  src={imgSrc}
                  // Stretch the full image to grid-size, then offset so just this patch shows.
                  style={{
                    position: 'absolute',
                    width: gridCols * STRIP_TILE_PX,
                    height: gridRows * STRIP_TILE_PX,
                    left: -p.col * STRIP_TILE_PX,
                    top: -p.row * STRIP_TILE_PX,
                    maxWidth: 'none',
                  }}
                />
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}