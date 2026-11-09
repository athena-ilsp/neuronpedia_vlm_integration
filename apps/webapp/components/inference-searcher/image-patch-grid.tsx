'use client';

import { patchCell } from '@/lib/utils/image-prompt';

/**
 * The prompt image with its patch grid on top: a picker (`onTogglePatch`), a heatmap (`values`),
 * or both. Patches are the image's placeholder tokens in row-major order, so `positions[k]` is the
 * token index of the k-th cell. Without a known grid only the image is shown.
 */
export default function ImagePatchGrid({
  imageSrc,
  rows,
  cols,
  positions,
  values,
  selected,
  onTogglePatch,
  sizePx = 240,
}: {
  imageSrc: string;
  rows: number | null | undefined;
  cols: number | null | undefined;
  positions: number[];
  /** Activation per token index; patches are shaded by value relative to the strongest patch. */
  values?: number[];
  /** Token indexes shown as selected. */
  selected?: number[];
  onTogglePatch?: (tokenIndex: number) => void;
  sizePx?: number;
}) {
  const hasGrid = !!rows && !!cols && rows * cols === positions.length;
  const patchValues = values ? positions.map((p) => values[p] ?? 0) : [];
  const maxAbs = patchValues.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const selectedSet = new Set(selected ?? []);

  return (
    <div className="relative shrink-0 overflow-hidden rounded border border-slate-200" style={{ width: sizePx }}>
      {/* eslint-disable-next-line @next/next/no-img-element -- a data URL from the user's own upload */}
      <img src={imageSrc} alt="Prompt image" className="block h-auto w-full" />
      {hasGrid && (
        <div
          className="absolute inset-0 grid"
          style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, gridTemplateRows: `repeat(${rows}, 1fr)` }}
        >
          {positions.map((tokenIndex, k) => {
            const v = patchValues[k] ?? 0;
            const strength = maxAbs > 0 ? Math.abs(v) / maxAbs : 0;
            const color = v < 0 ? `rgba(220, 38, 38, ${0.75 * strength})` : `rgba(5, 150, 105, ${0.75 * strength})`;
            const isSelected = selectedSet.has(tokenIndex);
            const { row, col } = patchCell(k, cols as number);
            const title = `patch ${row + 1},${col + 1}${values ? `: ${v.toFixed(2)}` : ''}`;
            const cellClass = `border-[0.5px] ${isSelected ? 'border-2 border-emerald-500' : 'border-white/10'}`;
            return onTogglePatch ? (
              <button
                key={tokenIndex}
                type="button"
                title={title}
                aria-pressed={isSelected}
                onClick={() => onTogglePatch(tokenIndex)}
                className={`${cellClass} hover:bg-emerald-200/40`}
                style={values ? { backgroundColor: color } : undefined}
              />
            ) : (
              <div
                key={tokenIndex}
                title={title}
                className={cellClass}
                style={values ? { backgroundColor: color } : undefined}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
