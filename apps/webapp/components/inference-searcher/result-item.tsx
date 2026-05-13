import ActivationItem from '@/components/activation-item';
import FeatureStats from '@/components/feature-stats';
import { useGlobalContext } from '@/components/provider/global-provider';
import { InferenceActivationAllResult } from '@/components/provider/inference-activation-all-provider';
import { replaceHtmlAnomalies } from '@/lib/utils/activations';
import { ExplanationPartialWithRelations } from '@/prisma/generated/zod';

export const VLM_PATCH_CELL = 20; // px per patch cell, shared with inference-searcher

// VLM change: 16×16 patch grid overlaid on the image, with activation highlights and optional click-to-select
export function ImagePatchGrid({
  patchValues,
  maxValue,
  imageBase64,
  selectedPatchIndexes,
  firstPatchTokenIndex,
  onPatchClick,
}: {
  patchValues: number[];
  maxValue: number;
  imageBase64: string;
  selectedPatchIndexes?: number[];
  firstPatchTokenIndex?: number;
  onPatchClick?: (tokenIndex: number) => void;
}) {
  const COLS = 16;
  const ROWS = Math.ceil(patchValues.length / COLS);
  const CELL = VLM_PATCH_CELL;
  return (
    <span
      className="relative mx-1 inline-block flex-shrink-0 align-middle"
      style={{ width: COLS * CELL, height: ROWS * CELL }}
    >
      {/* base image */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`data:image/jpeg;base64,${imageBase64}`}
        alt="input"
        className="absolute inset-0 h-full w-full rounded object-fill"
      />
      {/* one cell per patch */}
      {patchValues.map((val, idx) => {
        const col = idx % COLS;
        const row = Math.floor(idx / COLS);
        const strength = maxValue > 0 ? Math.min(val / maxValue, 1) : 0;
        const tokenIdx = firstPatchTokenIndex !== undefined ? firstPatchTokenIndex + idx : idx;
        const isSelected = selectedPatchIndexes?.includes(tokenIdx) ?? false;
        const borderColor = isSelected
          ? 'rgba(14, 165, 233, 0.95)'
          : strength > 0
            ? `rgba(34, 197, 94, ${0.4 + strength * 0.6})`
            : 'rgba(255,255,255,0.18)';
        const borderWidth = isSelected ? 3 : strength > 0 ? Math.max(1, Math.round(strength * 3)) : 1;
        const bgColor = isSelected
          ? 'rgba(14, 165, 233, 0.25)'
          : strength > 0
            ? `rgba(34, 197, 94, ${0.15 + strength * 0.55})`
            : 'transparent';
        // tooltip label — shown above the cell on hover
        const tooltipText = isSelected
          ? `Patch ${idx + 1}${val > 0 ? ` · ${val.toFixed(2)}` : ' · selected'}`
          : val > 0
            ? `Patch ${idx + 1} · ${val.toFixed(2)}`
            : `Patch ${idx + 1}`;
        // position tooltip: flip to bottom for top rows, flip to left for right edge patches
        const tooltipTop = row < 3;
        const tooltipRight = col >= COLS - 4;
        return (
          <span
            key={idx}
            className="group/patch"
            onClick={onPatchClick ? (e) => { e.preventDefault(); onPatchClick(tokenIdx); } : undefined}
            style={{
              position: 'absolute',
              left: col * CELL,
              top: row * CELL,
              width: CELL,
              height: CELL,
              boxSizing: 'border-box',
              border: `${borderWidth}px solid ${borderColor}`,
              backgroundColor: bgColor !== 'transparent' ? bgColor : undefined,
              cursor: onPatchClick ? 'pointer' : 'default',
              zIndex: 1,
            }}
          >
            {/* hover tooltip */}
            <span
              style={{
                position: 'absolute',
                [tooltipTop ? 'top' : 'bottom']: '100%',
                [tooltipRight ? 'right' : 'left']: 0,
                marginTop: tooltipTop ? 2 : undefined,
                marginBottom: tooltipTop ? undefined : 2,
                whiteSpace: 'nowrap',
                pointerEvents: 'none',
                zIndex: 50,
              }}
              className="hidden rounded bg-slate-800 px-1.5 py-0.5 text-[11px] font-semibold text-white shadow-lg group-hover/patch:block"
            >
              {tooltipText}
            </span>
          </span>
        );
      })}
    </span>
  );
}

export default function ResultItem({
  result,
  tokens,
  topExplanation,
  searchSortIndexes,
  showDashboards,
  searchImageBase64,
  onPatchClick,
}: {
  result: InferenceActivationAllResult;
  tokens: string[];
  topExplanation: ExplanationPartialWithRelations | undefined;
  searchSortIndexes: number[];
  showDashboards: boolean;
  searchImageBase64?: string;
  onPatchClick?: (tokenIndex: number) => void;
}) {
  const { getSourceSet } = useGlobalContext();

  // VLM change: build display tokens and patch info
  const firstPatchIdx = tokens.findIndex((t) => t === '<image_soft_token>');
  const hasImageTokens = firstPatchIdx !== -1 && !!searchImageBase64;
  // For non-image tokens, strip out image patch tokens and remap values
  const textOnlyTokens = hasImageTokens ? tokens.filter((t) => t !== '<image_soft_token>') : tokens;
  const textOnlyValues = hasImageTokens
    ? result.values.filter((_, i) => tokens[i] !== '<image_soft_token>')
    : result.values;
  const patchValues = hasImageTokens ? tokens.map((t, i) => (t === '<image_soft_token>' ? result.values[i] : 0)).filter((_, i) => tokens[i] === '<image_soft_token>') : [];
  const patchMaxValue = patchValues.length > 0 ? Math.max(...patchValues) : 0;
  // Remap maxValueIndex for text-only display
  let displayMaxValueIndex = result.maxValueIndex;
  if (hasImageTokens) {
    const isMaxInPatch = tokens[result.maxValueIndex] === '<image_soft_token>';
    if (isMaxInPatch) {
      displayMaxValueIndex = 0;
    } else {
      displayMaxValueIndex = tokens.slice(0, result.maxValueIndex).filter((t) => t !== '<image_soft_token>').length;
    }
  }
  const displayTokens = hasImageTokens ? textOnlyTokens : tokens;
  const displayValues = hasImageTokens ? textOnlyValues : result.values;
  return (
    <a
      href={`/${result.modelId}/${result.layer}/${result.index}`}
      target="_blank"
      rel="noreferrer"
      key={result.index}
      className="group flex w-full cursor-pointer flex-col items-center justify-center border-b border-slate-100 bg-white px-3 py-5 sm:px-5"
    >
      <div className="flex w-full max-w-screen-lg flex-col items-start justify-center gap-x-2 sm:flex-row sm:gap-x-5">
        <div className="flex basis-4/12 flex-col items-center justify-start font-mono text-[11px] font-medium text-slate-500 sm:basis-3/12">
          <span
            className={`mb-1 mt-0 flex flex-col font-sans leading-tight ${
              topExplanation ? 'text-[12px] text-slate-700 sm:text-sm' : 'text-[12px] text-slate-500 sm:text-xs'
            }`}
          >
            {topExplanation ? (
              topExplanation.description
            ) : (
              <div className="flex flex-col items-center justify-center gap-y-0.5 text-center">
                <div>No Explanation</div>
              </div>
            )}
          </span>
          <span className="mb-2.5 rounded-md px-2 py-1 text-[10px] font-bold uppercase text-slate-600 group-hover:bg-sky-200 sm:mb-1 sm:text-xs">
            {result.layer}:{result.index}
          </span>
        </div>
        <div className="flex basis-8/12 flex-col sm:basis-9/12">
          <div className={`mt-0 flex ${searchSortIndexes.length > 0 ? 'flex-col' : 'flex-row items-center'} gap-x-2`}>
            {searchSortIndexes.length > 0 ? (
              <div className="mb-1.5 flex max-w-screen-md flex-row gap-x-1 overflow-x-scroll">
                {searchSortIndexes.length > 1 && (
                  <div className="sticky left-0 flex flex-col items-center justify-center gap-y-0.5 bg-white pr-1">
                    <div className="px-1 text-[10px] font-bold uppercase text-slate-400">Sum</div>
                    <span className="mt-0 text-[11px] text-emerald-700">
                      {searchSortIndexes
                        .map((index) => result.values[index])
                        .reduce((a, b) => a + b, 0)
                        .toFixed(1)}
                    </span>
                  </div>
                )}
                {searchSortIndexes.map((searchSortIndex) => (
                  <div key={searchSortIndex} className="flex flex-col items-center justify-center gap-y-0.5">
                    <div className="whitespace-pre rounded bg-slate-200 px-1 font-mono text-xs font-medium">
                      {replaceHtmlAnomalies(tokens[searchSortIndex])}
                    </div>
                    <span className="mt-0 text-[11px] text-emerald-700">
                      {result.values[searchSortIndex].toFixed(1)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center gap-y-0.5 px-2 text-xs text-slate-700">
                <div className="whitespace-pre rounded bg-slate-200 px-1 font-mono font-medium">
                  {hasImageTokens && tokens[result.maxValueIndex] === '<image_soft_token>'
                    ? `Patch ${result.maxValueIndex - tokens.findIndex((t) => t === '<image_soft_token>') + 1}`
                    : replaceHtmlAnomalies(displayTokens[displayMaxValueIndex])}
                </div>
                <span className="mt-0 text-[11px] text-emerald-700">{result.maxValue.toFixed(1)}</span>
              </div>
            )}
            <div className="flex flex-row flex-wrap items-center gap-x-1">
              {displayTokens.length > 0 && (
                <ActivationItem
                  showLineBreaks={false}
                  activation={{
                    values: displayValues,
                    tokens: displayTokens,
                    maxValueTokenIndex: displayMaxValueIndex,
                    maxValue: result.maxValue,
                    dfaValues: result.dfaValues,
                    dfaMaxValue: result.dfaMaxValue,
                    dfaTargetIndex: result.dfaTargetIndex,
                  }}
                  enableExpanding={false}
                  overrideLeading="leading-none"
                  overrideTextSize="text-xs"
                  dfa={getSourceSet(result.neuron?.modelId || '', result.neuron?.sourceSetName || '')?.showDfa}
                />
              )}
              {hasImageTokens && (
                <ImagePatchGrid
                  patchValues={patchValues}
                  maxValue={patchMaxValue}
                  imageBase64={searchImageBase64!}
                  selectedPatchIndexes={searchSortIndexes}
                  firstPatchTokenIndex={firstPatchIdx}
                  onPatchClick={onPatchClick}
                />
              )}
            </div>
          </div>

          {showDashboards && result.neuron && (
            <div className="mt-3 flex flex-col rounded-md border border-slate-200 px-3 pb-3">
              {result.neuron && result.neuron.activations && result.neuron.activations[0] && (
                <>
                  <div className="mb-0.5 mt-3 text-left text-[12px] font-bold uppercase text-slate-600">
                    Top Activation
                  </div>
                  <div className="mt-1 flex flex-row gap-x-2">
                    {result.neuron.activations &&
                      result.neuron.activations.length > 0 &&
                      result.neuron.activations[0].tokens &&
                      result.neuron.activations[0].values && (
                        <div className="flex flex-col items-center gap-y-0.5 px-2 text-xs text-slate-700">
                          <span className="whitespace-pre rounded bg-slate-200 px-1 font-mono font-medium">
                            {replaceHtmlAnomalies(
                              result.neuron.activations[0].tokens[result.neuron.activations[0].maxValueTokenIndex || 0],
                            )}
                          </span>
                          <span className="mt-0 text-[11px] text-emerald-700">
                            {result.neuron.activations[0].values[
                              result.neuron.activations[0].maxValueTokenIndex || 0
                            ].toFixed(1)}
                          </span>
                        </div>
                      )}
                    <ActivationItem
                      showLineBreaks={false}
                      activation={{
                        values: result.neuron.activations[0].values,
                        tokens: result.neuron.activations[0].tokens,
                        maxValueTokenIndex: result.neuron.activations[0].maxValueTokenIndex,
                        maxValue: result.neuron.activations[0].maxValue,
                        dfaValues: result.neuron.activations[0].dfaValues,
                        dfaMaxValue: result.neuron.activations[0].dfaMaxValue,
                        dfaTargetIndex: result.neuron.activations[0].dfaTargetIndex,
                      }}
                      enableExpanding={false}
                      tokensToDisplayAroundMaxActToken={10}
                      overrideLeading="leading-none"
                      overrideTextSize="text-xs"
                      dfa={getSourceSet(result.neuron?.modelId, result.neuron?.sourceSetName || '')?.showDfa}
                    />
                  </div>
                </>
              )}

              {result.neuron?.pos_str && result.neuron?.pos_str.length > 0 && (
                <div className="pointer-events-none mb-0 mt-2 pt-2">
                  <FeatureStats currentNeuron={result.neuron} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </a>
  );
}
