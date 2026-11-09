'use client';

import { ImagePlus, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

/** Matches the inference server's cap on a decoded image (multimodal.MAX_IMAGE_BYTES). */
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

/**
 * Whether the inference host for `modelId` takes image prompts, as its /capabilities reports.
 * False until known, and whenever no host is up, so image controls stay hidden by default.
 */
export function useImageInputSupported(modelId: string): boolean {
  const [supported, setSupported] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setSupported(false);
    fetch('/api/model/image-input', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modelId }),
    })
      .then((response) => (response.ok ? response.json() : undefined))
      .then((json) => {
        if (!cancelled) {
          setSupported(json?.imageInput === true);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [modelId]);
  return supported;
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/**
 * Pick one image for the prompt. `image` is a data URL (what the inference server accepts as
 * `imageBase64`), or undefined when none is picked.
 */
export default function ImageInput({
  image,
  onChange,
  disabled,
}: {
  image: string | undefined;
  onChange: (image: string | undefined) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | undefined>(undefined);

  async function filePicked(file: File | undefined) {
    if (!file) {
      return;
    }
    if (!file.type.startsWith('image/')) {
      setError('Please choose an image file.');
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setError(`Images must be smaller than ${MAX_IMAGE_BYTES / (1024 * 1024)} MB.`);
      return;
    }
    setError(undefined);
    onChange(await readAsDataUrl(file));
  }

  return (
    <div className="flex flex-col items-center gap-y-1">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          filePicked(e.target.files?.[0]);
          e.target.value = '';
        }}
      />
      {image ? (
        <div className="relative h-14 w-14 shrink-0">
          {/* eslint-disable-next-line @next/next/no-img-element -- a data URL from the user's own upload */}
          <img src={image} alt="Prompt image" className="h-14 w-14 rounded-md border border-slate-200 object-cover" />
          <button
            type="button"
            disabled={disabled}
            onClick={() => onChange(undefined)}
            title="Remove image"
            className="absolute -right-1.5 -top-1.5 rounded-full bg-slate-600 p-0.5 text-white hover:bg-slate-800"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      ) : (
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          title="Add an image to the prompt"
          className="flex h-14 w-14 shrink-0 flex-col items-center justify-center gap-y-1 rounded-md border border-dashed border-slate-300 text-slate-500 hover:border-sky-600 hover:text-sky-700 disabled:opacity-50"
        >
          <ImagePlus className="h-5 w-5" />
          <span className="text-[8px] font-medium uppercase leading-none">Image</span>
        </button>
      )}
      {error && <span className="max-w-24 text-center text-[10px] leading-tight text-red-600">{error}</span>}
    </div>
  );
}
