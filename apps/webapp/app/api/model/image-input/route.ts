import { badRequest } from '@/lib/api-error';
import { assertUserCanAccessModel } from '@/lib/db/userCanAccess';
import { getInferenceImageInputSupported } from '@/lib/utils/inference';
import { RequestOptionalUser, withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
 * Whether searches on `modelId` can take an image: the inference host serving it reports
 * `image_input` in its /capabilities (the eager backend on a checkpoint with a vision tower).
 * The UI shows its image controls on this, so it reflects what the running host can do rather
 * than a list of model names. `{ imageInput: false }` when no host is up.
 */
export const POST = withOptionalUser(async (request: RequestOptionalUser) => {
  const body = await request.json().catch(() => undefined);
  const modelId = body?.modelId;
  if (typeof modelId !== 'string' || modelId === '') {
    throw badRequest('Missing modelId.');
  }
  await assertUserCanAccessModel(modelId, request.user);
  const imageInput = await getInferenceImageInputSupported(modelId, request.user);
  return NextResponse.json({ imageInput });
});
