/**
 * VLM change: seed script to register gemma-3-4b-it model + VLM SAE sources in the DB.
 *
 * Run with:
 *   cd apps/webapp && npx ts-node --compiler-options '{"module":"CommonJS"}' prisma/seed-vlm.ts
 * or:
 *   cd apps/webapp && npx tsx prisma/seed-vlm.ts
 */
import { InferenceEngine, PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// VLM change: configure these to match your setup
const MODEL_ID = 'gemma-3-4b-it'; // appears in URLs: localhost:3000/gemma-3-4b-it/...
const SOURCE_SET_NAME = 'vlm-sae'; // SAE set name, appears in URL after model: .../vlm-sae/...

// VLM change: one entry per SAE layer you have weights for.
// key = source ID as it appears in the URL (e.g. "10-vlm-sae" → localhost:3000/gemma-3-4b-it/10-vlm-sae/0)
// value = number of features (neurons) in that SAE
const SOURCES: Record<string, number> = {
  "0-vlm-sae": 20480,
  "1-vlm-sae": 20480,
  "2-vlm-sae": 20480,
  "10-vlm-sae": 20480,
  "11-vlm-sae": 20480,
  "12-vlm-sae": 20480,
  "13-vlm-sae": 20480,
  "15-vlm-sae": 20480,
  "16-vlm-sae": 20480,
  "17-vlm-sae": 20480,
  "18-vlm-sae": 20480,
  "19-vlm-sae": 20480,
  "20-vlm-sae": 20480,
  "21-vlm-sae": 20480,
  "22-vlm-sae": 20480,
  "24-vlm-sae": 20480,
  "25-vlm-sae": 20480,
  "26-vlm-sae": 20480,
  "27-vlm-sae": 20480,
};

// VLM change: URL of the local inference server (default for localhost dev)
const INFERENCE_HOST_URL = 'http://localhost:5002';

// bot user id from seed.ts
const ADMIN_USER_ID = 'clkht01d40000jv08hvalcvly';

async function main() {
  // VLM change: upsert the model
  const model = await prisma.model.upsert({
    where: { id: MODEL_ID },
    update: {},
    create: {
      id: MODEL_ID,
      displayNameShort: 'Gemma 3 4B IT',
      displayName: 'Gemma 3 4B Instruct (VLM)',
      creatorId: ADMIN_USER_ID,
      tlensId: 'gemma-3-4b-it',
      layers: 46, // VLM change: gemma-3-4b has 46 transformer layers
      neuronsPerLayer: 2048, // VLM change: gemma-3-4b hidden size
      owner: 'Google',
      website: 'https://huggingface.co/google/gemma-3-4b-it',
      inferenceEnabled: true,
      instruct: true,
      visibility: 'PUBLIC',
    },
  });
  console.log('Model:', model.id);

  // VLM change: upsert the source set
  const sourceSet = await prisma.sourceSet.upsert({
    where: { modelId_name: { modelId: MODEL_ID, name: SOURCE_SET_NAME } },
    update: {},
    create: {
      modelId: MODEL_ID,
      name: SOURCE_SET_NAME,
      description: 'VLM SAE trained on Gemma 3 4B IT',
      type: 'sae',
      creatorName: 'Local',
      creatorId: ADMIN_USER_ID,
      urls: [],
      visibility: 'UNLISTED',
      hasDashboards: true,
      allowInferenceSearch: true,
    },
  });
  console.log('SourceSet:', sourceSet.name);

  // VLM change: upsert each source (one per SAE layer)
  for (const [sourceId, numFeatures] of Object.entries(SOURCES)) {
    const source = await prisma.source.upsert({
      where: { modelId_id: { modelId: MODEL_ID, id: sourceId } },
      update: {},
      create: {
        id: sourceId,
        modelId: MODEL_ID,
        setName: SOURCE_SET_NAME,
        creatorId: ADMIN_USER_ID,
        inferenceEnabled: true,
        hasDashboards: true,
        visibility: 'UNLISTED',
      },
    });
    console.log('Source:', source.id, '— features:', numFeatures);

    // VLM change: create neurons (features) for this source so feature pages work
    // This creates stubs — activations are computed live by the inference server
    await prisma.neuron.createMany({
      data: Array.from({ length: numFeatures }, (_, i) => ({
        modelId: MODEL_ID,
        layer: sourceId,
        index: String(i),
        maxActApprox: 1.0, // VLM change: must be > 0 for browser to show features
        creatorId: ADMIN_USER_ID,
      })),
      skipDuplicates: true,
    });
    console.log(`Created ${numFeatures} neuron stubs for ${sourceId}`);
  }

  // VLM change: upsert inference host pointing at local inference server
  const inferenceHost = await prisma.inferenceHostSource.upsert({
    where: { id: `vlm-local-${MODEL_ID}` },
    update: { hostUrl: INFERENCE_HOST_URL },
    create: {
      id: `vlm-local-${MODEL_ID}`,
      name: 'VLM Local Inference',
      hostUrl: INFERENCE_HOST_URL,
      engine: InferenceEngine.TRANSFORMER_LENS, // VLM change: our adapter speaks the same API
      modelId: MODEL_ID,
    },
  });
  console.log('InferenceHost:', inferenceHost.id, '@', inferenceHost.hostUrl);

  // VLM change: link each source to the inference host
  for (const sourceId of Object.keys(SOURCES)) {
    await prisma.inferenceHostSourceOnSource.upsert({
      where: {
        sourceId_sourceModelId_inferenceHostId: {
          sourceId,
          sourceModelId: MODEL_ID,
          inferenceHostId: inferenceHost.id,
        },
      },
      update: {},
      create: {
        sourceId,
        sourceModelId: MODEL_ID,
        inferenceHostId: inferenceHost.id,
      },
    });
    console.log(`Linked source ${sourceId} to inference host`);
  }

  // VLM change: set the default source so /gemma-3-4b-it redirects to a source automatically
  const firstSourceId = Object.keys(SOURCES)[0];
  if (firstSourceId) {
    await prisma.model.update({
      where: { id: MODEL_ID },
      data: {
        defaultSourceSetName: SOURCE_SET_NAME,
        defaultSourceId: firstSourceId,
      },
    });
    console.log('Set default source:', firstSourceId);
  }

  console.log('\nDone! Visit: http://localhost:3000/gemma-3-4b-it/' + Object.keys(SOURCES)[0] + '/0');
  console.log('NOTE: if layer_10 stubs exist from a previous run, delete them:');
  console.log("  DELETE FROM \"Neuron\" WHERE \"modelId\" = 'gemma-3-4b-it' AND layer = 'layer_10';");
}

main()
  .then(async () => {
    await prisma.$disconnect();
  })
  .catch(async (e) => {
    console.error(e);
    await prisma.$disconnect();
    process.exit(1);
  });
