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
const SET_SAE_NAME = 'SAE-200m-8x';
const SET_TC_NAME = 'Transcoders-200m-16x';
const SET_TC_300M_NAME = 'Transcoders-300m-16x';

// VLM change: one entry per SAE layer you have weights for.
// key = source ID as it appears in the URL (e.g. "10-vlm-sae" → localhost:3000/gemma-3-4b-it/10-vlm-sae/0)
// value = number of features (neurons) in that SAE
const SOURCES_SAE: Record<string, number> = {};
const SOURCES_TC: Record<string, number> = {};
const SOURCES_TC_300M: Record<string, number> = {};
for (let i = 0; i <= 33; i++) {
  SOURCES_SAE[`${i}-SAE-200m-8x`] = 20480;
  SOURCES_TC[`${i}-Transcoders-200m-16x`] = 40960;
  SOURCES_TC_300M[`${i}-Transcoders-300m-16x`] = 40960;
}


// VLM change: URL of the local inference server (default for localhost dev)
const INFERENCE_HOST_URL = 'http://localhost:5002';

// bot user id from seed.ts
const ADMIN_USER_ID = 'clkht01d40000jv08hvalcvly';

async function main() {
  console.log('Cleaning up old source sets...');
  await prisma.source.deleteMany({ where: { modelId: MODEL_ID, setName: 'vlm-sae' } });
  await prisma.sourceSet.deleteMany({ where: { modelId: MODEL_ID, name: 'vlm-sae' } });
  await prisma.source.deleteMany({ where: { modelId: MODEL_ID, setName: SET_TC_300M_NAME } });
  await prisma.sourceSet.deleteMany({ where: { modelId: MODEL_ID, name: SET_TC_300M_NAME } });

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

  // Source Sets
  const configurations = [
    { name: SET_SAE_NAME, desc: 'VLM SAE trained on Gemma 3 4B IT', type: 'sae', sources: SOURCES_SAE },
    { name: SET_TC_NAME, desc: 'VLM Transcoder trained on Gemma 3 4B IT', type: 'sae', sources: SOURCES_TC },
    { name: SET_TC_300M_NAME, desc: 'VLM Transcoder 300M trained on Gemma 3 4B IT', type: 'sae', sources: SOURCES_TC_300M },
  ];

  for (const config of configurations) {
    await prisma.sourceSet.upsert({
      where: { modelId_name: { modelId: MODEL_ID, name: config.name } },
      update: { visibility: 'PUBLIC' },
      create: {
        modelId: MODEL_ID,
        name: config.name,
        description: config.desc,
        type: config.type,
        creatorName: 'Local',
        creatorId: ADMIN_USER_ID,
        urls: [],
        visibility: 'PUBLIC',
        hasDashboards: true,
        allowInferenceSearch: true,
      },
    });
    console.log('Upserted SourceSet:', config.name);
  }

  // Upsert Sources
  for (const config of configurations) {
    for (const [sourceId, numFeatures] of Object.entries(config.sources)) {
      await prisma.source.upsert({
        where: { modelId_id: { modelId: MODEL_ID, id: sourceId } },
        update: { visibility: 'PUBLIC' },
        create: {
          id: sourceId,
          modelId: MODEL_ID,
          setName: config.name,
          creatorId: ADMIN_USER_ID,
          inferenceEnabled: true,
          hasDashboards: true,
          visibility: 'PUBLIC',
        },
      });
      
      // Feature stubs via Postgres raw insert query to bypass 32k query limit overhead on massive lists!
      // (Prisma createMany falls over on >30k records, we do it in batches or simple loop if needed. Here skipDuplicates is ok because we deleted vlm-sae above.)
      await prisma.neuron.createMany({
        data: Array.from({ length: numFeatures }, (_, i) => ({
          modelId: MODEL_ID,
          layer: sourceId,
          index: String(i),
          maxActApprox: 1.0,
          creatorId: ADMIN_USER_ID,
        })),
        skipDuplicates: true,
      });
      console.log(`Created ${numFeatures} neuron stubs for ${sourceId}`);
    }
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
  for (const sourceId of Object.keys({...SOURCES_SAE, ...SOURCES_TC, ...SOURCES_TC_300M})) {
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
  const firstSourceId = Object.keys({...SOURCES_SAE, ...SOURCES_TC})[0];
  if (firstSourceId) {
    await prisma.model.update({
      where: { id: MODEL_ID },
      data: {
        defaultSourceSetName: SET_SAE_NAME,
        defaultSourceId: firstSourceId,
        defaultGraphSourceSetName: SET_TC_NAME,
      },
    });
    console.log('Set default source:', firstSourceId);
  }

  // VLM change: enable graph generation on the TC source set
  await prisma.sourceSet.update({
    where: { name_modelId: { name: SET_TC_NAME, modelId: MODEL_ID } },
    data: { graphEnabled: true },
  });
  console.log('Enabled graphEnabled on sourceSet:', SET_TC_NAME);

  // VLM change: register local graph server so /gemma-3-4b-it/graph can generate circuits
  const graphHostSource = await prisma.graphHostSource.upsert({
    where: { id: 'local-gemma3-4b-it-graph' },
    update: { hostUrl: 'http://graph:5004' },
    create: {
      id: 'local-gemma3-4b-it-graph',
      name: 'Local Gemma3-4B-IT Graph Server',
      hostUrl: 'http://graph:5004',
      modelId: MODEL_ID,
    },
  });
  console.log('GraphHostSource:', graphHostSource.id);

  for (const tcSetName of [SET_TC_NAME, SET_TC_300M_NAME]) {
    await prisma.graphHostSourceOnSourceSet.upsert({
      where: {
        sourceSetName_sourceSetModelId_graphHostSourceId: {
          sourceSetName: tcSetName,
          sourceSetModelId: MODEL_ID,
          graphHostSourceId: graphHostSource.id,
        },
      },
      update: {},
      create: {
        sourceSetName: tcSetName,
        sourceSetModelId: MODEL_ID,
        graphHostSourceId: graphHostSource.id,
      },
    });
    console.log('GraphHostSourceOnSourceSet linked:', tcSetName, '→', graphHostSource.id);
  }

  console.log('\nDone! Visit: http://localhost:3000/gemma-3-4b-it/' + Object.keys({...SOURCES_SAE, ...SOURCES_TC})[0] + '/0');
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
