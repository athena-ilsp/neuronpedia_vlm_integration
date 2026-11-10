/**
 * Register a model, one of its source sets, and that set's sources in the local database.
 *
 * The admin panel imports models and sources that Neuronpedia publishes. This is for the ones it
 * does not: a model's raw `neurons` set, or SAEs/transcoders you trained yourself and serve through
 * `apps/inference/neuronpedia_inference/extra_saes.yaml`. It writes only what search and steering
 * need -- no feature dashboards -- and is idempotent, so re-running it with more layers adds them.
 *
 *   ts-node scripts/source-set.ts add --model gemma-3-4b-it --model-layers 34 --instruct \
 *     --set neurons --layers 0-33
 *   ts-node scripts/source-set.ts add --model gemma-3-4b-it --set xai-tc-200m-16x --layers 0-33 \
 *     --description "Transcoders trained with xai_toolkit"
 *
 * Then register the inference host serving it with `make host-add`.
 */
import { PrismaClient, Visibility } from '@prisma/client';
import { NEURONS_SOURCESET, parseLayers, sourceId } from '../lib/utils/source-set-registration';

const prisma = new PrismaClient();

// The seeded bot user (prisma/seed.ts), the same default creator the webapp uses.
const DEFAULT_CREATOR_USER_ID = process.env.DEFAULT_CREATOR_USER_ID || 'clkht01d40000jv08hvalcvly';

const USAGE = `
Usage: source-set add --model <id> --set <name> --layers <spec> [options]

  --model <id>            Neuronpedia model id, e.g. gemma-3-4b-it
  --set <name>            source set name; "neurons" registers the model's raw MLP neurons
  --layers <spec>         layers to register, e.g. 0-33 or 0,5,10
  --description <text>   source set description (or SOURCE_SET_DESCRIPTION in the environment)
  --model-layers <n>      layer count, needed only when the model does not exist yet
  --owner <name>          model owner, when creating the model (default: Unknown)
  --instruct              mark a newly created model as instruction-tuned
  --stubs <count>         also create <count> feature stubs per source (indexes 0..count-1), so
                          feature pages exist; e.g. the SAE width, or the MLP width for neurons
  --remote                allow writing to a database that is not on this machine
`;

function parseArgs(argv: string[]): Map<string, string> {
  const args = new Map<string, string>();
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i].startsWith('--')) {
      const key = argv[i].slice(2);
      const value = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : 'true';
      args.set(key, value);
      if (value !== 'true') i += 1;
    }
  }
  return args;
}

function requireArg(args: Map<string, string>, key: string): string {
  const value = args.get(key);
  if (!value || value === 'true') {
    throw new Error(`Missing --${key}\n${USAGE}`);
  }
  return value;
}

/** Same guard as scripts/compute-host.ts: hand-registered rows are for local development only. */
function assertLocalDatabase(args: Map<string, string>): void {
  const url = process.env.POSTGRES_URL_NON_POOLING ?? process.env.POSTGRES_PRISMA_URL ?? '';
  let host = '';
  try {
    host = new URL(url).hostname;
  } catch {
    host = '';
  }
  if (['localhost', '127.0.0.1', '::1', 'host.docker.internal'].includes(host) || args.has('remote')) {
    return;
  }
  throw new Error(`Refusing to write to "${host || 'an unrecognised host'}", which is not a local database.`);
}

async function add(args: Map<string, string>) {
  assertLocalDatabase(args);
  const modelId = requireArg(args, 'model');
  const setName = requireArg(args, 'set');
  const layers = parseLayers(requireArg(args, 'layers'));

  const existing = await prisma.model.findUnique({ where: { id: modelId }, select: { id: true } });
  if (!existing) {
    const modelLayers = Number(requireArg(args, 'model-layers'));
    await prisma.model.create({
      data: {
        id: modelId,
        displayName: modelId,
        displayNameShort: modelId,
        creatorId: DEFAULT_CREATOR_USER_ID,
        layers: modelLayers,
        neuronsPerLayer: 0,
        owner: args.get('owner') || 'Unknown',
        visibility: Visibility.PUBLIC,
        inferenceEnabled: true,
        instruct: args.has('instruct'),
      },
    });
    console.log(`Created model ${modelId}`);
  }

  const description =
    args.get('description') ||
    process.env.SOURCE_SET_DESCRIPTION ||
    (setName === NEURONS_SOURCESET ? 'MLP neurons' : `${setName} (registered locally)`);
  const stubs = args.get('stubs');
  // With stubs every feature has a page (empty of precomputed activations), which is what the
  // webapp's `hasDashboards` gates on -- the model page lists, and searches, only sets that have it.
  // Without stubs there is nothing to open, so the flag stays off. Only ever turned on here.
  const dashboards = stubs !== undefined ? { hasDashboards: true } : {};
  await prisma.sourceSet.upsert({
    where: { modelId_name: { modelId, name: setName } },
    update: { description, visibility: Visibility.PUBLIC, allowInferenceSearch: true, ...dashboards },
    create: {
      modelId,
      name: setName,
      description,
      type: setName === NEURONS_SOURCESET ? NEURONS_SOURCESET : 'sae',
      creatorName: 'Local',
      creatorId: DEFAULT_CREATOR_USER_ID,
      visibility: Visibility.PUBLIC,
      hasDashboards: stubs !== undefined,
      allowInferenceSearch: true,
    },
  });

  const { count } = await prisma.source.createMany({
    data: layers.map((layer) => ({
      id: sourceId(layer, setName),
      modelId,
      setName,
      creatorId: DEFAULT_CREATOR_USER_ID,
      visibility: Visibility.PUBLIC,
      hasDashboards: stubs !== undefined,
      inferenceEnabled: true,
    })),
    skipDuplicates: true,
  });
  console.log(`Source set ${modelId}/${setName}: ${count} new of ${layers.length} sources`);

  if (stubs !== undefined) {
    await prisma.source.updateMany({
      where: { modelId, setName, id: { in: layers.map((layer) => sourceId(layer, setName)) } },
      data: { hasDashboards: true },
    });
    const perSource = Number(stubs);
    if (!Number.isInteger(perSource) || perSource < 1) {
      throw new Error(`--stubs must be a positive integer, not "${stubs}"`);
    }
    await createStubs(
      modelId,
      layers.map((layer) => sourceId(layer, setName)),
      perSource,
    );
  }
}

/** Empty Neuron rows (no activations or explanations) so each feature has a page to open. */
async function createStubs(modelId: string, sources: string[], perSource: number) {
  const CHUNK = 5000;
  let created = 0;
  for (const layer of sources) {
    for (let start = 0; start < perSource; start += CHUNK) {
      const end = Math.min(start + CHUNK, perSource);
      // eslint-disable-next-line no-await-in-loop -- sequential chunks keep each insert bounded
      const { count } = await prisma.neuron.createMany({
        data: Array.from({ length: end - start }, (_, i) => ({
          modelId,
          layer,
          index: `${start + i}`,
          creatorId: DEFAULT_CREATOR_USER_ID,
        })),
        skipDuplicates: true,
      });
      created += count;
    }
  }
  console.log(`Feature stubs: ${created} new (${perSource} per source x ${sources.length} sources)`);
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  const args = parseArgs(rest);
  if (command !== 'add') {
    throw new Error(USAGE);
  }
  await add(args);
}

if (require.main === module) {
  main()
    .catch((error) => {
      console.error(error instanceof Error ? error.message : error);
      process.exitCode = 1;
    })
    .finally(() => prisma.$disconnect());
}
