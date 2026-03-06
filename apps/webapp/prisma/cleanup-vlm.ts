/**
 * VLM change: cleanup script to remove stale layer_10 stubs and re-seed with layer_11
 * Run with: POSTGRES_PRISMA_URL="..." npx tsx prisma/cleanup-vlm.ts
 */
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function main() {
  // Delete stale layer_10 and layer_11 neurons
  const deleted = await prisma.neuron.deleteMany({
    where: { modelId: 'gemma-3-4b-it', layer: { in: ['layer_10', 'layer_11'] } },
  });
  console.log(`Deleted ${deleted.count} stale neuron stubs`);

  // Delete stale sources
  const deletedSource = await prisma.source.deleteMany({
    where: { modelId: 'gemma-3-4b-it', id: { in: ['layer_10', 'layer_11'] } },
  });
  console.log(`Deleted ${deletedSource.count} stale sources`);

  // Clear default source if it was layer_10
  await prisma.model.updateMany({
    where: { id: 'gemma-3-4b-it' },
    data: { defaultSourceId: null, defaultSourceSetName: null },
  });
  console.log('Cleared default source on model (if model existed)');
}

main()
  .then(async () => { await prisma.$disconnect(); })
  .catch(async (e) => { console.error(e); await prisma.$disconnect(); process.exit(1); });