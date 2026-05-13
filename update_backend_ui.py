import json
import re
import os

print("Updating .env")
env_path = "/home/nxiros/workspace/Neuronpedia/neuronpedia/.env.inference.gemma-3-vlm.layer10"
with open(env_path, "r") as f:
    env_content = f.read()

env_content = env_content.replace('-vlm-sae":', '-SAE-200m-8x":')
env_content = env_content.replace('-vlm-tc":', '-Transcoders-200m-16x":')

with open(env_path, "w") as f:
    f.write(env_content)


print("Updating sae_manager.py")
sae_manager_path = "/home/nxiros/workspace/Neuronpedia/neuronpedia/apps/inference/neuronpedia_inference/sae_manager.py"
with open(sae_manager_path, "r") as f:
    sae_eval = f.read()

old_block = """        if vlm_sae_paths:
            vlm_set_name = os.getenv("VLM_SAE_SET_NAME", "vlm-sae")
            vlm_sae_ids = list(vlm_sae_paths.keys())
            all_sae_ids.extend(vlm_sae_ids)
            self.vlm_sae_paths = vlm_sae_paths
            self.valid_sae_sets.append(vlm_set_name)
            self.sae_set_to_saes[vlm_set_name] = vlm_sae_ids
            logger.info(f"Found {len(vlm_sae_ids)} VLM SAEs to load: {vlm_sae_ids}")"""

new_block = """        if vlm_sae_paths:
            vlm_sae_ids = list(vlm_sae_paths.keys())
            for sae_id in vlm_sae_ids:
                parts = sae_id.split("-", 1)
                current_set = parts[1] if len(parts) == 2 else os.getenv("VLM_SAE_SET_NAME", "vlm-sae")
                
                if current_set not in self.valid_sae_sets:
                    self.valid_sae_sets.append(current_set)
                    self.sae_set_to_saes[current_set] = []
                self.sae_set_to_saes[current_set].append(sae_id)

            all_sae_ids.extend(vlm_sae_ids)
            self.vlm_sae_paths = vlm_sae_paths
            logger.info(f"Found {len(vlm_sae_ids)} VLM SAEs to load: {vlm_sae_ids}")"""

sae_eval = sae_eval.replace(old_block, new_block)
with open(sae_manager_path, "w") as f:
    f.write(sae_eval)


print("Updating seed-vlm.ts")
seed_path = "/home/nxiros/workspace/Neuronpedia/neuronpedia/apps/webapp/prisma/seed-vlm.ts"
with open(seed_path, "r") as f:
    seed_content = f.read()

# Replace SOURCE_SET_NAME var with two arrays
seed_content = re.sub(
    r"const SOURCE_SET_NAME.*?\n",
    "const SET_SAE_NAME = 'SAE-200m-8x';\nconst SET_TC_NAME = 'Transcoders-200m-16x';\n",
    seed_content
)

# Rename the massive SOURCES object into two smaller ones
# Since the script creates them we can just rewrite the SOURCES block.
sources_match = re.search(r"const SOURCES: Record<string, number> = \{.*?\};\n", seed_content, re.DOTALL)
if sources_match:
    new_sources = "const SOURCES_SAE: Record<string, number> = {};\n"
    new_sources += "const SOURCES_TC: Record<string, number> = {};\n"
    new_sources += "for (let i = 0; i <= 33; i++) {\n"
    new_sources += "  SOURCES_SAE[`${i}-SAE-200m-8x`] = 20480;\n"
    new_sources += "  SOURCES_TC[`${i}-Transcoders-200m-16x`] = 40960;\n"
    new_sources += "}\n\n"
    seed_content = seed_content[:sources_match.start()] + new_sources + seed_content[sources_match.end():]

# Now, we need to rewrite the DB seeding parts.
# Under `async function main() {`, we insert a cleanup statement:
#   await prisma.sourceSet.deleteMany({ where: { modelId: MODEL_ID, name: 'vlm-sae' } });
seed_content = re.sub(
    r"(async function main\(\) \{)",
    r"\1\n  console.log('Cleaning up old vlm-sae sourceSet...');\n  await prisma.sourceSet.deleteMany({ where: { modelId: MODEL_ID, name: 'vlm-sae' } });\n",
    seed_content
)

# Upsert source sets
old_source_set_block = """  // VLM change: upsert the source set
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
  console.log('SourceSet:', sourceSet.name);"""

new_source_set_block = """  // Source Sets
  const configurations = [
    { name: SET_SAE_NAME, desc: 'VLM SAE trained on Gemma 3 4B IT', type: 'sae', sources: SOURCES_SAE },
    { name: SET_TC_NAME, desc: 'VLM Transcoder trained on Gemma 3 4B IT', type: 'sae', sources: SOURCES_TC }
  ];

  for (const config of configurations) {
    await prisma.sourceSet.upsert({
      where: { modelId_name: { modelId: MODEL_ID, name: config.name } },
      update: {},
      create: {
        modelId: MODEL_ID,
        name: config.name,
        description: config.desc,
        type: config.type,
        creatorName: 'Local',
        creatorId: ADMIN_USER_ID,
        urls: [],
        visibility: 'UNLISTED',
        hasDashboards: true,
        allowInferenceSearch: true,
      },
    });
    console.log('Upserted SourceSet:', config.name);
  }"""
seed_content = seed_content.replace(old_source_set_block, new_source_set_block)

# Upsert Sources block
old_sources_block = """  // VLM change: upsert each source (one per SAE layer)
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
  }"""
new_sources_block = """  // Upsert Sources
  for (const config of configurations) {
    for (const [sourceId, numFeatures] of Object.entries(config.sources)) {
      await prisma.source.upsert({
        where: { modelId_id: { modelId: MODEL_ID, id: sourceId } },
        update: {},
        create: {
          id: sourceId,
          modelId: MODEL_ID,
          setName: config.name,
          creatorId: ADMIN_USER_ID,
          inferenceEnabled: true,
          hasDashboards: true,
          visibility: 'UNLISTED',
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
  }"""
seed_content = seed_content.replace(old_sources_block, new_sources_block)

# Inference host linkage
seed_content = seed_content.replace("Object.keys(SOURCES)", "Object.keys({...SOURCES_SAE, ...SOURCES_TC})")

# Set Default Source
seed_content = seed_content.replace("defaultSourceSetName: SOURCE_SET_NAME,", "defaultSourceSetName: SET_SAE_NAME,")

with open(seed_path, "w") as f:
    f.write(seed_content)

print("Done")
