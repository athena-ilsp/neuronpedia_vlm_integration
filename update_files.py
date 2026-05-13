import json
import re

sae_base = "/sae_weights/VLM/SAE_L1_image_text_200m_8x/final__leonardo_work_EUHPC_A06_067_ddamianos_hf_cache_hub_models--google--gemma-3-4b-it_snapshots_093f9f388b31de276ce2de164bdc2081324b9767__language_model.model.layers.{}.hook_mlp_out_20480.pt"
tc_base = "/sae_weights/VLM/TC_L1_image_text_200M_16x/final__leonardo_work_EUHPC_A06_067_ddamianos_hf_cache_hub_models--google--gemma-3-4b-it_snapshots_093f9f388b31de276ce2de164bdc2081324b9767__language_model.model.layers.{}.hook_mlp_in_40960.pt"

paths = {}
sources_lines = []
for i in range(34):
    paths[f"{i}-vlm-sae"] = sae_base.format(i)
    paths[f"{i}-vlm-tc"] = tc_base.format(i)
    
    sources_lines.append(f'  "{i}-vlm-sae": 20480,')
    sources_lines.append(f'  "{i}-vlm-tc": 40960,')

paths_json = json.dumps(paths)

# Update env
with open(".env.inference.gemma-3-vlm.layer10", "r") as f:
    env_content = f.read()

env_content = re.sub(r"VLM_SAE_PATHS='\{.*?\}'", f"VLM_SAE_PATHS='{paths_json}'", env_content)

with open(".env.inference.gemma-3-vlm.layer10", "w") as f:
    f.write(env_content)

# Update seed script
with open("apps/webapp/prisma/seed-vlm.ts", "r") as f:
    seed_content = f.read()

new_sources = "const SOURCES: Record<string, number> = {\n" + "\n".join(sources_lines) + "\n};"
seed_content = re.sub(r'const SOURCES: Record<string, number> = \{.*?\};', new_sources, seed_content, flags=re.DOTALL)

with open("apps/webapp/prisma/seed-vlm.ts", "w") as f:
    f.write(seed_content)

print("Updated config files.")

