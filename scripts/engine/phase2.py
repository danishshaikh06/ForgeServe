from forgeserve.engine.config import GenerationConfig
from forgeserve.engine.generation import GenerationEngine
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler

runtime = Runtime("Qwen/Qwen2.5-0.5B-Instruct")
prompt = "Hello! how are you?"

greddy = GreedySampler()

engine = GenerationEngine(runtime, greddy)

config = GenerationConfig(
    max_new_tokens = 256,
    system_prompt= "You are an ai inference model created by danish "
)

token_generation = engine.generate(prompt, config)

print(token_generation)
