from forgeserve.engine.generation import GenerationEngine
from forgeserve.model.runtime import Runtime
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.sampler.base import Sampler

runtime = Runtime("Qwen/Qwen2.5-0.5B-Instruct")
prompt = 'Hello! how are you?'

greddy = GreedySampler()

engine = GenerationEngine(runtime,greddy)

token_generation = engine.generate(prompt)

print(token_generation)