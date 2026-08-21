import torch 
from forgeserve.page_attention.block_manager import BlockManager
from forgeserve.model.paged_runtime import PagedRuntime
from forgeserve.model.types import AttentionImplementation
from forgeserve.sampler.greedy import GreedySampler
from forgeserve.engine.paged_generation import PagedGenerationEngine
from forgeserve.engine.config import GenerationConfig

runtime = PagedRuntime(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    attention=AttentionImplementation.SDPA,
)

block_manager = BlockManager.from_model_config(
    num_blocks=256,
    block_size=16,
    model = runtime.model,
    device="cuda",
)

runtime.attach_block_manager(block_manager)

engine = PagedGenerationEngine(
    runtime = runtime,
    sampler=GreedySampler(),
)

config = GenerationConfig(max_new_tokens=100)
prompt = "Explain how transformers work in detail."

print(f"Before: {block_manager.num_free_blocks} free blocks")

tokenize = runtime.tokenize(prompt)
input_ids = tokenize["input_ids"]
print(f"the length of the tokens are {(input_ids.shape[1])}")

response = engine.generate(prompt=prompt, config=config)

print(f"Free blocks after:{block_manager.num_free_blocks}")
print(f"Generated tokens:{response.generated_tokens}")
print(f"Finish reason:{response.finish_reason}")
print(f"Text preview:{response.text}")

print(f"After: {block_manager.num_free_blocks} free blocks")

assert block_manager.num_free_blocks == 256, \
    f"Block leak! Expected 256, got {block_manager.num_free_blocks}"
print("✅ No block leaks detected")