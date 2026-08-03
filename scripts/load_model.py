from forgeserve.model.loader import ModelLoader

loader = ModelLoader(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device='cuda'
    )

model , tokenizer = loader.load()

print(model.__class__.__name__)
print(tokenizer.__class__.__name__)