from forgeserve.model.runtime import Runtime

run = Runtime(model_name="Qwen/Qwen2.5-0.5B-Instruct")
text = "Hello, How are you?"

tokenizer = run.tokenize(text)
input_ids = tokenizer["input_ids"]
attention_mask = tokenizer["attention_mask"]
output = run.forward(input_ids, attention_mask)

print(input_ids)
print(input_ids.shape)

print(attention_mask)
print(attention_mask.shape)

print(output.logits)
