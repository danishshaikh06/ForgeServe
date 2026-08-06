from forgeserve.model.runtime import Runtime

run = Runtime(model_name="Qwen/Qwen2.5-0.5B-Instruct")
text = "Hello, How are you?"
system_prompt = "Act as a general knowledge model"
tokenizer = run.tokenize(text,system_prompt)
input_ids = tokenizer["input_ids"]
attention_mask = tokenizer["attention_mask"]
output = run.forward(input_ids, attention_mask)

print(input_ids)
print(input_ids.shape)

print(attention_mask)
print(attention_mask.shape)

print('-' * 40)
print(output)
print('-' * 40)
print(output.logits)
print('-' * 40)
print(type(output))

print('-' * 40)
print(output.past_key_values)

