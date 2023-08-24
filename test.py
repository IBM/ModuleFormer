import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from obsidian import SparseGPTForCausalLM, SparseGPTConfig, SparseGPTForSequenceClassification
AutoConfig.register("sparsegpt", SparseGPTConfig)
AutoModelForCausalLM.register(SparseGPTConfig, SparseGPTForCausalLM)
AutoModelForSequenceClassification.register(SparseGPTConfig, SparseGPTForSequenceClassification)

model_path = "/dccstor/codeai/yikang/pretrained_models/obsidian-8b-dolly"

model = AutoModelForSequenceClassification.from_pretrained(model_path)

x = torch.randint(low=10, high=101, size=(5, 7))

# 选择模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 输入文本
text = "The quick brown fox jumps over the lazy dog"

# 对文本进行 tokenization 和 padding
input_ids = tokenizer.encode(text, return_tensors="pt")

y = model(input_ids)

print(y)

# print(input_ids.shape)
# for i, o in enumerate(y):
#     print(i, type(o), (o.shape if 'ensor' in str(type(o)) else o))

# logits = y.logits
# print(logits.shape)

# prob = logits.softmax(dim = -1)

# for i in range(1, input_ids.shape[1]):
#     print(input_ids[0,i], prob[0,i-1,input_ids[0,i]])

