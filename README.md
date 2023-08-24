# ModuleFormer

We propose a new neural network architecture, ModuleFormer, that leverages modularity to improve the efficiency and flexibility of large language models.

## Usage

To install the package:

```
pip install -e .
```

To load the model from `path_to_ckpt`:

```
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from moduleformer import ModuleFormerForCausalLM, ModuleFormerConfig, ModuleFormerForSequenceClassification
AutoConfig.register("sparsegpt", ModuleFormerConfig)
AutoModelForCausalLM.register(ModuleFormerConfig, ModuleFormerForCausalLM)
AutoModelForSequenceClassification.register(ModuleFormerConfig, ModuleFormerForSequenceClassification)

tokenizer = AutoTokenizer.from_pretrained(path_to_ckpt)
model = AutoModelForCausalLM.from_pretrained(path_to_ckpt)
```
