# **ModuleFormer**

**ModuleFormer** is a modular architecture that includes two different types of modules, new stick-breaking attention heads, and feedforward experts.
Different modules are sparsely activated conditions on the input token during training and inference.
In our experiment, we found that the modular architecture enables three important abilities for large pre-trained language models:

1) Efficiency, since ModuleFormer only activates a subset of its modules for each input token, thus it could achieve the same performance as dense LLMs with more than two times throughput;
2) Extendability, ModuleFormer is more immune to catastrophic forgetting than dense LLMs and can be easily extended with new modules to learn new knowledge that is not included in the training data;
3) Specialisation, finetuning ModuleFormer could specialize a subset of modules to the finetuning task, and the task-unrelated modules could be easily pruned for a lightweight deployment.

**MoLM** is a collection of ModuleFormer-based language models ranging in scale from 4 billion to 8 billion parameters.

**Model Usage**
To load the models, you need install this package:
```
pip install -e .
```

Then you can load the model with the following code:
```
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from moduleformer import ModuleFormerForCausalLM, ModuleFormerConfig, ModuleFormerForSequenceClassification
AutoConfig.register("moduleformer", ModuleFormerConfig)
AutoModelForCausalLM.register(ModuleFormerConfig, ModuleFormerForCausalLM)
AutoModelForSequenceClassification.register(ModuleFormerConfig, ModuleFormerForSequenceClassification)

tokenizer = AutoTokenizer.from_pretrained('ibm/MoLM-350M-4B')
model = AutoModelForCausalLM.from_pretrained('ibm/MoLM-350M-4B')
```

**Model Details**
MoLM-350M-4B is a MoE-based language model. It has 4 billion parameters, but each input token only activates 350M parameters. Thus, it's computationally equivalent to a 350M dense model. 
MoLM-700M-4B has 4 billion parameters and is computationally equivalent to a 700M dense model. 
MoLM-700M-8B has 8 billion parameters and is computationally equivalent to a 700M dense model. All models are trained on 300 billion tokens from publicly available sources.
All models are trained on 300 billion tokens from publicly available sources, with a learning rate of 3.0 x 10<sup>-4</sup> and a global batch-size of 3M tokens.

**Model Developers** IBM

**Variations** MoLM comes in two different parameter sizes — 4B and 8B. The 4B models has two variants with different computation cost — 350M and 700M.

**Input** Models input text only.

**Output** Models generate text only.

**Model Architecture** MoLM is an auto-regressive language model that uses the ModuleFormer architecture. It has 16 attention modules in each attention layer and 32 MLP modules in each MLP layer. During inference, in each layer, MoLM-350M-4B and MoLM-700M-8B activate 2 modules for each token, while MoLM-700M-4B activate 4 modules. MoLM-350M-4B and MoLM-700M-4B has 24 blocks and MoLM-700M-8B has 48 blocks.

**Status** This is a static model trained on an offline dataset. Future versions of the tuned models will be released as we improve model safety with community feedback.

**Research Paper** ["ModuleFormer: Modularity Emerges from Mixture-of-Experts"](https://arxiv.org/abs/2306.04640)

## Training Data
MoLM models are pretrained on 300 billion tokens of data from publicly available sources.

## Evaluation Results

In this section, we report the results for the MoLM models on standard academic benchmarks. For all the evaluations, we use [LM evaluations Harness](https://github.com/EleutherAI/lm-evaluation-harness).

|Model|Latency|Memory|Throughput|Hellaswag|PIQA|ARC-e|ARC-c|OBQA|
|---|---|---|---|---|---|---|---|---|
||ms|GB|tokens/sec|acc|acc|acc|acc|acc|
|Pythia 410M|554|25|59594|33.72|66.70|51.89|21.42|18.2|
|GPT-Neo 1.3B|991|23|32857|38.66|71.11|56.19|23.12|21.4|
|Pythia 1.4B|918|42|35559|40.41|70.84|60.52|26.11|22.2|
|MoLM-350M-4B|497|27|71017|39.21|70.13|56.44|23.55|20.8|
|GPT-Neo 2.7B|1737|35|18788|42.71|72.2|61.07|27.47|23.2|
|Pythia 2.8B|2111|70|15522|45.34|73.99|64.35|29.35|23.8|
|MoLM-700M-4B|863|27|39931|42.20|73.01|60.82|25.94|22.6|
|MoLM-700M-8B|939|38|37419|43.33|72.91|62.46|27.90|23.8|

|Model| |TriviaQA| | | HumanEval| |Wikitext|
|---|---|---|---|---|---|---|---|
||0-shot |1-shot |5-shot |pass@1 |pass@10 |pass@100 |PPL|
|Pythia 410M |2.32 |5.02 |6.42 |1.20 |3.85 |9.98 |20.09 |
|GPT-Neo 1.3B |5.24 |8.01 |9.74 |3.62 |6.87 |14.50 |16.16 |
|Pythia 1.4B |5.30 |9.87 |12.84 |2.19 |7.31 |14.33 |14.71|
|MoLM-350M-4B |5.40 |11.12 |13.70 |3.04 |6.99 |13.79 |15.15 |
|GPT-Neo 2.7B |4.82 |11.23 |13.67 |4.89 |9.54 |17.90 |13.93 |
|Pythia 2.8B |7.38 |15.58 |18.98 |4.91 |11.76 |21.54 |12.68|
|MoLM-700M-4B|9.07|14.24|16.49|5.50|10.65|20.27|13.20|
|MoLM-700M-8B |11.47 |16.73 |20.75 |5.51 |12.58 |20.40 |12.97 |

## Ethical Considerations and Limitations
MoLM is a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, MoLM’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate, biased or other objectionable responses to user prompts. Therefore, before deploying any applications of MoLM, developers should perform safety testing and tuning tailored to their specific applications of the model.

## MoLM Model Index
|Model|MoLM|
|---|---|
|350M-4B| [Link](https://huggingface.co/ibm/MoLM-350M-4B) |
|700M-4B| [Link](https://huggingface.co/ibm/MoLM-700M-4B) |
|700M-8B| [Link](https://huggingface.co/ibm/MoLM-700M-8B) |