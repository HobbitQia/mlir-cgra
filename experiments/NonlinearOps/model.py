import torch
import torch_mlir

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import CamembertTokenizer, CamembertForTokenClassification, TokenClassificationPipeline
from transformers import CamembertModel, CamembertTokenizer


def prepare_sentence_tokens(hf_model: str, sentence: str):
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer = torch.tensor([tokenizer.encode(sentence)])
    model = AutoModelForTokenClassification.from_pretrained(hf_model).roberta.embeddings.eval()
    output = model(tokenizer)
    return output

import torch.nn.functional as F

class SwiGLU(torch.nn.Module):

    def __init__(self, w1, w2, w3) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, input):
        x1 = F.linear(input, self.w1.weight)
        x2 = F.linear(input, self.w2.weight)
        hidden = F.silu(x1) * x2
        return F.linear(hidden, self.w3.weight)

class GeGLU(torch.nn.Module):
    
    def __init__(self, w1, w2, w3) -> None:
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def forward(self, input):
        x1 = F.linear(input, self.w1.weight)
        x2 = F.linear(input, self.w2.weight)
        hidden = F.gelu(x1) * x2
        return F.linear(hidden, self.w3.weight)

class RMSNorm(torch.nn.Module):

    def __init__(self, shape, eps=1e-05, elementwise_affine=True, dtype=torch.float32) -> None:
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        rms = torch.sqrt(torch.mean(input**2, dim=-1, keepdim=True) + self.eps)
        return (input / rms) * self.weight

class NonlinearOps(torch.nn.Module):
    """Wrapper that returns only the logits from a HuggingFace model."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.GELU()
        self.layer2 = torch.nn.LayerNorm((1,768), eps=1e-05, elementwise_affine=True, dtype=torch.float32)
        torch.nn.init.normal_(self.layer2.weight)
        torch.nn.init.uniform_(self.layer2.bias)
        self.layer3 = SwiGLU(
            torch.nn.Linear(768, 768),
            torch.nn.Linear(768, 768),
            torch.nn.Linear(768, 768)
        )
        self.layer4 = torch.nn.ReLU()
        # self.layer5 = RMSNorm((1,768), eps=1e-05, elementwise_affine=True, dtype=torch.float32)
        # self.layer5.weight = self.layer2.weight


    def forward(self, input):
        # Return only the logits.
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.nn.functional.softmax(x)
        return x


# Suppress warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

example_input = torch.randn((1,768))
print(example_input.shape)

print("Instantiating model.")
model = NonlinearOps()
print(model(example_input).shape)

linalg_on_tensors_mlir = torch_mlir.compile(
    model,
    example_input,
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=True)
file_path = 'bert.txt'
new_path = '02-linalg.mlir'
with open(file_path, 'wt') as f:
    print(linalg_on_tensors_mlir.operation.get_asm(), file=f)
os.rename(file_path,new_path)

result = model.forward(example_input)
print(result)
