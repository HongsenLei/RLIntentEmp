# tensor([[ 0.0258, -0.0082,  0.0198,  0.0138, -0.0009,  0.0169, -0.0045, -0.0212,
#          -0.0019, -0.0098,  0.0203,  0.0114, -0.0061,  0.0098, -0.0158,  0.0008,
#          -0.0235,  0.0016,  0.0019, -0.0067]])
from transformers.models.llama.modeling_llama import LlamaForSequenceClassification

model  = LlamaForSequenceClassification.from_pretrained("/seu_share/home/wutianxing/220222120/experients/vm_lr_5e-6_bz_64_centra/checkpoint-298")

print(model.score.weight.data[:,:20])

# tensor([[ 0.0258, -0.0082,  0.0198,  0.0138, -0.0009,  0.0169, -0.0045, -0.0212,
#          -0.0019, -0.0098,  0.0203,  0.0114, -0.0061,  0.0098, -0.0158,  0.0008,
#          -0.0235,  0.0016,  0.0019, -0.0067]])