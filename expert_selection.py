import json
import torch
from tqdm import tqdm

w, nw = torch.load("/export/ruc/dongzican/experts/expert_info/pt_math.pt")


topk_experts = torch.topk(w, 128, dim=-1)[1]
mask = torch.zeros_like(w).float()
print(mask.shape)
print(topk_experts.shape)
mask.scatter_(1, topk_experts, 1)
with open("/export/ruc/dongzican/experts/expert_sets/pt_math.json",'w') as fw:
    import json
    fw.write(json.dumps(mask.float().tolist()))

# torch.save((torch.stack(expert_frequency_list,dim=0), torch.stack(expert_weight_list, dim=0)),'math_expert_counting_and_probability_new.pt')
