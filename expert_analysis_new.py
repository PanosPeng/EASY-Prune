import json
import torch
from tqdm import tqdm
import math
import random
random.seed(42)
with open("/export/ruc/DeepSeek-V3/inference/pt_math.jsonl",'r') as fp:
    lines  = fp.readlines()

    data = ([line for line in lines])
    random.shuffle(data)
data = sorted(data, key=len, reverse=True)
data = [json.loads(d) for d in data[:25]]
print(len(data[0]), )
expert_frequency_list = []
expert_br_list = []
expert_norm_list = []
expert_sf_list = []
for sample in tqdm(data):
    print(len(sample['simibr']), len(sample['simibr'][0]))
    expert_frequency = torch.zeros((58, 256)).int()
    expert_br = torch.zeros((58, 256)).float()
    expert_norm = torch.zeros((58, 256)).float()
    expert_sf = torch.zeros((58, 256)).float()
    for layer in range(58):
        for token in range(len(sample['idxs'][0])):
            idxs = sample['idxs'][layer][token]
            weight = sample['weights'][layer][token]
            norm = sample['norms'][layer][token]
            simibr = (max(1 - sample['simibr'][layer][0][token],0))
            # print(simibr)
            # print(len(sample['simibr']), len(sample['simibr'][0]), print(len(simibr)))
            simisf = (max(1 - sample['simisf'][layer][0][token],0))
            # print(simibr.shape)
        
            for i, idx in enumerate(idxs):
                # expert_frequency[layer, idx]+=1
                expert_br[layer, idx]+=weight[i]* simibr*norm[i]
                # expert_norm[layer, idx]+=norm[i]
                expert_sf[layer, idx]+=norm[i]*weight[i]* simisf
    # expert_frequency_list.append(expert_frequency)
    expert_br_list.append(expert_br)
    # expert_norm_list.append(expert_norm)
    expert_sf_list.append(expert_sf)

# frequency_info = torch.sum(torch.stack(expert_frequency_list, dim=0), dim=0)
weight_info = torch.sum(torch.stack(expert_br_list,dim=0),dim=0)
# fw_info = weight_info/frequency_info
# norm_info =torch.sum(torch.stack(expert_norm_list,dim=0),dim=0)
expert_sf_info =torch.sum(torch.stack(expert_sf_list,dim=0),dim=0)

torch.save((weight_info,expert_sf_info),'/export/ruc/dongzican/experts/expert_info/pt_math.pt')
# topk_experts = torch.topk(expert_info,k=128, dim=-1)[1]
# mask = torch.zeros_like(expert_info)
# print(mask.shape)
# print(topk_experts.shape)
# mask.scatter_(1, topk_experts, 1)
# with open("cc_experts_100.json",'w') as fw:
#     import json
#     fw.write(json.dumps(mask.tolist()))

# torch.save((torch.stack(expert_frequency_list,dim=0), torch.stack(expert_br_list, dim=0)),'math_expert_counting_and_probability_new.pt')
