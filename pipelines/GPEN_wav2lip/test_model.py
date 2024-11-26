import torch
import pprint

def print_model_structure(model_path):
    # 加载模型
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 如果 state_dict 是一个字典，并且包含 'g' 键
    if isinstance(state_dict, dict) and 'g' in state_dict:
        print("模型包含 'g' 键，其内容为：")
        pprint.pprint(state_dict['g'])
    else:
        print("模型结构：")
        pprint.pprint(state_dict)
    
    # # 打印键的数量
    # print(f"\n模型中的键的总数：{len(state_dict)}")
    
    # # 打印前几个键值对的形状
    # print("\n前几个键值对的形状：")
    # for i, (key, value) in enumerate(state_dict.items()):
    #     if i < 5:  # 只打印前5个
    #         print(f"{key}: {value.shape}")
    #     else:
    #         print("...")
    #         break

# 使用示例
model_path = "./training-run/202409011321/checkpoints/checkpoint_150000.pt"
print_model_structure(model_path)