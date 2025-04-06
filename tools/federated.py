import os.path

import torch
import torch.nn as nn

# 要平均的模型权重路径列表
model_paths = ['CREStereoDataset/NMRF/sceneflow-pretrained/ckpt/epoch_2/pytorch_model.bin',
               'DynamicReplicaDataset/NMRF/sceneflow-pretrained/ckpt/epoch_3/pytorch_model.bin',
               'FallingThingsDataset/NMRF/sceneflow-pretrained/ckpt/epoch_9/pytorch_model.bin',  # 2
               'TartanAirDataset/NMRF/sceneflow-pretrained/ckpt/epoch_1/pytorch_model.bin',
               'VirtualKitti2Dataset/NMRF/sceneflow-pretrained/ckpt/epoch_26/pytorch_model.bin',
               'SintelDataset/NMRF/sceneflow-pretrained/ckpt/epoch_522/pytorch_model.bin',  # 1
               'ArgoverseDataset/NMRF/sceneflow-pretrained/ckpt/epoch_90/pytorch_model.bin',
               # 'VirtualKitti2Dataset/NMRF/sceneflow-pretrained/ckpt/epoch_26/pytorch_model.bin',
               'InStereo2KDataset/NMRF/sceneflow-pretrained/ckpt/epoch_248/pytorch_model.bin',
               'UnrealStereo4KDataset/NMRF/sceneflow-pretrained/ckpt/epoch_67/pytorch_model.bin',
               'SpringDataset/NMRF/sceneflow-pretrained/ckpt/epoch_99/pytorch_model.bin',
               '/file_system/vepfs/algorithm/chenming.zhang/misc/pytorch_model.bin']

root = '/file_system/nas/algorithm/chenming.zhang/3090_files/code/LightStereoX/output'


def average_model_weights(model_paths):
    # 加载第一个模型的权重作为初始值
    avg_state_dict = torch.load(os.path.join(root, model_paths[0]), map_location='cpu')

    # 累加所有模型的权重
    for path in model_paths[1:]:
        state_dict = torch.load(os.path.join(root, path), map_location='cpu')
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key]

    # 计算平均值
    num_models = len(model_paths)
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / num_models

    return avg_state_dict


avg_weights = average_model_weights(model_paths)
torch.save(avg_weights, '/file_system/nas/algorithm/chenming.zhang/checkpoints/LightStereoX/output/Federated/average/ckpt/epoch_1/pytorch_model.bin')
