import os.path
import torch
import torch.nn.functional as F

# 要平均的模型权重路径列表
model_paths = ['TartanAirDataset/NMRF/sceneflow-pretrained/ckpt/epoch_1/pytorch_model.bin',
               '/file_system/vepfs/algorithm/chenming.zhang/misc/pytorch_model.bin',  # carla
               'CREStereoDataset/NMRF/sceneflow-pretrained/ckpt/epoch_2/pytorch_model.bin',
               'SpringDataset/NMRF/sceneflow-pretrained/ckpt/epoch_99/pytorch_model.bin',
               'SintelDataset/NMRF/sceneflow-pretrained/ckpt/epoch_522/pytorch_model.bin',
               'DynamicReplicaDataset/NMRF/sceneflow-pretrained/ckpt/epoch_3/pytorch_model.bin',
               'FallingThingsDataset/NMRF/sceneflow-pretrained/ckpt/epoch_9/pytorch_model.bin',
               'InStereo2KDataset/NMRF/sceneflow-pretrained/ckpt/epoch_248/pytorch_model.bin',
               'VirtualKitti2Dataset/NMRF/sceneflow-pretrained/ckpt/epoch_26/pytorch_model.bin',
               # 'ArgoverseDataset/NMRF/sceneflow-pretrained/ckpt/epoch_90/pytorch_model.bin',
               # 'VirtualKitti2Dataset/NMRF/sceneflow-pretrained/ckpt/epoch_26/pytorch_model.bin',
               # 'UnrealStereo4KDataset/NMRF/sceneflow-pretrained/ckpt/epoch_67/pytorch_model.bin',
               ]

root = '/file_system/nas/algorithm/chenming.zhang/3090_files/code/LightStereoX/output'


def average_model_weights(model_paths):
    logits = torch.tensor([3.07, 13.31, 3.0, 2.7, 3.27, 3.3, 2.72, 4.43, 2.97])
    probs = F.softmax(-logits, dim=0).tolist()
    # 加载第一个模型的权重作为初始值
    avg_state_dict = torch.load(os.path.join(root, model_paths[0]), map_location='cpu')
    for key in avg_state_dict:
        if avg_state_dict[key].dtype in [torch.float32, torch.float16, torch.float64]:
            avg_state_dict[key] *= probs[0]

    # 累加所有模型的权重
    for i, path in enumerate(model_paths[1:], start=1):
        state_dict = torch.load(os.path.join(root, path), map_location='cpu')
        for key in avg_state_dict:
            if avg_state_dict[key].dtype in [torch.float32, torch.float16, torch.float64]:
                avg_state_dict[key] += state_dict[key] * probs[i]

    # 计算平均值
    # num_models = len(model_paths)
    # for key in avg_state_dict:
    #     avg_state_dict[key] = avg_state_dict[key] / num_models

    return avg_state_dict


avg_weights = average_model_weights(model_paths)
torch.save(avg_weights,
           '/file_system/nas/algorithm/chenming.zhang/checkpoints/LightStereoX/output/Federated/weighted_average/ckpt/epoch_1/pytorch_model.bin')

# kitti12
# run 9个model
# 每张图9个结果
# mean 每张图得到一个结果，把这个mean的结果当作gt
# 每个model在kitti12上的epe就可以得到（pred - mean_gt).mean()

# 9个epe
# softmax(-epe)
# 得到了一个和为1的概率
# 根据这个概率，以及上面的代码，把这9个model的权重做加权求和
# 保存最终权重
# {'epe': 3.07, 'd1_all': 23.72, 'thres_1': 77.57, 'thres_2': 47.06, 'thres_3': 27.26}
# {'epe': 13.31, 'd1_all': 87.17, 'thres_1': 95.32, 'thres_2': 90.73, 'thres_3': 87.18}
# {'epe': 3.0, 'd1_all': 22.8, 'thres_1': 80.14, 'thres_2': 49.03, 'thres_3': 26.53}
# {'epe': 2.7, 'd1_all': 16.57, 'thres_1': 71.88, 'thres_2': 40.03, 'thres_3': 20.24}
# {'epe': 3.27, 'd1_all': 20.58, 'thres_1': 76.03, 'thres_2': 44.48, 'thres_3': 24.37}
# {'epe': 3.3, 'd1_all': 36.03, 'thres_1': 79.01, 'thres_2': 56.58, 'thres_3': 39.33}
# {'epe': 2.72, 'd1_all': 16.42, 'thres_1': 74.01, 'thres_2': 40.86, 'thres_3': 20.09}
# {'epe': 4.43, 'd1_all': 33.96, 'thres_1': 84.72, 'thres_2': 56.93, 'thres_3': 37.46}
# {'epe': 2.97, 'd1_all': 21.61, 'thres_1': 78.02, 'thres_2': 46.57, 'thres_3': 24.66}
