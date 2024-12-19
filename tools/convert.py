# @Time    : 2024/12/17 15:40
# @Author  : zhangchenming
import torch


def covnert_model(filename):
    checkpoint = torch.load(filename, map_location='cpu')
    state = {'model_state': checkpoint}
    torch.save(state, '/mnt/nas/algorithm/chenming.zhang/misc/result.pt')


def covnert_lightstereo(filename):
    checkpoint = torch.load(filename, map_location='cpu')
    result = {}
    for key, val in checkpoint.items():
        if 'neck' in key:
            result[key.replace('neck', 'backbone')] = val
        else:
            result[key] = val
    state = {'model_state': result}
    torch.save(state, '/mnt/nas/algorithm/chenming.zhang/misc/result.pt')


if __name__ == '__main__':
    covnert_lightstereo('output/MixDataset/LightStereo_L/carla-ds-cre-ft-i2k-tar-st/ckpt/epoch_0/pytorch_model.bin')
