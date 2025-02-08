# @Time    : 2024/3/1 11:17
# @Author  : zhangchenming
import time
import torch
import argparse
import thop

from tqdm import tqdm

from stereo.config.lazy import LazyConfig
from stereo.config.instantiate import instantiate


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, required=True, help='specify the config')

    args = parser.parse_args()
    cfg = LazyConfig.load(args.cfg_file)
    args.run_mode = 'measure'
    return cfg, args


def main():
    cfg, args = parse_config()
    model = instantiate(cfg.model).cuda()

    shape = [1, 3, 544, 960]
    infer_time(model, shape)
    measure(model, shape)


@torch.no_grad()
def measure(model, shape):
    model.eval()

    inputs = {'left': torch.randn(shape).cuda(),
              'right': torch.randn(shape).cuda()}

    flops, params = thop.profile(model, inputs=(inputs,))
    print("Number of calculates:%.2fGFlops" % (flops / 1e9))
    print("Number of parameters:%.2fM" % (params / 1e6))


@torch.no_grad()
def infer_time(model, shape):
    model.eval()
    repetitions = 100

    inputs = {'left': torch.randn(shape).cuda(),
              'right': torch.randn(shape).cuda()}

    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)

    all_time = 0
    print('testing ...\n')
    with torch.no_grad():
        for _ in tqdm(range(repetitions)):
            infer_start = time.perf_counter()
            result = model(inputs)
            # print(result.keys())
            all_time += time.perf_counter() - infer_start

    print(all_time / repetitions * 1000)


if __name__ == '__main__':
    main()
