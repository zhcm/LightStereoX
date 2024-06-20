# @Time    : 2024/4/19 10:24
# @Author  : zhangchenming
import glob
from pathlib import Path


root_dir = Path('/mnt/nas/public_data/KITTI/virtualkitti2')

# scenes = glob.glob(str(root_dir.joinpath('*')), recursive=False)
# with open("virtualkitti2_trainval21260.txt", "w") as f:
#     for scene in scenes:
#         variations = glob.glob(scene + '/*', recursive=False)
#         for variation in variations:
#             left_images = glob.glob(variation + '/frames/rgb/Camera_0/*', recursive=False)
#             for left_image in left_images:
#                 left_image = left_image[len(str(root_dir))+1:]
#                 right_image = left_image.replace('Camera_0', 'Camera_1')
#                 left_depth = left_image.replace('rgb', 'depth').replace('jpg', 'png')
#                 right_depth = right_image.replace('rgb', 'depth').replace('jpg', 'png')
#                 f.write(left_image + ' ' + right_image + ' ' + left_depth + ' ' + right_depth + '\n')


data_list = []
with open('virtualkitti2_trainval21260.txt', 'r') as fp:
    for x in fp.readlines():
        data_list.extend(x.strip().split(' '))
    for each in data_list:
        assert root_dir.joinpath(each).exists()
