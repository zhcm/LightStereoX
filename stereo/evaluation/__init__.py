# @Time    : 2023/8/29 18:23
# @Author  : zhangchenming
from functools import partial
from .metric_per_image import epe_metric, threshold_metric, d1_metric

metric_funcs = {
    'epe': epe_metric,
    'd1_all': d1_metric,
    'thres_1': partial(threshold_metric, threshold=1),
    'thres_2': partial(threshold_metric, threshold=2),
    'thres_3': partial(threshold_metric, threshold=3),
}

metric_names = ['epe', 'd1_all', 'thres_1', 'thres_2', 'thres_3']
