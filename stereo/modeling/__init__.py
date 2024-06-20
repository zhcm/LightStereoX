# @Time    : 2023/8/26 13:02
# @Author  : zhangchenming
from stereo.modeling.models.psmnet.psmnet import PSMNet
from stereo.modeling.models.igev.igev_stereo import IGEVStereo as IGEVStereo
from stereo.modeling.models.stereobase.stereobase import StereoBase
from stereo.modeling.models.stereobase.stereobase_gru import StereoBase as StereoBaseGRU
from stereo.modeling.models.sttr.sttr import STTR
from stereo.modeling.models.stereonet.stereonet import StereoNet
from stereo.modeling.models.diff.basic_diff import IgevNoGruDiff
from stereo.modeling.models.diff.basic_diff_convgru import IgevNoGruDiff as IgevGruDiff
from stereo.modeling.models.eccvstereo.eccvstereo import StereoBase as ECCVStereo
from stereo.modeling.models.coex.coex import CoEx
from stereo.modeling.models.eccvstereo.igev_stereo_nogru import IGEVStereo as LightBest
from stereo.modeling.models.lightstereo.lightstereo import LightStereo
from stereo.modeling.models.lightstereo_new.lightstereo import LightStereo as LightStereoNew
from stereo.modeling.models.lightfast.lightstereo import LightStereo as LightFast
from stereo.modeling.models.igev_nogru.igev_stereo_nogru import IGEVStereo as IGEVStereoNogru


__all__ = {
    'IGEVStereoNogru': IGEVStereoNogru,
    'LightFast': LightFast,
    'LightStereoNew': LightStereoNew,
    'LightStereo': LightStereo,
    'LightBest': LightBest,
    'CoEx': CoEx,
    'ECCVStereo': ECCVStereo,
    'PSMNet': PSMNet,
    'IGEVStereo': IGEVStereo, # 重新实现的igev，与openstereo训练过程可以完全对齐
    'StereoBase': StereoBase,
    'StereoBaseGRU': StereoBaseGRU,
    'Diff': IgevNoGruDiff,
    'DiffGRU': IgevGruDiff,
    'STTR': STTR,
    'StereoNet': StereoNet
}


def build_network(model_cfg):
    model = __all__[model_cfg.NAME](model_cfg)
    return model