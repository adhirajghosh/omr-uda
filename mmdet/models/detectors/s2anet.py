from .single_stage import SingleStageDetector, SingleStageDetectorDA
from ..registry import DETECTORS


@DETECTORS.register_module
class S2ANetDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(S2ANetDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                             test_cfg, pretrained)


@DETECTORS.register_module
class S2ANetUDA(SingleStageDetectorDA):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(S2ANetUDA, self).__init__(backbone, neck, bbox_head, train_cfg,
                                             test_cfg, pretrained)
