# from .mmdet_builder import DETECTORS
import torch
import torch.nn as nn

from networks.fpn import FPN
from networks.fcos_head import FCOSHead
# from .single_stage import SingleStageDetector

from networks.resnet import resnet50
# from mmdet.utils import get_root_logger

# @DETECTORS.register_module()
class FCOS(nn.Module):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone_args,
                 neck_args,
                 bbox_head_args,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        # super(FCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
        #                            test_cfg, pretrained)
        super(FCOS, self).__init__()
        self.backbone = resnet50(**backbone_args)
        if neck_args is not None:
            self.neck = FPN(**neck_args)
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = FCOSHead(**bbox_head_args)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # super(SingleStageDetector, self).init_weights(pretrained)
        if pretrained is not None:
            # logger = get_root_logger()
            # print_log(f'load model from: {pretrained}', logger=logger)
            print(f'load model from: {pretrained}')
        # self.backbone.init_weights(pretrained=pretrained)  # default resnet does not have .init_weights attribute
        # if self.with_neck:
        #     if isinstance(self.neck, nn.Sequential):
        #         for m in self.neck:
        #             m.init_weights()
        #     else:
        #         self.neck.init_weights()
        self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        # x = self.backbone(img)
        _, x = self.backbone(img, return_feat=True)
        # if self.with_neck:
        #     x = self.neck(x)
        x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        return self.forward(img)

    def forward(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


if __name__ == "__main__":
    # backbone_args = dict(depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=1,
    #     norm_cfg=dict(type='BN', requires_grad=False),
    #     norm_eval=True,
    #     style='caffe')
    backbone_args = dict(
        pretrained=False,
        progress=True,
        width=1,
        in_channel=3,
    )
    neck_args = dict(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True,
    )
    bbox_head_args = dict(
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
    )
    fcos = FCOS(backbone_args=backbone_args, neck_args=neck_args, bbox_head_args=bbox_head_args)
    print(fcos)

    image = torch.randn(2, 3, 224, 224)
    
    outs = fcos(image)
    print(type(outs))
    for out in outs:
        print(len(out))

    state_dict = fcos.state_dict()

    meta = {}
    checkpoint = {
        'meta': meta,
        'state_dict': state_dict,  # weights_to_cpu(get_state_dict(model))
    }

    torch.save(checkpoint, 'fcos_base.pth')
