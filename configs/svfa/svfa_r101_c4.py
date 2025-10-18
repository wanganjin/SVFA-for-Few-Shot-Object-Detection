_base_ = [
    './meta-rcnn_r50_c4.py',
]
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    type='SVFA',
    post_rpn=True,
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='SVFARoIHead',
        shared_head=dict(pretrained=pretrained),
        bbox_head=dict(num_classes=20, num_meta_classes=20),
        novel_class=(15, 16, 17, 18, 19),
        prototypes_distillation=dict(
            type='PrototypesDistillation',
            num_queries=5, dim=1024, num_base_cls=15),
        prototypes_assignment=dict(
            type='PrototypesAssignment',
            dim=1024, num_bg=5),
    )

)
