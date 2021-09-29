model = dict(
    type='DynamicFasterRCNN',
    backbone=dict(
        type='DynamicResNet',
        in_channels=3,
        stem_width=64,
        body_depth=[4, 6, 29, 4],
        body_width=[80, 160, 320, 640],
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        conv_cfg=dict(type='DynConv2d'),
        norm_cfg=dict(type='DynBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='DynamicFPN',
        in_channels=[320, 640, 1440, 2560],
        out_channels=256,
        conv_cfg=dict(type='DynConv2d'),
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=696,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=4.0),
            loss_bbox=dict(type='SmoothL1Loss', loss_weight=4.5))))
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=5,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1),
        sampler=dict(
            type='CombinedSampler',
            num=512,
            pos_fraction=0.25,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3)),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
stem_width_range = dict(
    key='arch.backbone.stem.width', start=32, end=64, step=16)
body_width_range = dict(
    key='arch.backbone.body.width',
    start=[48, 96, 192, 384],
    end=[80, 160, 320, 640],
    step=[16, 32, 64, 128],
    ascending=True)
body_depth_range = dict(
    key='arch.backbone.body.depth',
    start=[2, 2, 5, 2],
    end=[4, 6, 29, 4],
    step=[1, 2, 2, 1])
MAX = dict({
    'name': 'MAX',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [80, 160, 320, 640],
    'arch.backbone.body.depth': [4, 6, 29, 4]
})
MIN = dict({
    'name': 'MIN',
    'arch.backbone.stem.width': 32,
    'arch.backbone.body.width': [48, 96, 192, 384],
    'arch.backbone.body.depth': [2, 2, 5, 2]
})
R50 = dict({
    'name': 'R50',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 6, 3]
})
R77 = dict({
    'name': 'R77',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 15, 3]
})
R101 = dict({
    'name': 'R101',
    'arch.backbone.stem.width': 64,
    'arch.backbone.body.width': [64, 128, 256, 512],
    'arch.backbone.body.depth': [3, 4, 23, 3]
})
train_sampler = dict(
    type='concat',
    model_samplers=[
        dict(
            type='anchor',
            anchors=[
                dict({
                    'name': 'MAX',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [80, 160, 320, 640],
                    'arch.backbone.body.depth': [4, 6, 29, 4]
                }),
                dict({
                    'name': 'MIN',
                    'arch.backbone.stem.width': 32,
                    'arch.backbone.body.width': [48, 96, 192, 384],
                    'arch.backbone.body.depth': [2, 2, 5, 2]
                }),
                dict({
                    'name': 'R101',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [64, 128, 256, 512],
                    'arch.backbone.body.depth': [3, 4, 23, 3]
                }),
                dict({
                    'name': 'R77',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [64, 128, 256, 512],
                    'arch.backbone.body.depth': [3, 4, 15, 3]
                }),
                dict({
                    'name': 'R50',
                    'arch.backbone.stem.width': 64,
                    'arch.backbone.body.width': [64, 128, 256, 512],
                    'arch.backbone.body.depth': [3, 4, 6, 3]
                })
            ]),
        dict(
            type='repeat',
            times=3,
            model_sampler=dict(
                type='composite',
                model_samplers=[
                    dict(
                        type='range',
                        key='arch.backbone.stem.width',
                        start=32,
                        end=64,
                        step=16),
                    dict(
                        type='range',
                        key='arch.backbone.body.width',
                        start=[48, 96, 192, 384],
                        end=[80, 160, 320, 640],
                        step=[16, 32, 64, 128],
                        ascending=True),
                    dict(
                        type='range',
                        key='arch.backbone.body.depth',
                        start=[2, 2, 5, 2],
                        end=[4, 6, 29, 4],
                        step=[1, 2, 2, 1])
                ]))
    ])
val_sampler = dict(
    type='anchor',
    anchors=[
        dict({
            'name': 'R50',
            'arch.backbone.stem.width': 64,
            'arch.backbone.body.width': [64, 128, 256, 512],
            'arch.backbone.body.depth': [3, 4, 6, 3]
        }),
        dict({
            'name': 'R77',
            'arch.backbone.stem.width': 64,
            'arch.backbone.body.width': [64, 128, 256, 512],
            'arch.backbone.body.depth': [3, 4, 15, 3]
        }),
        dict({
            'name': 'R101',
            'arch.backbone.stem.width': 64,
            'arch.backbone.body.width': [64, 128, 256, 512],
            'arch.backbone.body.depth': [3, 4, 23, 3]
        })
    ])
dataset_types = dict(
    coco='CocoDataset',
    object365='NamedCustomDataset',
    openimages='NamedCustomDataset')
data_roots = dict(
    coco='/data3/ming_zeng/gaia/coco',
    openimages='/data3/ming_zeng/gaia/openimages',
    object365='/data3/ming_zeng/gaia/object365')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        multiscale_mode='range',
        img_scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        datasets=[
            dict(
                type='CocoDataset',
                ann_file=
                '/data3/ming_zeng/gaia/coco/annotations/instances_train2017.json',
                img_prefix='/data3/ming_zeng/gaia/coco/images/train2017',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Resize',
                        multiscale_mode='range',
                        img_scale=[(1333, 640), (1333, 800)],
                        keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ]),
            dict(
                type='NamedCustomDataset',
                name='object365',
                ann_file=
                '/data3/ming_zeng/gaia/object365/annotations/objects365_generic_train.json',
                img_prefix='/data3/ming_zeng/gaia/object365/images/train',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Resize',
                        multiscale_mode='range',
                        img_scale=[(1333, 640), (1333, 800)],
                        keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
                ])
        ]),
    val=dict(
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        dataset_names=['coco'],
        test_mode=True,
        datasets=[
            dict(
                type='CocoDataset',
                ann_file=
                '/data3/ming_zeng/gaia/coco/annotations/instances_val2017.json',
                img_prefix='/data3/ming_zeng/gaia/coco/images/val2017',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='MultiScaleFlipAug',
                        img_scale=(1333, 800),
                        flip=False,
                        transforms=[
                            dict(type='Resize', keep_ratio=True),
                            dict(type='RandomFlip'),
                            dict(
                                type='Normalize',
                                mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375],
                                to_rgb=True),
                            dict(type='Pad', size_divisor=32),
                            dict(type='ImageToTensor', keys=['img']),
                            dict(type='Collect', keys=['img'])
                        ])
                ])
        ]),
    test=dict(
        samples_per_gpu=8,
        type='UniConcatDataset',
        label_pool_file='hubs/labels/uni.0.0.3.csv',
        dataset_names=['coco'],
        test_mode=True,
        datasets=[
            dict(
                type='CocoDataset',
                ann_file=
                '/data3/ming_zeng/gaia/coco/annotations/instances_val2017.json',
                img_prefix='/data3/ming_zeng/gaia/coco/images/val2017',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='MultiScaleFlipAug',
                        img_scale=(1333, 800),
                        flip=False,
                        transforms=[
                            dict(type='Resize', keep_ratio=True),
                            dict(type='RandomFlip'),
                            dict(
                                type='Normalize',
                                mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375],
                                to_rgb=True),
                            dict(type='Pad', size_divisor=32),
                            dict(type='ImageToTensor', keys=['img']),
                            dict(type='Collect', keys=['img'])
                        ])
                ])
        ]))
evaluation = dict(
    interval=1, metric='bbox', save_best='arch_avg', dataset_name='coco')
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=1e-05)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_scaler = dict(policy='linear', base_lr=0.00125)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    step=[32, 38, 41])
total_epochs = 1
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.0)
work_dir = 'work_dir'
gpu_ids = range(0, 8)
