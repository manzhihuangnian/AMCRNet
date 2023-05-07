# model setting
window_size=15
max_person_mun_persec=-1
# data_root_path="your images path (extracted from AVA videos) "
data_root_path="junyu"

model = dict(
    type='AMCRNet_Dynamic',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            fusion_kernel=7,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),

    pose_net=dict(
            type="convpos",
            deepth=6,
            out_channels=1024,
            dims=[1,2,2,2,2,4,16],
            conv_size=[[3,3],[3,3],[3,3],[3,3],[3,3],[3,3]],
            dilation=[[1,2],[1,2],[1,2],[1,2],[1,2],[1,2]],
            out_conv=[[3,1,1024]],
            norm_cfg=dict(type="BN")
    ),

    head=dict(
        type="AMCRNet_Dynamic_Head",
        mask_mode="zero",
        grid=False,
        roi_head=dict(
            type='AMCRNet_roiextractor',
            roi_layer_type='RoIAlign',
            output_size=8,
            channels=3072,
            temporal_pool_size=2,
            channel_ratio=6,
            sampling_ratio=0,
            temporal_down_type="avgpool",
            use_channel_ratio=True,
            norm_cfg=dict(type="BN3d")
        ),

        bbox_head=dict(
        type="AnchorFree_head",
        activate=False,
        ),

        context_module=dict(
            type='pool_out',
            channels=1024,
            pool_ratio=[[3, 3], [5, 5], [7, 7]],
            norm_cfg=dict(type="BN"),
            layer_embed=False),

        BHOI_module_cfg=dict(
            type="MHSA",
            num_head=8,
            deepth=4,
            dims=1024,
            mlp_ratio=1,
            drop_path=0.1,
            ),

        window_size=window_size,

        Dynamic_LFB=dict(
            type="Dynamic_Feature_Bank",
            window_size=window_size,
            max_person_mun_persec=max_person_mun_persec,
            ),

        BHOI_module_high_cfg=dict(
            type="MHSA",
            num_head=8,
            deepth=2,
            dims=1024,
            mlp_ratio=1,
            drop_path=0.1),

        cls_head=dict(
            num_classes=81,
            in_channels=1024,
            dropout_ratio=0.45,
            multilabel=True,
            )
    ),
        test_cfg=dict(action_thr=0.002)
)

gpu_ids=[0]
dataset_type = 'AVADataset'
data_root = f'/data/{data_root_path}/datasets/ava/rawframes'
anno_root = '../data/ava/annotations'
ann_file_val = f'{anno_root}/ava_val_v2.2.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.2.csv'
label_file = f'{anno_root}/ava_action_list_v2.2_for_activitynet_2019.pbtxt'
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=32, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key=['proposals'], stack=False),
                                         dict(key=['img'], stack=True)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['scores', 'img_shape', "img_key", 'gt_labels','gt_bboxes'],
        nested=True)
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        person_det_score_thr=0.9,
        data_prefix=data_root))
data['test'] = data['val']

dist_params = dict(backend='nccl')
log_level = 'INFO'

