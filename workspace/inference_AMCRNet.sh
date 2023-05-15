#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port="29502" ../tools/test_AMCRNet_Twostage.py\
    --launcher pytorch\
    --config ../checkpoints/configs/AMCRNet/AMCRNet_Dynamic_LFB_OneBranch_no_ema_r50_8x8_test.py\
    --checkpoint ../checkpoints/checkpoints/AMCRNet/AMCRNet-slim_R50_8x8.pth\
    --eval mAP \
    --out ./results/results_fast50_8x8_k400.csv

# python -m torch.distributed.launch --nproc_per_node=1 --master_port="29502" ../tools/test_AMCRNet_Twostage.py\
#     --launcher pytorch\
#     --config ../checkpoints/configs/AMCRNet/AHR_Dynamic_LFB_OneBranch_no_ema_slowonly50_4x16.py\
#     --checkpoint ../checkpoints/checkpoints/AMCRNet/AMCRNet-slim_slowonly50_4x16.pth\
#     --eval mAP \
#     --out ./results/results_slowonly50_4x16_k400.csv

# python -m torch.distributed.launch --nproc_per_node=1 --master_port="29502" ../tools/test_AMCRNet_Twostage.py\
#     --launcher pytorch\
#     --config ../checkpoints/configs/AMCRNet/AHR_Dynamic_LFB_OneBranch_no_ema_r50_4x16.py\
#     --checkpoint ../checkpoints/checkpoints/AMCRNet/AMCRNet-slim_R50_4x16.pth\
#     --eval mAP \
#     --out ./results/results_slowfastR50_4x16_k400.csv

# python -m torch.distributed.launch --nproc_per_node=1 --master_port="29502" ../tools/test_AMCRNet_Twostage.py\
#     --launcher pytorch\
#     --config ../checkpoints/configs/AMCRNet/AHR_Dynamic_LFB_OneBranch_no_ema_r101_8x8.py\
#     --checkpoint ../checkpoints/checkpoints/AMCRNet/AMCRNet-slim_R101_8x8.pth\
#     --eval mAP \
#     --out ./results/results_slowfastR101_8x8_k400.csv
