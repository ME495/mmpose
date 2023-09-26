#!/bin/bash

python demo/topdown_demo_with_mmdet.py \
    /home/ljh/disk_4T/cj/mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_ir_right_left.py \
    /home/ljh/disk_4T/cj/mmdetection/work_dirs/rtmdet_tiny_ir_right_left/epoch_50.pth \
    configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py \
    work_dirs/rtmpose_m_coco_hand2/epoch_20.pth \
    --input /home/ljh/disk_4T/cj/datasets/qiyuan_right_left/20230915_152914/1694762981114436865_ir1.jpg   \
    --output-root ouputs/ --draw-bbox --kpt-thr 0.05