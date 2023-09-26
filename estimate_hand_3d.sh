#!/bin/bash

python demo/estimate_hand_3d.py\
    /home/ljh/disk_4T/cj/mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_ir_right_left.py \
    /home/ljh/disk_4T/cj/mmdetection/work_dirs/rtmdet_tiny_ir_right_left/epoch_50.pth \
    configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py \
    work_dirs/rtmpose_m_coco_hand/epoch_210.pth \
    --output-root ouputs/ --draw-bbox --kpt-thr 0.05