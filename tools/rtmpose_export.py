import torch
import torch.nn as nn
import onnx

from mmpose.apis import init_model

config = 'configs/hand_2d_keypoint/rtmpose/coco_wholebody_hand/rtmpose-s_8xb32-210e_coco-wholebody-hand-256x256.py'
checkpoint = 'work_dirs/rtmpose_s_coco_hand2/best_AUC_epoch_100.pth'
device = 'cuda:0'

class RTMPoseWithDecode(nn.Module):
    def __init__(self, config, checkpoint):
        super().__init__()
        self.detector = init_model(config, checkpoint, device)
        
    def forward(self, x):
        simcc_x, simcc_y = self.detector.forward(x, None)
        
        max_val_x, x_locs = torch.max(simcc_x, dim=2)
        max_val_y, y_locs = torch.max(simcc_y, dim=2)
        scores = torch.max(torch.stack([max_val_x, max_val_y], dim=-1), dim=-1)[0]
        keypoints = torch.stack([x_locs, y_locs], dim=-1)
        keypoints = keypoints.float() / self.detector.cfg.codec.simcc_split_ratio
        
        return keypoints, scores


model = RTMPoseWithDecode(config, checkpoint)
dummy_image = torch.zeros((1, 3, 256, 256), device=device)
keypoints, scores = model.forward(dummy_image)
print(keypoints.shape, scores.shape)

torch.onnx.export(
        model,
        dummy_image,
        'rtmpose.onnx',
        input_names=['input'],
        opset_version=11,
        dynamic_axes={'input': {
            0: 'batch'
        }}
        )