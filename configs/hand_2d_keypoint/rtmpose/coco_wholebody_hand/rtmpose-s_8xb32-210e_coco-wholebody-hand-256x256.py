_base_ = ['rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py']

model = dict(
    backbone=dict(
        deepen_factor=0.33,
        widen_factor=0.5),
    head=dict(in_channels=512))

# runtime
max_epochs = 100
stage2_num_epochs = 5
base_lr = 1e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[25, 40],
        gamma=0.1)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=10)
val_dataloader = dict(
    batch_size=256,
    num_workers=10)
test_dataloader = val_dataloader