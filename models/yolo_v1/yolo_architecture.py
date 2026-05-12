

lambda_coord = 5
lambda_noobj = 0.5

conf_threshold = 0.4
iou_threshold = 0.5
dropout_rate = 0.5

# batch_size = 64
batch_size = 16
# epochs = 135
epochs = 10
lr = 1e-5
weight_decay = 5e-4

YOLO_architecture = [
    ("Conv", 7, 64, 2, 3),
    ("Maxpool", 2, 2),

    ("Conv", 3, 192, 1, 1),
    ("Maxpool", 2, 2),

    ("Conv", 1, 128, 1, 0),
    ("Conv", 3, 256, 1, 1),
    ("Conv", 1, 256, 1, 0),
    ("Conv", 3, 512, 1, 1),

    ("Maxpool", 2, 2),

    ("Repeat", 4,
        [
            ("Conv", 1, 256, 1, 0),
            ("Conv", 3, 512, 1, 1),
        ]
    ),

    ("Conv", 1, 512, 1, 0),
    ("Conv", 3, 1024, 1, 1),

    ("Maxpool", 2, 2),

    ("Repeat", 2,
        [
            ("Conv", 1, 512, 1, 0),
            ("Conv", 3, 1024, 1, 1),
        ]
    ),

    ("Conv", 3, 1024, 1, 1),
    ("Conv", 3, 1024, 2, 1),
    ("Conv", 3, 1024, 1, 1),
    ("Conv", 3, 1024, 1, 1),



    ("Flatten",),

    ("FC", 1024 * S * S, 4096),

    ("LeakyReLU", 0.1),

    ("Dropout", dropout_rate),

    ("FC", 4096, S * S * (C + B * 5)),
]