import torch
from models.yolo import Model

model = Model('models/yolov5LPD.yaml', nc=1)
print("Model created successfully")
x = torch.randn(1, 3, 640, 640)
outputs = model(x)
print([out.shape for out in outputs])  # 预期：[bs, na*(nc+5), 80, 80], [bs, na*(nc+5), 40, 40]