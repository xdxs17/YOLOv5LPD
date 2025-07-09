# YOLOv5 for License Plate Detection (YOLOv5LPD)

In YOLOv5, BiFPN replaces PANet, and the 20x20 large object detection head is removed to optimise the network structure (because in real-world scenarios, license plates typically fall under the category of small objects. The 20x20 detection scale's receptive field is too large for license plate detection). Relevant details can be found in models/models/yolov5LPD.yaml.

BiFPN requires five feature maps as input, so this repository has made corresponding modifications and simplifications to BiFPN to adapt it to the neck structure of YOLOv5.

# Training

python train.py --img 640 --batch 8 --epochs 200 --data CCPD.yaml --weights '' --cfg yolov5LPD.yaml

# Reference

https://github.com/ultralytics/yolov5

https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

Tan, R. Pang, and Q. Le, “EfficientDet: scalable and efficient object detection,” in Proc. **the 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition** (**CVPR** **2020**), Seattle, WA, USA, Jun., 2020, pp. 10778-10787.
