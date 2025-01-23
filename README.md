# Safety Equipment Detection

This project involves the detection of safety equipment using YOLOv8, a cutting-edge object detection algorithm. The dataset used for training and evaluation includes labeled images of safety equipment such as helmets, gloves, vests, and more, ensuring accurate detection for workplace safety compliance.

---

## **Project Overview**
This project aims to:
1. Detect safety equipment (e.g., hard hats, gloves, masks) using the YOLOv8 model.
2. Train, validate, and test the model on a custom dataset.
3. Convert annotation formats (Pascal VOC to YOLOv8).
4. Visualize detection results to assess model performance.

---

## **Directory Structure**
```plaintext
.
├── README.md                # Project documentation
├── datasets                 # Dataset folder
│   ├── classes.txt          # List of classes
│   ├── pascalVOC_to_yolo.py # Script for converting PascalVOC annotations to YOLO format
├── hardhat7041yolov8.yaml   # YOLOv8 dataset configuration file
├── train_yolov8_for_hardhat_detection.ipynb # Training and evaluation notebook
└── runs                     # Output folder for training, validation, and detection results
```

---

## **Dataset Details**
- **Dataset Root:** `D:\syook\Syook-Project\hard_hat_workers`
- **Training Images:** `5269`
- **Validation Images:** `1766`
- **Classes:**
  1. Head
  2. Helmet
  3. Person

---

## **Key Files**

### **1. `hardhat7041yolov8.yaml`**
This file defines the dataset configuration for YOLOv8.
```yaml
path: D:\syook\Syook-Project\hard_hat_workers
train: train\images
val: test\images
nc: 3
names: ["head", "helmet", "person"]
```

### **2. `pascalVOC_to_yolo.py`**
Converts PascalVOC XML annotations to YOLO format.
- **Input:** PascalVOC XML files.
- **Output:** YOLO annotation text files.

**Usage:**
```bash
python pascalVOC_to_yolo.py <input_dir> <output_dir>
```

---

## **YOLOv8 Training and Inference**

### **1. Model Training**
The model is trained using YOLOv8 with the following parameters:
- **Pretrained Weights:** `yolov8s.pt`
- **Image Size:** `672`
- **Batch Size:** `10`
- **Epochs:** `1`

**Code Snippet:**
```python
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Train the model
results = model.train(
    data="D:\\syook\\Syook-Project\\hardhat7041yolov8.yaml",
    imgsz=672, batch=10,
    project="runs/train/", name="exp_hardhat", epochs=1, cache=True
)
```

### **2. Model Evaluation**
Evaluate the trained model on the validation set:
```python
model = YOLO("runs/train/exp_hardhat/weights/best.pt")
results = model.val(project="runs/val/", name="exp_hardhat")
```

### **3. Inference**
Run inference on test images and save the results:
```python
results = model.predict(
    source="../datasets/test_images/hardhat/", save=True,
    project="runs/detect/", name="exp_hardhat", save_txt=True, line_thickness=2
)
```

---

## **Visualization**
Visualize detection results using `matplotlib`:
```python
import matplotlib.pyplot as plt
bbox_img1 = plt.imread("runs/detect/exp_hardhat/image1.jpg")
plt.imshow(bbox_img1)
plt.axis("off")
plt.show()
```

---

## **Classes**
The dataset includes the following safety equipment classes:
1. Person
2. Hard Hat
3. Gloves
4. Mask
5. Glasses
6. Boots
7. Vest
8. PPE Suit
9. Ear Protector
10. Safety Harness

---

## **Environment Setup**

- **Python Version:** 3.10
- **Framework:** YOLOv8
- **GPU Support:** Ensure CUDA is available for faster training.

**Verify Setup:**
```python
import torch
print("Torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print("NO GPU")
```

---

## **Results**
- **Best Model Path:** `runs/train/exp_hardhat/weights/best.pt`
- **File Size:** Check using:
```python
import os
file_path = "D:\\syook\\Syook-Project\\runs\\train\\exp_hardhat\\weights\\best.pt"
file_size = os.path.getsize(file_path)
print(f"File Size: {file_size} bytes")
```

---

## **Future Scope**
1. Extend the dataset to include more safety equipment.
2. Optimize YOLOv8 training parameters for better accuracy.
3. Deploy the trained model as a real-time detection system using a webcam or CCTV feed.

---

## **Acknowledgments**
This project utilizes the YOLOv8 framework by Ultralytics and leverages custom datasets for safety equipment detection.
