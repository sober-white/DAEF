DAEF: A Real-Time End-to-End Battery Defect Detection Model with Edge Feature Aggregation and Hard Sample Focusing

## 1. Description
            
This dataset, named **BSD**, was collected from battery production lines. It contains 1,266 images in total, each of size 5184×3456 pixels. The dataset is designed to aid in real-time battery surface defect detection and includes five distinct defect categories:

- **Indentation (Id): irregular lateral surface indentations**
- **Explosion valve indentation (Ei): smaller dimensions on explosion-proof valves**
- **Leakage (Le): electrolyte leakage from terminals**
- **Squeeze (Sq): lateral protrusions, sometimes confused with Id**
- **Blue Film deficiency (Bd): missing blue film on valves, often co-occurring with Ei**

Images were captured under spherical lighting to minimize shadows, and any blurry or overexposed images have been filtered out. The dataset is accompanied by annotated defect regions in COCO-format JSON files, facilitating training, validation, and testing of defect detection models.

## 2. Dataset Size

- **Total Dataset Size:** 5.77 GB
- **Number of Images:** 1,266
- **Image Resolution:** 5184×3456 pixels per image

## 3. Platform

The dataset and associated code can be used in the following environments:

- **IDE Platforms:** Visual Studio Code (VSCode) or PyCharm

## 4. Environment Requirements

To work with the dataset and implement the defect detection model, ensure the following libraries are installed:

- **Python Libraries:**
  - torch >= 2.0.1
  - torchvision >= 0.15.2
  - faster-coco-eval >= 1.6.5
  - PyYAML
  - tensorboard
  - scipy
  - calflops
  - thop
  - transformers
  - pytorch_wavelets == 1.3.0
  - timm == 1.0.7
  - mmengine == 0.10.7
  - mmcv == 2.2.0
  - grad-cam == 1.5.4

- **Installation Command:**  
  Run the following command to install required dependencies:
  ```bash
  pip install -r requirements.txt

## 5. Structure your dataset directories as follows:

    ```shell
    dataset/
    ├── train/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...    
    │   │── annotations/
    │       ├── instances_train.json
    ├── val/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...    
    │   │── annotations/
    │       ├── instances_val.json
    ├── test/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...    
    │   │── annotations/
    │       ├── instances_test.json
    ```

## 6. Set-up Instructions:

Follow these steps to set up the dataset and prepare for training:
- **Clone the repository containing the code.** 
- **Ensure that all required libraries are installed as described in Environment Requirements.**  
- **Download and extract the [BSD](https://pan.baidu.com/s/1GS3CKhxnUPV_73OZjhLuSQ?pwd=hcqm) dataset.** 
- **Place the dataset into the directory structure described above.** 
- **Modify the batter_detection.yaml configuration file to point to the correct paths for images and annotations (see below for an example).** 

    ```yaml
    task: detection

    evaluator:
    type: CocoEvaluator
    iou_types: ['bbox', ]

    num_classes: 5 # your dataset classes
    remap_mscoco_category: False

    train_dataloader:
    type: DataLoader
    dataset:
        type: CocoDetection
        img_folder: datasets\battery-train\images
        ann_file: battery-train\annotations\train.json
        return_masks: False
        transforms:
        type: Compose
        ops: ~
    shuffle: True
    num_workers: 4
    drop_last: True
    collate_fn:
        type: BatchImageCollateFunction

    val_dataloader:
    type: DataLoader
    dataset:
        type: CocoDetection
        img_folder: datasets\battery-val\images
        ann_file: datasets\battery-val\annotations\val.json
        return_masks: False
        transforms:
        type: Compose
        ops: ~
    shuffle: False
    num_workers: 4
    drop_last: False
    collate_fn:
        type: BatchImageCollateFunction
    ```

## 7. Training Instructions:
After setting up the dataset, you can start training by running the following command:
``` shell
python train.py -c configs/deim_dfine/deim_hgnetv2_n_battery.yml -d cuda:0 --seed=0 -t deim_dfine_hgnetv2_n_coco_160e.pth
```
Trained weights will be saved in the deim_outputs/ directory.

## 8. Testing Instructions:
Modify your [batter_detection.yaml].

```yaml
val_dataloader:
type: DataLoader
dataset:
    type: CocoDetection
    img_folder: datasets\battery-test\images  
    ann_file: datasets\battery-test\annotations\test.json ## Modify the path to the test set
    return_masks: False
    transforms:
    type: Compose
    ops: ~
shuffle: False
num_workers: 4
drop_last: False
collate_fn:
    type: BatchImageCollateFunction
```

```shell
python train.py -c configs/deim_dfine/deim_hgnetv2_n_battery.yml --test-only -r deim_outputs/deim_hgnetv2_n_neu_base_200e/best_stg2.pth
```
The results of the coco indicators will be displayed on the terminal.
    
```shell
python tools/benchmark/get_info.py -c configs/test/deim_hgnetv2_n_battery.yml
```
It will display FLOPs, MACs and Params on the terminal.

## 9. Output Description
Upon running the model, the expected outputs include:
- **Detected bounding boxes for defects on battery images.** 
- **Classification labels for each detected defect (Id, Ei, Le, Sq, Bd).** 
- **Performance metrics (AP, FPS) displayed in the terminal.** 

## 10. Acknowledgement
This work builds upon the DEIM framework. For more details on DEIM, please refer to the official repository: [DEIM on GitHub](https://github.com/Intellindust-AI-Lab/DEIM)

For any questions regarding the BSD dataset or the DAEF model, please contact: [DAEF](https://github.com/sober-white/DAEF)

✨ Feel free to contribute and reach out if you have any questions! ✨



