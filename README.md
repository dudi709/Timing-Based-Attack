# The Adversarial Implications of Variable-Time Inference
This project contains the implementation of our ??? 2023 paper [link](https://www.google.com).<br/>
In this work, we present novel findings by demonstrating the potential to enhance a decision-based attack.
Our adversary simply measures the execution time of an algorithm applied to post-process the predictions of the ML model under attack.
We focus our investigation on leakage in the NMS algorithm, which is ubiquitous in object detectors. We demonstrate attackers against the YOLOv3 detector, that use timing to evade object detection with adversarial examples or perform membership inference.
## Evasion Attack Using Timing Leakage

Our adversary wishes to evade detection via performing adversarial pertubations on an image.
<br/>
Our attacker employs an iterative genetic algorithm, where in each iteration the algorithm draws instances (a population) near the object.
The instances are sent to the object detector for inference, and their execution times are used as fitness functions to approximate the quality of perturbations according to the following the principle that images with low execution times are closer to evasion and are therefore more "fit".
The instances in the population are then bred to create a new mutation.

### Install

We recommend using conda to install the required libraries:
<br/>
1. Setup conda:
```
conda deactivate
conda create --name timing_attack python=3.6
conda activate timing_attack
```
2. Other dependencies:
```
pip install tensorflow==2.0.0
pip install keras==2.3.0
pip install matplotlib==3.2.2
pip install pillow==7.2.0
pip install scipy==1.1.0
pip install h5py==2.10.0
```

### Quick Demo

1. Set the folders for placing the pre-trained model:
```
cd Adversarial-Implications-Variable-Time-Inference
mkdir YOLO
mkdir YOLO/model
```
1.1. Please download the pre-trained model from the following link and put it into `YOLO/model/`:
- [yolov3 model](https://drive.google.com/file/d/19XC9ujio7AwpT52tcWiUmaoxoDWdjrQw/view?usp=sharing) trained on COCO dataset.

2. Set the folder for placing the images you want to attack:
```
mkdir COCO
```
2.2. Place the images you want to attack into `COCO/`. You can select any image you want with the png or jpg extension (to add more options you need to make code changes to the main file). Our recommendation, work with images from [COCO-MS](https://cocodataset.org/#download) dataset.

3. Run demo:
```
python main.py
```
After running the program you can find your outputs under `time_attack_samples/` directory.

### Citation
If you find the project useful for your research, please cite:
```
@article{???,
  title={The Adversarial Implications of Variable-Time Inference},
  author={????},
  journal={????},
  year={???}
}
```