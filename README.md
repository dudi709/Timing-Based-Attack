# The Adversarial Implications of Variable-Time Inference
<!-- This project contains the implementation of our ??? 2023 paper [link](https://www.google.com).<br/> -->
This work presents novel findings by demonstrating the potential to enhance a decision-based attack.
Our adversary simply measures the execution time of an algorithm applied to post-process the predictions of the ML model under attack.
We focus our investigation on leakage in the NMS algorithm, ubiquitous in object detectors. We demonstrate attackers against the YOLOv3 detector, that use timing to evade object detection with adversarial examples.
## Evasion Attack Using Timing Leakage
<img src="https://github.com/dudi709/Adversarial-Implications-Variable-Time-Inference/blob/main/doc/algo.png" width="300">
<br/>
Our adversary wishes to evade detection by performing adversarial perturbations on an image.

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
2.2. Place the images you want to attack into `COCO/`. You can select any image you want with the png or jpg extension (to add more options you need to make code changes to the main file). Our recommendation, work with images from the [COCO-MS](https://cocodataset.org/#download) dataset.

3. Run demo:
```
python main.py
```
After running the program you can find your outputs under the `time_attack_samples/` directory.

### Citations
If you find our work useful in your research, please consider citing:
```
@inproceedings{biton2023adversarial,
  title={The Adversarial Implications of Variable-Time Inference},
  author={Biton, Dudi and Misra, Aditi and Levy, Efrat and Kotak, Jaidip and Bitton, Ron and Schuster, Roei and Papernot, Nicolas and Elovici, Yuval and Nassi, Ben},
  booktitle={Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security},
  pages={103--114},
  year={2023}
}
```

### License
Distributed under the MIT License. See the [LICENSE](/LICENSE.txt) for more information.
