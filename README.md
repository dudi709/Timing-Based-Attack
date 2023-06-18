# The Adversarial Implications of Variable-Time Inference
## Evasion Using Timing Leakage

Our adversary wishes to evade detection via performing adversarial pertubations on an image.
We consider a black-box setting where the adversary can send an image to the object detector, receive a response, and measure the latency corresponding to the execution time of the inference procedure.
Specifically, the attacker does not have an access to the detector's weights, architecture, or prediction confidence values. 
<br/>
Our attacker employs an iterative genetic algorithm, where in each iteration the algorithm draws instances (a population) near the object.
The instances are sent to the object detector for inference, and their execution times are used as fitness functions to approximate the quality of perturbations according to the following the principle that images with low execution times are closer to evasion and are therefore more "fit".
The instances in the population are then bred to create a new mutation.

## Requirements

Please install the Anaconda virtual environment we used:
<br/>
cd into the TimingAttack directory and run:

```setup
conda deactivate
conda env create --file environment.yaml
conda activate yolo_env
```
## Pre-trained Model

You can download the pre-trained model here:

- [yolov3 model](https://drive.google.com/file/d/1ws5rxG4mMF2qTQy5Hb0xHDwcyo_cYWdc/view) trained on COCO dataset. 

## Prerequisites for running

Before executing main.py file:
- Place the images you want to attack under `COCO/` directory. You can select any image you want with the png or jpg extension (to add more options you need to make code changes to the main file). Our recommendation, work with images from [COCO](https://cocodataset.org/#download).
- Please make sure you have downloaded the pre-trained model from "Pre-trained Model" section and placed it under `YOLO/model/` directory. <br/><br/>
cd into the TimingAttack directory and run:
```
python main.py
```
After running the program you can find your outputs under `time_attack_samples/` directory.
