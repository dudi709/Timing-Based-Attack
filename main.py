from yolov3 import load_yolo_model, load_image_pixels, predict_with_yolo
from utils import is_grayscale, crop_gadget, prepare_img_for_yolo
from timing_based_evasion import timing_based_evasion
from pathlib import Path
import numpy as np
import scipy
import argparse
import os


def extract_gadgets_from_img(model, photo_path, input_w, input_h):
    # load and prepare image for yolo
    image, image_wI, image_hI = load_image_pixels(photo_path, (input_w, input_h))
    # get the details of the detected objects
    v_boxes, v_labels, v_scores, yolo_runtime, nms_runtime, bb_counter = predict_with_yolo(model,
                                                                                           image,
                                                                                           image_wI,
                                                                                           image_hI,
                                                                                           input_w,
                                                                                           input_h)

    print(f'detected objects before cropping: {v_labels}')

    gadget_lst = list()
    # go over each label
    for i in range(len(v_labels)):
        # get cropped gadget from image
        gadget = crop_gadget(photo_path, [v_boxes[i]], 10)
        gadget_w, gadget_h = gadget.width, gadget.height
        original_gadget_label = v_labels[i]
        gadget_id = i
        # prepare cropped object for yolo
        test_cropped_gadget = prepare_img_for_yolo(gadget.copy(), input_w, input_h)
        # send cropped gadget to yolo
        test_gadget_v_boxes, test_gadget_v_labels, test_gadget_v_scores, test_gadget_yolo_runtime, \
        test_gadget_nms_runtime, test_gadget_bb_counter = predict_with_yolo(model,
                                                                            test_cropped_gadget,
                                                                            gadget_w,
                                                                            gadget_h,
                                                                            input_w,
                                                                            input_h)
        # check if the object is still detected after cropping
        if len(test_gadget_v_labels) == 1:
            if test_gadget_v_labels[0] == original_gadget_label:
                gadget_lst.append((gadget, original_gadget_label, [v_boxes[i]], gadget_w, gadget_h, gadget_id))
            else:
                print(f'gadget-{gadget_id} ({original_gadget_label}) is not detected after cropping')
        else:
            print(f'gadget-{gadget_id} ({original_gadget_label}) is not detected after cropping')
    # returns a list of objects detected after cropping with information about each gadget
    return gadget_lst


def attack(args):
    model_path = os.path.join('YOLO', 'model', args.model_file)
    model = load_yolo_model(model_path)

    for filename in os.listdir(args.dataset_name):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            photo_path = os.path.join(args.dataset_name, filename)
            if is_grayscale(photo_path) is not True:
                print(f'attack image: {filename}')
                # define the expected input shape for the model
                input_w, input_h = 416, 416
                # get list of gadgets from image
                gadget_lst = extract_gadgets_from_img(model, photo_path, input_w, input_h)

                for i in range(len(gadget_lst)):
                    gadget_id = gadget_lst[i][5]
                    filename_without_extension = filename[0: filename.index(".")]
                    gadget_num = f'gadget_{gadget_id}'
                    img_id = f'{filename_without_extension}_{gadget_num}'
                    # create a folder to save samples
                    save_files_dir = f'time_attack_samples/Attack - delta={args.delta} ' \
                                     f'max_iteartions={args.max_iterations} epsilon={args.epsilon} ' \
                                     f'population={args.population}/{filename_without_extension}/'
                    Path(save_files_dir).mkdir(parents=True, exist_ok=True)

                    gadget = gadget_lst[i][0]
                    original_gadget_label = gadget_lst[i][1]
                    gadget_location = gadget_lst[i][2]
                    gadget_wI = gadget_lst[i][3]
                    gadget_hI = gadget_lst[i][4]

                    # defining factors for the amplified gadget
                    factorW = int(input_w / gadget_wI)
                    factorH = int(input_h / gadget_hI)
                    if factorW == 0 and factorH == 0:
                        factorW = 1
                        factorH = 1
                    elif factorW == 0:
                        factorW = 1
                    elif factorH == 0:
                        factorH = 1

                    print(f'Attack gadget: {gadget_num} - {original_gadget_label}')

                    normalized_org_gadget, perturbed_gadget = timing_based_evasion(model,
                                                                                   gadget.copy(),
                                                                                   original_gadget_label,
                                                                                   gadget_wI, gadget_hI,
                                                                                   factorW, factorH,
                                                                                   input_w, input_h,
                                                                                   clip_max=1,
                                                                                   clip_min=0,
                                                                                   delta=args.delta,
                                                                                   epsilon=args.epsilon,
                                                                                   max_iterations=args.max_iterations,
                                                                                   population=args.population,
                                                                                   channels=args.channels,
                                                                                   gradf_epsilon=args.gradf_epsilon)

                    image = np.concatenate([normalized_org_gadget, np.zeros((gadget_hI, 8, 3)), perturbed_gadget], axis=1)
                    scipy.misc.imsave(f'{save_files_dir}{gadget_num}.png', image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--delta', type=float,
                        default=25.0)

    parser.add_argument('--epsilon', type=float,
                        default=0.5)

    parser.add_argument('--population', type=float,
                        default=20)

    parser.add_argument('--max_iterations', type=int,
                        default=300)

    parser.add_argument('--channels', type=int,
                        default=3)

    parser.add_argument('--gradf_epsilon', type=float,
                        default=0.001)

    parser.add_argument('--model_file', type=str,
                        choices=['yolo_model.h5'],
                        default='yolo_model.h5')

    parser.add_argument('--dataset_name', type=str,
                        choices=['COCO'],
                        default='COCO')

    args = parser.parse_args()
    attack(args)
