from keras.preprocessing.image import img_to_array, array_to_img
from utils import normalize_picture, prepare_img_for_yolo
from yolov3 import predict_with_yolo
from PIL import Image
import numpy as np


def timing_based_evasion(model,
                         gadget,
                         original_gadget_label,
                         gadget_wI, gadget_hI,
                         factorW, factorH,
                         input_w, input_h,
                         clip_max=1,
                         clip_min=0,
                         delta=25.0,
                         epsilon=0.5,
                         max_iterations=300,
                         population=20,
                         channels=3,
                         gradf_epsilon=0.001):
    """
    Main algorithm for Evasion Using Timing Leakage Attack.

    :param model: the model that allows us to make predictions.
    :param gadget: the original gadget as image.
    :param original_gadget_label: the label of the original gadget.
    :param gadget_wI: width of the original gadget.
    :param gadget_hI: height of the original gadget.
    :param factorW: width of the amplified gadget.
    :param factorH: height of the amplified gadget.
    :param input_w: define the expected width shape for the model.
    :param input_h: define the expected height shape for the model.
    :param clip_max: upper bound of the image.
    :param clip_min: lower bound of the image.
    :param delta: the radius around the gadget
    :param epsilon: learning-rate parameter.
    :param max_iterations: sets the attack to be limited to a certain amount of iterations.
    :param population: sample size (amount of lotteries drawn).
    :param channels: the amount of channels in the image.
    :param gradf_epsilon: helps prevent division by 0 before receiving the gradient.
    :return gadgets_dict[0]: original gadget as array.
    :return gadgets_dict[iteration]: original gadget as array.
    """

    gadget_as_array = img_to_array(gadget)
    # scale pixel values to [0, 1]
    gadget_as_array = normalize_picture(gadget_as_array, 'down')
    gadgets_dict = {0: np.copy(gadget_as_array)}

    params = {'gadget_shape': gadget_as_array.shape,
              'original_gadget_label': original_gadget_label,
              'gadget_wI': gadget_wI, 'gadget_hI': gadget_hI,
              'factorW': factorW, 'factorH': factorH,
              'input_w': input_w, 'input_h': input_h,
              'clip_max': clip_max, 'clip_min': clip_min,
              'delta': delta,
              'epsilon': epsilon,
              'max_iterations': max_iterations,
              'population': population,
              'channels': channels,
              'gradf_epsilon': gradf_epsilon}

    for iteration in range(params['max_iterations']):
        # compute distance from the original gadget.
        l2_gadget = np.linalg.norm(gadgets_dict[iteration] - gadgets_dict[0])
        l_inf_gadget = np.max(abs(gadgets_dict[iteration] - gadgets_dict[0]))
        mse_gadget = mse(gadgets_dict[iteration], gadgets_dict[0])

        if iteration == params['max_iterations'] - 1:
            print(f'Exceeding the amount of iterations. L2 = {l2_gadget}')
            return gadgets_dict[0], gadgets_dict[iteration]

        # prepare gadget for yolo
        temp_gadget = prepare_img_for_yolo(array_to_img(np.copy(gadgets_dict[iteration])), input_w, input_h)
        # send gadget to yolo
        v_boxes_temp_gadget, v_labels_temp_gadget, v_scores_temp_gadget, yolo_runtime_temp_gadget, nms_runtime_temp_gadget, \
        bb_counter_temp_gadget = predict_with_yolo(model,
                                                   temp_gadget,
                                                   params['gadget_wI'],
                                                   params['gadget_hI'],
                                                   params['input_w'],
                                                   params['input_h'])

        # check if the gadget is still detected by yolo
        if len(v_labels_temp_gadget) == 0:
            print(f'Found adversarial example in {iteration} iterations - gadget was not detected. L2 = {l2_gadget}')
            return gadgets_dict[0], gadgets_dict[iteration]
        elif params['original_gadget_label'] not in v_labels_temp_gadget:
            print(f'Found adversarial example in {iteration} iterations - gadget label is different from the original '
                  f'label. L2 = {l2_gadget}')
            return gadgets_dict[0], gadgets_dict[iteration]

        # if there is a need to amplify
        if params['factorW'] != 1 or params['factorH'] != 1:
            # chain the gadget to itself (factorW x factorH) times
            amplified_gadget = amplify(array_to_img(np.copy(gadgets_dict[iteration])), params['factorW'],
                                       params['factorH'])
            amplified_gadget_w, amplified_gadget_h = amplified_gadget.width, amplified_gadget.height
            # prepare amplified gadget for yolo
            amplified_gadget = prepare_img_for_yolo(amplified_gadget, input_w, input_h)
            # send amplified gadget to yolo
            v_boxes_amplified_gadget, v_labels_amplified_gadget, v_scores_amplified_gadget, \
            yolo_runtime_amplified_gadget, nms_runtime_amplified_gadget, \
            bb_counter_amplified_gadget = predict_with_yolo(model,
                                                            amplified_gadget,
                                                            amplified_gadget_w,
                                                            amplified_gadget_h,
                                                            params['input_w'],
                                                            params['input_h'])

            print(f'Gadget[{iteration}]: L2 = {l2_gadget}, yolo_runtime = {yolo_runtime_amplified_gadget}')

            # draw #sample size noise vectors
            noise_vectors = drawBatchInRange((params['gadget_hI'], params['gadget_wI'], params['channels']),
                                             params['population'])
            # normalize noise vectors
            normalized_noise = normalize_noise(noise_vectors)
            neighbour_with_noise = np.ndarray(shape=normalized_noise.shape)
            clipped_neighbour_with_noise = np.ndarray(shape=normalized_noise.shape)
            actual_noise = np.ndarray(shape=normalized_noise.shape)
            decisions = np.ndarray(shape=normalized_noise.shape[0])
            amplified_neighbour_runtime = np.ndarray(shape=normalized_noise.shape[0])

            for index in range(len(normalized_noise)):
                neighbour_with_noise[index] = gadgets_dict[iteration] + (delta * normalized_noise[index])
                clipped_neighbour_with_noise[index] = clip_image(neighbour_with_noise[index], params['clip_min'],
                                                                 params['clip_max'])
                normalized_noise[index] = (clipped_neighbour_with_noise[index] - gadgets_dict[iteration]) / delta
                actual_noise[index] = clipped_neighbour_with_noise[index] - gadgets_dict[iteration]

                # compute distance from the original gadget.
                l2_neighbour = np.linalg.norm(clipped_neighbour_with_noise[index] - gadgets_dict[0])
                l_inf_neighbour = np.max(abs(clipped_neighbour_with_noise[index] - gadgets_dict[0]))
                mse_neighbor = mse(clipped_neighbour_with_noise[index], gadgets_dict[0])

                # chain the neighbour to itself (factorW x factorH) times
                amplified_neighbour = amplify(array_to_img(np.copy(clipped_neighbour_with_noise[index])),
                                              params['factorW'], params['factorH'])
                amplified_neighbour_w, amplified_neighbour_h = amplified_neighbour.width, amplified_neighbour.height
                # prepare amplified neighbour for yolo
                amplified_neighbour = prepare_img_for_yolo(amplified_neighbour, input_w, input_h)
                # send amplified neighbour to yolo
                v_boxes_amplified_neighbour, v_labels_amplified_neighbour, v_scores_amplified_neighbour, \
                yolo_runtime_amplified_neighbour, nms_runtime_amplified_neighbour, \
                bb_counter_amplified_neighbour = predict_with_yolo(model,
                                                                   amplified_neighbour,
                                                                   amplified_neighbour_w,
                                                                   amplified_neighbour_h,
                                                                   params['input_w'],
                                                                   params['input_h'])
                # amplified neighbor run time storage
                amplified_neighbour_runtime[index] = yolo_runtime_amplified_neighbour

                print(
                    f'Gadget[{iteration}][{index}]: L2 = {l2_neighbour}, yolo_runtime = {yolo_runtime_amplified_neighbour}')

                # decision is based on the runtime of the amplified neighbor and amplified gadget in the same
                # iteration when the amplified neighbor run time is higher than the amplified gadget run time we do
                # not want to go in that direction
                if yolo_runtime_amplified_neighbour >= yolo_runtime_amplified_gadget:
                    decisions[index] = -1.0
                else:
                    decisions[index] = 1.0
            decision_shape = [len(decisions)] + [1] * len(params['gadget_shape'])
            fval = decisions.astype(float).reshape(decision_shape)

            # update delta for the next iteration
            if np.mean(fval) > (1 / 3):
                params['delta'] = params['delta'] / 2
            elif np.mean(fval) < -(1 / 3):
                params['delta'] = params['delta'] * 2

            # fitness for population-member
            delta_time = abs(amplified_neighbour_runtime - yolo_runtime_amplified_gadget)
            fitness = delta_time / np.sum(delta_time)

            # Baseline subtraction (when fval differs)
            if np.mean(fval) == 1.0:
                gradf = np.mean(np.transpose((np.transpose(normalized_noise) * fitness)), axis=0)
            elif np.mean(fval) == -1.0:
                gradf = - np.mean(np.transpose((np.transpose(normalized_noise) * fitness)), axis=0)
            else:
                fval -= np.mean(fval)
                gradf = np.mean(np.transpose(
                    (np.transpose(fval) * np.transpose(normalized_noise) * fitness)), axis=0)

            if np.linalg.norm(gradf) == 0:
                gradf += params['gradf_epsilon']
            # Get the gradient direction
            gradf = gradf / np.linalg.norm(gradf)

            # Update the sample for the next iteration
            new_instance = gadgets_dict[iteration] + (gradf * epsilon)
            new_instance = clip_image(new_instance, params['clip_min'], params['clip_max'])
            gadgets_dict[iteration + 1] = new_instance


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def amplify(img, factorW, factorH):
    # get the width and height of img
    w, h = img.size
    # concatenating image horizontal
    new_im1 = Image.new('RGB', (w * factorW, h))
    x_offset = 0
    for im in range(0, factorW):
        new_im1.paste(img, (x_offset, 0))
        # each iteration we want to connect the image to the end of the previous image (horizontally)
        x_offset += img.size[0]

    # concatenating vertical
    # get the width and height of new_img1 (image concatenate horizontally)
    w, h = new_im1.size
    new_im2 = Image.new('RGB', (w, h * factorH))
    y_offset = 0
    for im in range(0, factorH):
        new_im2.paste(new_im1, (0, y_offset))
        # Each iteration we want to connect the image to the end of the previous image (vertically)
        y_offset += img.size[1]
    return new_im2


def drawBatchInRange(shape, times):
    (w, h, channels) = shape
    noise = np.random.uniform(low=-1, high=1, size=(times, w, h, channels))
    return noise


def normalize_noise(original_noise):
    rv = original_noise / np.sqrt(np.sum(original_noise ** 2, axis=(1, 2, 3), keepdims=True))
    return rv


def clip_image(image, clip_min, clip_max):
    # clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max)
