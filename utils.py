from PIL import Image, ImageStat
from keras.preprocessing.image import img_to_array
import numpy as np


def is_grayscale(path):
    im = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(im)
    # check the avg with any element value
    if sum(stat.sum) / 3 == stat.sum[0]:
        # if grayscale
        return True
    else:
        # else its colour
        return False


def crop_gadget(photo_path, object_location, deviation=0):
    # Opens a image in RGB mode
    im = Image.open(photo_path)

    # Setting the points for cropped image
    left = object_location[0].xmin - deviation
    top = object_location[0].ymin - deviation
    right = object_location[0].xmax + deviation
    bottom = object_location[0].ymax + deviation

    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))
    return im1


def reshape_picture(img, new_width, new_height):
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    return img


def normalize_picture(image_as_array, direction):
    # scale pixel values to [0, 1]
    image_as_array = image_as_array.astype('float32')
    if direction == 'down':
        image_as_array /= 255.0
    elif direction == 'up':
        image_as_array *= 255.0
    return image_as_array


def prepare_img_for_yolo(img, input_w, input_h):
    # prepare an image of a suitable size for yolo
    new_img = reshape_picture(img, input_w, input_h)
    new_img = img_to_array(new_img)
    # scale pixel values to [0, 1]
    new_img = normalize_picture(new_img, 'down')
    # add a dimension so that we have one sample
    new_img = np.expand_dims(new_img, 0)
    return new_img




