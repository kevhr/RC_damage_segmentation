# make an script to count pixels and determine the area of spalls for one image
#
# Path: pixel_counter_2.py
# Compare this snippet from pixel_counter_w.py:
import torch
from torchvision import transforms
from PIL import Image
import os
from ultralytics import YOLO
import cv2
import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import random

# Use Tensor.cpu() to copy the tensor to host memory first


## the task is generatea script that counts segmented pixels for each class form a image with pretined model YOLOv8

# 1. load model
model = YOLO('YOLOv8_model/YOLO_spall_model.pt')
model.conf = 0.5


def load_preprocess_image(img):
    im = Image.open(img)

    # verify width is more than height
    width, height = im.size
    if width < height:
        im = im.transpose(Image.ROTATE_90)

    # resize
    if im.size[0] > 512 or im.size[1] > 512:
        im = im.resize((384, 512))

    image = np.array(im)
    return image


def load_process_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image


def predict_segmentation_mask(image_path):
    print(image_path)
    # 5.2 predict
    image = load_preprocess_image(image_path)
    results = model.predict(image)
    H, W, _ = image.shape

    n_classes = results[0].masks.data.shape
    classes_predicted = results[0].boxes.cls

    return results, H, W, n_classes, classes_predicted


def count_pixels(results, H, W, n_classes, classes_predicted):
    # 5.3 count pixels for each class
    # create a dictionary to store the results

    spall_pixels = 0
    square_pixels = 0
    rectangle_pixels = 0

    spall_diameter = 0
    pt_f = (0, 0)
    mask_total = np.zeros((H, W))
    print(classes_predicted)
    for i, i_class in enumerate(classes_predicted):

        if i_class == 1:
            # spall
            spall_pixels += results[0].masks.data[i].cpu().numpy().sum()
            mask_total += results[0].masks.data[i].cpu().numpy()
            # calculate diameter
            pt1 = (int(results[0].boxes.xyxy[i][0]), int(results[0].boxes.xyxy[i][1]))
            pt2 = (int(results[0].boxes.xyxy[i][2]), int(results[0].boxes.xyxy[i][3]))
            pt = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
            # diameter = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

            diameter_x = np.abs(pt1[0] - pt2[0])
            diameter_y = np.abs(pt1[1] - pt2[1])
            diameter = np.max([diameter_x, diameter_y])
            # calculate the center of the spall
            if diameter > spall_diameter:
                spall_diameter = diameter
                pt_f = pt

        elif i_class == 2:
            # square
            square_pixels += results[0].masks.data[i].cpu().numpy().sum()
        elif i_class == 0:
            # rectangle
            rectangle_pixels += results[0].masks.data[i].cpu().numpy().sum()

    mask_total = mask_total.astype(np.uint8)

    # calculate the area
    # pattern  areas
    rectangle_area = 10 * 30  # 300 mm2
    square_area = 20 * 20  # 400 mm2

    if square_pixels > rectangle_pixels and square_pixels > 0:
        ratio_square = square_area / square_pixels
        spall_area = spall_pixels * ratio_square
        len_ratio_square = np.sqrt(ratio_square)
        spall_diameter_predicted = len_ratio_square * spall_diameter

    elif rectangle_pixels > square_pixels and rectangle_pixels > 0:
        ratio_rectangle = rectangle_area / rectangle_pixels
        spall_area = spall_pixels * ratio_rectangle
        len_ratio_rectangle = np.sqrt(ratio_rectangle)
        spall_diameter_predicted = len_ratio_rectangle * spall_diameter

    else:
        spall_area = 0
        spall_diameter_predicted = 0

    return spall_pixels, square_pixels, rectangle_pixels, spall_diameter, pt_f, mask_total, spall_area, spall_diameter_predicted


def get_mask(image_path, mask_total, pt_f, spall_diameter):
    # plot the mask
    mask_total = mask_total * 255
    mask_total = mask_total.astype(np.uint8)

    image = load_preprocess_image(image_path)
    mask = cv2.circle(mask_total, pt_f, int(spall_diameter / 2), (255, 255, 255), 2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.bitwise_and(image, image, mask=mask)
    mask = cv2.addWeighted(mask, 0.75, image, 0.2, 0.0)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    return mask


def plot_predicted_mask(original_img, predicted_mask):
    """
    Inputs: image and mask 
    Outputs: plot both image and plot side by side
    """
    image = load_preprocess_image(original_img)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 5))
    axes[0].imshow(image)
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[0].set_title('Original data')
    axes[1].imshow(predicted_mask)
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[1].set_title('Predicted Mask')
    fig.tight_layout()
    # generate a name for the file mask_ + original image name
    # filename = 'mask_' + str(original_img)[:-4] + '.png'

    filename = 'pair' + str(random.randint(100,1000)) + str(random.randint(100,1000) ) + '.png'
    plt.savefig(filename)

    print('File saved successully')

    #filename = open(filename)
    print(filename)

    return filename, fig


def final_function(image_path):
    results, H, W, n_classes, classes_predicted = predict_segmentation_mask(image_path)
    spall_pixels, square_pixels, rectangle_pixels, spall_diameter, pt_f, mask_total, spall_area, spall_diameter_predicted = count_pixels(
        results, H, W, n_classes, classes_predicted)
    mask = get_mask(image_path, mask_total, pt_f, spall_diameter)
    filename, fig = plot_predicted_mask(image_path, mask)

    return filename, spall_pixels, square_pixels, rectangle_pixels, spall_diameter, pt_f, mask_total, spall_area, spall_diameter_predicted