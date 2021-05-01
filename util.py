# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def load_image(path):
    img = img2rgb(path)  # RGB
    assert img is not None, 'Image Not Found ' + path
    return img


def img2rgb(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)

    img = img / 255

    img = np.array(img).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_predictions(images, predictions, paths):
    scores, predictions = torch.max(predictions, dim=1)

    images = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    scores = scores.cpu().numpy()

    for img, score, pred, path in zip(images, scores, predictions, paths):
        img = img.transpose(1, 2, 0)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.suptitle(f'{path}')
        ax.set_title(f'label={pred}:{score}')
        ax.axis('off')
        ax.imshow(img)
        plt.show()

def save_predictions(images, predictions, paths):
    scores, predictions = torch.max(predictions, dim=1)

    images = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    scores = scores.cpu().numpy()

    for img, score, pred, path in zip(images, scores, predictions, paths):
        img = img.transpose(1, 2, 0)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.suptitle(f'{path}')
        ax.set_title(f'label={pred}:{score}')
        ax.axis('off')
        ax.imshow(img)
        plt.savefig(str(Path(path)))