# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

from albumentations.pytorch.transforms import ToTensorV2
import torch
import argparse
import albumentations as A

from config import parse_detect_config, DEVICE, read_yaml_config
from model import Model
from util import plot_predictions, save_predictions
from general import load_checkpoint
from dataset import LoadImages


def generatePredictions(model, dataset):
    for img, path in dataset:
        with torch.no_grad():
            img = img.to(DEVICE, non_blocking=True).unsqueeze(0)

            predictions = model(img)
            yield img, predictions, path


def detect(model=None, config=None):
    torch.backends.cudnn.benchmark = True

    config = parse_detect_config() if not config else config

    transform = A.Compose(
        [
            A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
            ToTensorV2(),
        ]
    )

    dataset = LoadImages(config.JSON, transform=transform)

    if not model:
        model = Model()
        model = model.to(DEVICE)
        _, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    model.eval()
    for img, predictions, path in generatePredictions(model, dataset):
        plot_predictions(img, predictions, [path])
        save_predictions(img, predictions, [path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run inference on model')
    parser.add_argument('--detect', type=str, default="detect.yaml", help='detect config file')
    opt = parser.parse_args()

    config_detect = parse_detect_config(read_yaml_config(opt.detect))
    
    detect(config=config_detect)