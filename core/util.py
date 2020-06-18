import os
import torch


class Logger():
    def __init__(self, targetFile, separator=';'):
        self.targetFile = targetFile
        self.separator = separator

    def writeHeaders(self, headers):
        with open(self.targetFile, 'a') as fh:
            for aHeader in headers:
                fh.write(aHeader + self.separator)
            fh.write('\n')

    def writeDataLog(self, dataArray):
        with open(self.targetFile, 'a') as fh:
            for dataItem in dataArray:
                fh.write(str(dataItem) + self.separator)
            fh.write('\n')

def configure_backbone(backbone, size, pretrained_arg=True, num_classes_arg=2):
    return backbone(pretrained=pretrained_arg, rgb_stack_size=size,
                    num_classes=num_classes_arg)

def configure_backbone_forward_phase(backbone, pretrained_weights_path, size, pretrained_arg=True, num_classes_arg=2):
    return backbone(pretrained_weights_path, rgb_stack_size=size,
                    num_classes=num_classes_arg)

def load_train_video_set():
    files = os.listdir('.../AVA/csv/train')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos

def load_val_video_set():
    files = os.listdir('.../AVA/csv/val')
    videos = [f[:-18] for f in files]
    videos.sort()
    return videos
