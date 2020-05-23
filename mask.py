# Import Modules
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import visualize
import Mask_RCNN.mrcnn.model as modellib
import tensorflow as tf
import numpy as np
import warnings
import json
import cv2
import os


def ignore_warnings():
    # Ignore warnings
    old_v = tf.compat.v1.logging.get_verbosity()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings(action='ignore')


class SingleMasking:
    def __init__(self, img_dir, img_size, threshold, gpu_count, images_per_gpu):
        # Configuration
        self.MODEL_DIR = "Source/mask_rcnn_fashion_0006.h5"
        self.LABEL_DIR = "Source/label_descriptions.json"
        self.MASK_DIR = "Mask_RCNN"
        self.IMG_DIR = img_dir
        self.NUM_CATS = 46
        self.IMAGE_SIZE = img_size
        ignore_warnings()

        # From label_descriptions['categories'] to label_names
        with open(self.LABEL_DIR) as f:
            self.label_descriptions = json.load(f)
        self.label_names = [x['name'] for x in self.label_descriptions['categories']]

        # Setup Configuration
        class InferenceConfig(Config):
            NAME = "fashion"
            NUM_CLASSES = self.NUM_CATS + 1  # +1 for the background class
            GPU_COUNT = 1
            IMAGES_PER_GPU = 4
            BACKBONE = 'resnet101'
            IMAGE_MIN_DIM = self.IMAGE_SIZE
            IMAGE_MAX_DIM = self.IMAGE_SIZE
            IMAGE_RESIZE_MODE = 'none'
            RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
            DETECTION_MIN_CONFIDENCE = threshold
            if gpu_count == "":
                GPU_COUNT = 1
            else:
                GPU_COUNT = gpu_count
            if images_per_gpu == "":
                IMAGES_PER_GPU = 1
            else:
                IMAGES_PER_GPU = images_per_gpu

        # Execute Inference Configuration
        self.inference_config = InferenceConfig()
        # Load Weight File
        self.model = modellib.MaskRCNN(mode='inference', config=self.inference_config, model_dir=self.MASK_DIR)
        self.model.load_weights(self.MODEL_DIR, by_name=True)

        # Resize Image from image_path
        def resize_image(image_path):
            temp = cv2.imread(image_path)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp, (self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            return temp

        # Since the submission system does not permit overlapped masks, we have to fix them
        def refine_masks(masks, rois):
            areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
            mask_index = np.argsort(areas)
            union_mask = np.zeros(masks.shape[:-1], dtype=bool)
            for m in mask_index:
                masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
                union_mask = np.logical_or(masks[:, :, m], union_mask)
            for m in range(masks.shape[-1]):
                mask_pos = np.where(masks[:, :, m] == True)
                if np.any(mask_pos):
                    y1, x1 = np.min(mask_pos, axis=1)
                    y2, x2 = np.max(mask_pos, axis=1)
                    rois[m, :] = [y1, x1, y2, x2]
            return masks, rois

        # Python code to remove duplicate elements
        def remove(duplicate):
            final_list = []
            duplicate_list = []
            for num in duplicate:
                if num not in final_list:
                    final_list.append(num)
                else:
                    duplicate_list.append(num)
            return final_list, duplicate_list

        # Single Image Masking
        self.img = cv2.imread(self.IMG_DIR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.result = self.model.detect([resize_image(self.IMG_DIR)], verbose=1)
        self.r = self.result[0]
        if self.r['masks'].size > 0:
            self.masks = np.zeros((self.img.shape[0], self.img.shape[1], self.r['masks'].shape[-1]), dtype=np.uint8)
            for m in range(self.r['masks'].shape[-1]):
                self.masks[:, :, m] = cv2.resize(self.r['masks'][:, :, m].astype('uint8'),
                                            (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_NEAREST)
            y_scale = self.img.shape[0] / self.IMAGE_SIZE
            x_scale = self.img.shape[1] / self.IMAGE_SIZE
            rois = (self.r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)
            masks, rois = refine_masks(self.masks, rois)
        else:
            masks, rois = self.r['masks'], self.r['rois']

        visualize.display_instances(self.img, rois, masks, self.r['class_ids'],
                                    ['bg'] + self.label_names, self.r['scores'],
                                    title='camera1', figsize=(12, 12))
        visualize.display_top_masks(self.img, self.masks, self.r['class_ids'], self.label_names, limit=8)
