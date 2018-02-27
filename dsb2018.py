import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import label

from config import Config
import utils
import model as modellib
import visualize
from model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
class DSB2018Config(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "DSB2018"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

class DSB2018Dataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_DSB2018(self, data_dir, set_name, config=None):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("DSB2018", 1, "nuclie")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        filenames = np.genfromtxt(set_name, dtype=str)
        for i in range(len(filenames)):
            self.add_image("DSB2018", image_id=i, path=None, channels=3, data_dir=data_dir, name=filenames[i])

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        data_dir = info['data_dir']
        name = info['name']
        channels = info['channels']
        
        imgs = os.listdir(os.path.join(data_dir, name, 'images'))
        img = plt.imread(os.path.join(data_dir, name, 'images', '%s' % (imgs[0])))
        img = img[:,:,:channels] * 256
        return img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        data_dir = info['data_dir']
        name = info['name']
        mask_files = os.listdir(os.path.join(data_dir, name, 'masks'))
        masks = []
        for i in range(len(mask_files)):
            mask = plt.imread(os.path.join(data_dir, name, 'masks', '%s' % (mask_files[i])))
            masks.append(mask)
            
        class_ids = np.ones(len(masks))
        masks = np.array(masks)
        masks = np.moveaxis(masks, 0, -1)
        return masks, class_ids.astype(np.int32)
    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on DSB2018")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config = DSB2018Config()
    config.display()
    
    
    if args.command == 'train':
        data_dir = '/home/htang6/data/dsb2018/stage1_train/'
        # Training dataset
        set_name = '/home/htang6/workspace/Mask_RCNN/filenames/filenames_train.csv'
        dataset_train = DSB2018Dataset()
        dataset_train.load_DSB2018(data_dir, set_name)
        dataset_train.prepare()

        # Validation dataset
        set_name = '/home/htang6/workspace/Mask_RCNN/filenames/filenames_val.csv'
        dataset_val = DSB2018Dataset()
        dataset_val.load_DSB2018(data_dir, set_name)
        dataset_val.prepare()
        
        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)


        # Which weights to start with?
        init_with = args.model  # imagenet, coco, or last

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last()[1], by_name=True)
        
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=1, 
                    layers='heads')

        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also 
        # pass a regular expression to select which layers to
        # train by name pattern.
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100, 
                    layers="all")
    elif args.command == 'evaluate':
        # Test dataset
        data_dir = '/home/htang6/data/dsb2018/stage1_test/'
        set_name = '/home/htang6/workspace/Mask_RCNN/filenames/filenames_test1.csv'
        dataset = DSB2018Dataset()
        dataset.load_DSB2018(data_dir, set_name)
        dataset.prepare()
        
        # Create model in evaluate mode
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=MODEL_DIR)


        # Which weights to start with?
        init_with = args.model  # imagenet, coco, or last

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last()[1], by_name=True)
        
        # Pick DSB2018 images from the dataset
        image_ids = dataset.image_ids

        # Get corresponding COCO image IDs.
        DSB2018_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

        t_prediction = 0
        t_start = time.time()

        results = []
        rles = []
        test_ids = []
        for i, image_id in enumerate(image_ids):
            # Load image
            image = dataset.load_image(image_id)
            info = dataset.image_info[image_id]
            name = info['name']
            
            # Run detection
            t = time.time()
            r = model.detect([image], verbose=0)[0]
            results.append(r)
            
            print('Time for predicting %s is %f\n' % (name, time.time() - t))
            
            masks = r['masks']
            
            reduced = []
            for i in range(masks.shape[-1]):
                mask = np.copy(masks[:,:,i])
                for j in range(len(reduced)):
                    intersection = mask & reduced[j]
                    if np.any(intersection):
                        mask -= intersection

                if np.any(mask):
                    reduced.append(mask)
                    
            for m in reduced:
                rles.append(rle_encoding(m))
            test_ids.extend([name] * len(reduced))
            
#            no merge or deduction
#             whole = np.zeros(shape=masks.shape[:2])
#             for i in range(masks.shape[-1]):
#                 whole = np.logical_or(whole, masks[:,:,i])
#             rle = list(prob_to_rles(whole))
#             rles.extend(rle)
#             test_ids.extend([name] * len(rle))
            
        sub = pd.DataFrame()
        sub['ImageId'] = test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        sub.to_csv('sub-dsbowl2018-1.csv', index=False)
            
        
        
        
