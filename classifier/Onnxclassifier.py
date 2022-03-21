import cv2
import os
import numpy as np
import time
from django.conf import settings

import logging

#################################################
logger=logging.getLogger("Classifier")
c_handler = logging.StreamHandler()

logger.setLevel('INFO')
c_format = logging.Formatter('%(asctime)s %(module)s %(levelname)s %(message)s')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
##################################################



class Classify():
    def __init__(self, label_path, model_path):
        
        base_classfier_models=f"{settings.BASE_DIR}{os.sep}classifier{os.sep}models"
        label_path=f"{base_classfier_models}{os.sep}{label_path}"
        model_path=f"{base_classfier_models}{os.sep}{model_path}"

        self.model = cv2.dnn.readNet(model=model_path)
        self.labels = self.load_labels(label_path)

    def load_labels(self, path):
        with open(path, 'r') as f:
            return {i: line.replace(str(i),"").strip() for i, line in enumerate(f.readlines())}

    
    def run(self, image):
        start_time=time.time()
        image = cv2.imread(image)
        # blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(128, 128), mean=(104, 117, 123))
        blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(128, 128), swapRB=True, crop=False, mean=(104, 117, 123))
        self.model.setInput(blob)
        outputs =self.model.forward()
        final_outputs = outputs[0]
        # make all the outputs 1D
        final_outputs = final_outputs.reshape(9, 1)
        # get the class label
        label_id = np.argmax(final_outputs)
        # convert the output scores to softmax probabilities
        probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
        # get the final highest probability
        final_prob = np.max(probs) * 100.
        # map the max confidence to the class label names
        out_name = self.labels[label_id]
        elapsed_ms = (time.time() - start_time) * 1000
        # label_id, prob = results, 1
        logger.info("{} detected with with prob {} in {} milliseconds".format(out_name,final_prob,elapsed_ms))
        return out_name, final_prob




