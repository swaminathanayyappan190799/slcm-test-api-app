from datetime import datetime
from turtle import right
import cv2
import time
import sys
import numpy as np
import os
import random
from django.conf import settings
import tempfile

from classifier.Onnxclassifier import Classify


class DetectONNX():
    def __init__(self) -> None:
        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 640
        self.SCORE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.2
        self.CONFIDENCE_THRESHOLD = 0.2
        self.is_cuda = True if settings.DEVICE == "cuda" else False
        # self.is_cuda = False
        self.colors = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in range(1,20) ]
        self.start = time.time_ns()
        frame_count = 0
        self.total_frames = 0
        fps = -1
        self.out_path =os.path.join(settings.BASE_DIR, 'media')
        # self.out_path = '../media'
        self.outurl = '/media/'
        self.load_classes()
        self.build_model()
        #classifier
        self.image_classifier = Classify('chana_labels.txt','250_best.onnx')

    def build_model(self, model=f"detector{os.sep}models{os.sep}bestprev.onnx"):
        self.net = cv2.dnn.readNet(model)
        if self.is_cuda:
            print("Attempty to use CUDA")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()
        return preds


    def load_classes(self, classes_file=f"detector{os.sep}models{os.sep}classes.txt"):
        self.class_list = []
        with open(classes_file, "r") as f:
            self.class_list = [cname.strip() for cname in f.readlines()]


    def wrap_detection(self, input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []

        rows = output_data.shape[0]

        image_width, image_height, _ = input_image.shape

        x_factor = image_width / self.INPUT_WIDTH
        y_factor =  image_height / self.INPUT_HEIGHT

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= self.CONFIDENCE_THRESHOLD:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > self.SCORE_THRESHOLD):

                    confidences.append(confidence)

                    class_ids.append(class_id)

                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    left = 0 if left < 0 else left
                    top = 0 if top < 0 else top
                    box = np.array([left, top, width, height])
                    boxes.append(box)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD) 

        result_class_ids = []
        result_confidences = []
        result_boxes = []

        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes

    def format_yolov5(self, frame):

        row, col, _ = frame.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = frame
        return result

    def run_detection(self, path):

        file_name = path.split(os.sep)[-1]
        frame=cv2.imread(path)
        classify_frame= frame.copy()
        if frame is None:
            print("End of stream")
            

        inputImage = self.format_yolov5(frame)
        # inputImage = frame

        outs = self.detect(inputImage)
        # outs = detect(inputImage, net)

        class_ids, confidences, boxes = self.wrap_detection(inputImage, outs[0])
        # grains = zip(class_ids, confidences, boxes)
        number_of_grains = len(class_ids)
        # frame_count += 1
        # total_frames += 1
        data = {}
        i=0
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            # color = self.colors[int(classid) % len(self.colors)]
            now = datetime.now().strftime("%Y%m%d-%H%M%S%f")
            x, y, w, h = box
            # print(x, y, w, h)
            detected_object, prob= self.store_for_classification(file_name, classify_frame, box, now)
            # number_of_grains += 1
            #####detector lables
            label = self.class_list[classid]
            ### we use classifier label
            if detected_object in data.keys():
                data[detected_object]['count']+=1
                data[detected_object]['annots'].extend([x,y,w,h])
                color = data[detected_object]['color']
            else:
                data[detected_object]={}
                data[detected_object]['count']=1
                data[detected_object]['annots']=[[x,y,w,h]]
                color = random.choice(self.colors)
                data[detected_object]['color'] = color

            
            i += 1

            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, f"{detected_object}:{prob}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
            

        cv2.imwrite(f"{self.out_path}{os.sep}in_{file_name}",inputImage)
        cv2.imwrite(f"{self.out_path}{os.sep}out_{file_name}",frame)
        out_url = f"{self.outurl}out_{file_name}"
        in_url = f"{self.outurl}in_{file_name}"
        # print(data, out_url, number_of_grains, in_url)
        return data, out_url, number_of_grains, in_url

    def store_for_classification(self, file_name, frame, box, id):
        x,y,w,h = box
        y = y-10 if y-10 > 0 else y
        x = x-10 if x-10 > 0 else x
        h=h+10
        w=w+10
        #for classification
        cv2.imwrite(f"{tempfile.gettempdir()}{os.sep}{id}.jpg",frame[y:y+h, x:x+w])
        classify_dir = file_name.split(".")[0]
        category, prob = self.image_classifier.run(f"{tempfile.gettempdir()}{os.sep}{id}.jpg")
        if prob < 90:
            category="undetected"
        classify_dir = f"{self.out_path}{os.sep}classify_{classify_dir}{os.sep}{category}"
        os.remove(f"{tempfile.gettempdir()}{os.sep}{id}.jpg")
        print(f"storing in dir {classify_dir}")
        if not os.path.exists(f"{classify_dir}"):
            os.makedirs(f"{classify_dir}")

        cv2.imwrite(f"{classify_dir}{os.sep}{id}.jpg",frame[y:y+h, x:x+w])
        return category, prob


    
        

        
# print("Total frames: " + str(total_frames))
# # run_detection('/Volumes/Projects/Projects/cv/chana/train/images/broken-11_jpg.rf.ef1b5accc6cb445b5cc534f03353e0cd.jpg')
# import glob
# ut=DetectONNX()
# base_dir= "/Volumes/Projects/Projects/cv/chana_c/"
# for file in glob.glob(f'{base_dir}/*/images/*.jpg'):
#     ut.run_detection(file)