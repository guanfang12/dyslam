#!/home/wxl/anaconda3/envs/torch/bin/python3
#-*-coding:utf-8-*-

# import sys
# print(sys.path)
import rospy
import rospkg
import os
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from yolact_ros.msg import Box,Mask,Detection,Detections
import torch
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import gc

# for making bounding boxes pretty
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

class DrawNode:
    # 订阅 image_raw 和网络预测结果，发布预测的可视化结果
    def __init__(self):
        self.draw_pub = rospy.Publisher("/prediction/image_draw/image_raw", Image, queue_size=1)

        # self.image_sub = rospy.Subscriber('/image_view/image_raw', Image, self.callback)
        self.dets_sub = rospy.Subscriber("/prediction/detections", Detections, self.callback2)

        self.publish_draw = True

        self.arrs = []         # 订阅到的图像数据
        self.stamps = []       # 订阅图像数据对应的时间戳，用于和预测结果匹配
        self.ts = []           # 订阅到的预测结果
        self.stamps_dets = []  # 订阅到的预测结果与对应的时间戳

        self.frame_counter = 0

    # 图像订阅的回调函数
    def callback(self,msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.arr = np_arr.reshape((msg.height,msg.width, -1)) # [:,:,::-1] # BGR ==> RGB
        self.arrs.append(self.arr)
        self.stamps.append(msg.header.stamp)

    # 预测结果订阅的回调函数，会将消息解码，并存至类元素的列表中
    def callback2(self,msg):
        
        classes = []
        scores = []
        boxes = []
        masks = []
        for i in range(len(msg.detections)):
            det = msg.detections[i]
            classes.append(det.class_name)
            scores.append(det.score)
            boxes.append([det.box.x1,det.box.y1,det.box.x2,det.box.y2])
            mask = np.frombuffer(det.mask.mask, np.uint8) # np.array(det.mask.mask,dtype=np.uint8)
            mask = mask.reshape((det.mask.height,det.mask.width))
            # mask = np.unpackbits(mask,axis=1)
            # mask = mask[:det.mask.height,:det.mask.width]
            masks.append(mask)
        self.stamps_dets.append(msg.header.stamp)
        self.ts.append([classes,scores,boxes,masks])

        data = msg.frame

        np_arr = np.frombuffer(data.data, np.uint8)
        self.arr = np_arr.reshape((data.height,data.width, -1)) # [:,:,::-1] # BGR ==> RGB
        self.arrs.append(self.arr)
        self.stamps.append(data.header.stamp)


    # 将预测结果绘制到图像上
    def draw(self,index):
        stamp = self.stamps_dets[index]   # 网络结果对应的时间戳
        # 根据时间戳将预测结果与图像进行配对，若配对失败则跳过
        if not stamp in self.stamps:
            return None
        index_2 = self.stamps.index(stamp) # 获得网络预测结果对应的图像索引
        frame = self.arrs[index_2]
        t = self.ts[index]
        classes,scores,boxes,masks = t
        numDetections = len(classes)
        dynamic_path = "/home/wxl/graduate/dyslam_ws/src/dynamic_orb_slam2/ModelsCNN/dynamic.txt"
        with open(dynamic_path, 'r', encoding='utf-8') as f:
            dynamic_obj = f.read().splitlines()
        # 绘制预测结果
        for i in range(numDetections):

            if not(classes[i] in dynamic_obj):
                continue                
            
            box = boxes[i]
            mask = masks[i]
            score = scores[i]
            color = COLORS[i % len(COLORS)]
            
            # Draw bounding box, colorize and show the mask on the image
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 178, 50), 3)
    
            # Print a label of class.
            label = '%.2f' % score
            label = '%s:%s' % (classes[i], label)
            
            # Display the label at the top of the bounding box
            font_face = cv2.FONT_HERSHEY_DUPLEX  # cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 # 0.5
            font_thickness = 1
            text_w, text_h = cv2.getTextSize(label, font_face, font_scale, font_thickness) # text_w, text_h
            # (x1 + text_w, y1 - text_h - 4)
            cv2.rectangle(frame, (box[0], box[1]), (box[0] + text_w[0], box[1]- text_w[1] - 4), color, cv2.FILLED)
            cv2.putText(frame, label, (box[0], box[1]- 3), font_face, font_scale, (255,255,255), font_thickness)

            # Resize the mask, threshold, color and apply it on the image
            mask = (mask > 0)
            roi = frame[box[1]:box[3], box[0]:box[2]][mask]

            frame = np.array(frame)
            
            if frame.shape[2] == 3:
                frame[box[1]:box[3], box[0]:box[2]][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
            else:
                # frame[box[1]:box[3], box[0]:box[2]][mask] = (0.7*255 + 0.3 * roi).astype(np.uint8)
                temp = np.zeros((frame.shape[0],frame.shape[1],3))
                for i in range(3):
                    temp[:,:,i] = frame[:,:,0]
                frame = temp.astype(np.uint8)
                frame[box[1]:box[3], box[0]:box[2]][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
            # Draw the contours on the image 是否画出轮廓
            # mask = mask.astype(np.uint8)
            # contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(frame[box[1]:box[3]+1, box[0]:box[2]+1], contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)

        del self.ts[index]
        # del self.stamps_dets[index] 后面还要用，用完再释放资源

        # 因为预测结果和图像都是时序的，所以存储的图像若某一帧才获得了对应预测，该帧之前的所有帧都将不会再出现匹配的预测结果
        del self.arrs[:index_2 + 1]
        del self.stamps[:index_2 + 1]

        return frame

    # 将预测结果绘制为黑白图像
    def draw_bin(self,index):
        stamp = self.stamps_dets[index]   # 网络结果对应的时间戳
        # 根据时间戳将预测结果与图像进行配对，若配对失败则跳过
        if not stamp in self.stamps:
            return None
        index_2 = self.stamps.index(stamp) # 获得网络预测结果对应的图像索引
        frame = self.arrs[index_2]
        t = self.ts[index]
        classes,scores,boxes,masks = t
        numDetections = len(classes)


        frame_bin = np.ones((frame.shape[0],frame.shape[1]))*255

        for i in range(numDetections):

            box = boxes[i]
            mask = masks[i]
    
            # Resize the mask, threshold, color and apply it on the image
            mask = (mask > 0)
            roi = frame_bin[box[1]:box[3], box[0]:box[2]][mask]

            # frame_bin = np.array(frame_bin)
            
            frame_bin[box[1]:box[3], box[0]:box[2]][mask] = (0 + 0 * roi).astype(np.uint8)

        del self.ts[index]
        # del self.stamps_dets[index] 后面还要用，用完再释放资源

        # 因为预测结果和图像都是时序的，所以存储的图像若某一帧才获得了对应预测，该帧之前的所有帧都将不会再出现匹配的预测结果
        del self.arrs[:index_2 + 1]
        del self.stamps[:index_2 + 1]

        return frame_bin.astype(np.uint8)


if __name__ == '__main__':

    rospy.init_node('Draw', anonymous=True) #定义节点

    print('Start!!!')

    draw = DrawNode()

    rate = rospy.Rate(30) # 30Hz
    symble = 1 # 0 ==> not find  1 ==> find
    while not rospy.is_shutdown():
        if draw.frame_counter < len(draw.ts):
            # print("There is {} frames!!".format(len(draw.ts)))
            # frame = draw.draw(draw.frame_counter)
            frame = draw.draw_bin(draw.frame_counter)
            # 若没有找到预测结果对应的图像帧，则跳过该预测
            if frame is None:
                if symble == 1:
                    print("Cannot find corresponding image!!")
                    symble = 0
                continue
            # print("There is {} frames!!".format(len(draw.ts)))
            if symble == 0:
                print("Find corresponding image!!")
            ros_frame = Image()
            header = Header(stamp = draw.stamps_dets[draw.frame_counter])
            header.frame_id = "Camera"
            ros_frame.header=header
            ros_frame.width = 640
            ros_frame.height = 480
            print(frame.shape)
            if len(frame.shape) == 3:
                ros_frame.encoding = "bgr8"
            else:
                ros_frame.encoding = "8UC1" 
            print(frame.shape,ros_frame.encoding)
            ros_frame.step = 1920
            ros_frame.data = np.array(frame).tobytes() #图片格式转换

            draw.draw_pub.publish(ros_frame) #发布消息
            # draw.frame_counter += 1
            del draw.stamps_dets[draw.frame_counter]
            gc.collect()

            # plt.imshow(frame[:,:,::-1])
            # plt.pause(0.01)