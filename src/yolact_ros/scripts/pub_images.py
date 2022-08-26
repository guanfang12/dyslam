#!/home/wxl/anaconda3/envs/torch/bin/python3
#-*-coding:utf-8-*-

# import sys
# print(sys.path)
import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
# from yolact_ros.msg import Box,Mask,Detection,Detections
# import rospkg
# import os
# import time
# import itertools
# from cv_bridge import CvBridge , CvBridgeError

import random
import matplotlib.pyplot as plt
import cv2

import numpy as np
import glob

def imageSegmentationGenerator(images_path, deps_path = None):

    assert images_path[-1] == '/'
    images = sorted(glob.glob(images_path + "*.jpg") +
                    glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg"))

    if deps_path is None:
        depths = []
    else:
        assert deps_path[-1] == '/'
        depths = sorted(glob.glob(deps_path + "*.jpg") +
                           glob.glob(deps_path + "*.png") + glob.glob(deps_path + "*.jpeg"))
    
    return images, depths

    # zipped = itertools.cycle(zip(images, depths))

    # while True:
    #     im, dep = zipped.__next__()
    #     im = cv2.imread(im, cv2.IMREAD_COLOR)
    #     dep = cv2.imread(dep, cv2.IMREAD_UNCHANGED)
    #     # for _ in range(batch_size):
    #     #     im, seg = zipped.__next__()
    #     #     im = cv2.imread(im, cv2.IMREAD_COLOR)
    #     #     seg = cv2.imread(seg, cv2.IMREAD_UNCHANGED)

    #     #     assert im.shape[:2] == seg.shape[:2]

    #     #     X.append(getImageArr(im))
    #     #     Y.append(
    #     #         getSegmentationArr(
    #     #             seg,
    #     #             n_classes,
    #     #             input_height,
    #     #             input_width))

    #     yield np.array(im), np.array(dep)


if __name__ == '__main__':
    # G = imageSegmentationGenerator("/home/wxl/Data/TUM/rgbd_dataset_freiburg3_walking_xyz/rgb/",
                                #    "/home/wxl/Data/TUM/rgbd_dataset_freiburg3_walking_xyz/depth/")
    
    # print(1)
    # x, y = G.__next__()
    # print(x.shape, y.shape)

    rospy.init_node('Camera', anonymous=True) #定义节点 /camera/image_raw /image_view/image_raw
    image_pub=rospy.Publisher('/image_view/image_raw', Image, queue_size = 10) #定义话题
    depth_pub=rospy.Publisher('/image_view/depth_raw', Image, queue_size = 10) #定义话题

    print("Star to get paths of RGBs and Depths!")
    # images, depths = imageSegmentationGenerator("/home/wxl/Data/TUM/rgbd_dataset_freiburg3_walking_xyz/rgb/")

    images, depths = imageSegmentationGenerator("/home/wxl/Data/TUM/rgbd_dataset_freiburg3_walking_xyz/rgb/",
                                                "/home/wxl/Data/TUM/rgbd_dataset_freiburg3_walking_xyz/depth/")

    print("Finish to get paths of RGBs and Depths!")

    rate = rospy.Rate(24) # 24hz 
    # bridge = CvBridge()
    i = 0
    print("Start to pub images!!!!")
    ros_frame = Image()

    while i < len(images) and (not rospy.is_shutdown()):       
        frame = cv2.imread(images[i], cv2.IMREAD_COLOR)
        # frame = cv2.flip(frame,0)   #镜像操作
        # frame = cv2.flip(frame,1)   #镜像操作   
        stamp = rospy.Time.now()
        header = Header(stamp = stamp)
        header.frame_id = "Camera"
        ros_frame.header=header
        ros_frame.width = 640
        ros_frame.height = 480
        ros_frame.encoding = "bgr8" # 图像编码，可见　http://docs.ros.org/en/jade/api/sensor_msgs/html/image__encodings_8h.html
        ros_frame.step = 1920
        ros_frame.data = np.array(frame).tobytes() #图片格式转换
        # ros_frame = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        image_pub.publish(ros_frame) #发布消息

        frame = cv2.imread(depths[i], cv2.IMREAD_UNCHANGED)
        # frame = cv2.flip(frame,0)   #镜像操作
        # frame = cv2.flip(frame,1)   #镜像操作           
        header = Header(stamp = stamp)
        header.frame_id = "Camera"
        ros_frame.header=header
        ros_frame.width = 640
        ros_frame.height = 480
        ros_frame.encoding = "16UC1" # 图像编码，可见　http://docs.ros.org/en/jade/api/sensor_msgs/html/image__encodings_8h.html
        ros_frame.step = 1920
        ros_frame.data = np.array(frame).tobytes() #图片格式转换
        # ros_frame = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        depth_pub.publish(ros_frame) #发布消息
                
        rate.sleep()
        i = (i + 1) % len(depths) # max = len(images - 1)

