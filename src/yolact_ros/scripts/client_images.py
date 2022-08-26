#!/home/wxl/anaconda3/envs/torch/bin/python3
#-*-coding:utf-8-*-

# import sys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# print(sys.path)
import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from yolact_ros.msg import Box,Mask,Detection,Detections
from yolact_ros.srv import Req_detct,Req_detctRequest

import random
import matplotlib.pyplot as plt
import cv2

import numpy as np
import glob
import time

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

pro_num = 0
tar_num = 1
t1 = time.time()
t2 = time.time()
# 订阅预测结果
def callback(msg):
    # self.images.append(data)
    # 测试回应是否正确
    print('There are %d objects being detected!'%len(msg.detections))
    global tar_num,t1,t2
    tar_num += 1
    t2 = time.time()
    print("Pub and Sub Use {:2f} ms".format((t2-t1)*1000))

    

if __name__ == '__main__':
    rospy.init_node('Camera_client', anonymous=True) #定义节点 /camera/image_raw /image_view/image_raw
    image_pub=rospy.Publisher('/image_view/image_raw', Image, queue_size = 10) #定义话题
    depth_pub=rospy.Publisher('/image_view/depth_raw', Image, queue_size = 10) #定义话题
    detections_sub = rospy.Subscriber("/prediction/detections", Detections, callback)

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
    
    run_service = True
    if run_service:
        person_client = rospy.ServiceProxy('/detect_image', Req_detct)
        # 发现/detect_image服务后，创建一个服务客户端，连接名为/detect_image的service
        rospy.wait_for_service('/detect_image')

    while i < len(images) and (not rospy.is_shutdown()):    
        if pro_num >= tar_num:
            continue  
        t2 = time.time()
        print("All Use {:2f} ms".format((t2-t1)*1000))
        t1 = time.time()

        frame = cv2.imread(depths[i], cv2.IMREAD_UNCHANGED)

        stamp = rospy.Time.now()
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

        frame = cv2.imread(images[i], cv2.IMREAD_COLOR)
        # frame = cv2.flip(frame,0)   #镜像操作
        # frame = cv2.flip(frame,1)   #镜像操作   
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

        if run_service:
            try:
                t1 = time.time()
                # 请求服务调用，输入请求数据
                response = person_client(images[i], ros_frame)
                t2 = time.time()
                print("predict Use {:2f} ms".format((t2-t1)*1000))

                t1 = time.time()
                # 请求服务调用，输入请求数据
                response = person_client("None", ros_frame)
                t2 = time.time()
                print("Request service Use {:2f} ms".format((t2-t1)*1000))
                t1 = time.time()
                # 请求服务调用，输入请求数据
                response = person_client("None", ros_frame)
                t2 = time.time()
                print("Request serviceUse {:2f} ms".format((t2-t1)*1000))

                # 测试回应是否正确
                print('There are %d objects being detected!'%len(response.detections))
                # print(response.detections[0].class_name)
            except rospy.ServiceException:
                print("Service call failed ")
        else:
            pro_num += 1
                
        rate.sleep()
        i = (i + 1) % len(depths) # max = len(images - 1)