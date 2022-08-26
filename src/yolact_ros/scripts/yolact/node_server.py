#!/home/wxl/anaconda3/envs/torch/bin/python3
#-*-coding:utf-8-*-

# import sys
# print(sys.path)
from email import header
from matplotlib import image
from numpy.core.shape_base import block
import rospy
import rospkg
import os
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from yolact_ros.msg import Box,Mask,Detection,Detections
from yolact_ros.srv import Req_detct,Req_detctResponse
from eval import load_net
from eval import FastBaseTransform,prep_process,prep_display
from data import cfg
from data.config import COCO_CLASSES
from layers.output_utils import postprocess, undo_image_transformation
from utils import timer
import torch
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import gc

import warnings
warnings.filterwarnings("ignore")

class YolactNode:
    
    def __init__(self, net):
        self.net = net

        self.detections_pub = rospy.Publisher("/prediction/detections", Detections, queue_size=1)
        self.image_pub = rospy.Publisher("/prediction/image", Image, queue_size=1)

        # set parameter default values (will be overwritten by dynamic reconfigure callback)
        self.image_topic = rospy.get_param('~image_topic', default='/image_view/image_raw')
        self.use_compressed_image = False
        self.publish_detections = True

        self.display_fps = False
        self.score_threshold = 0.15 # 0.0
        self.crop_masks = True
        self.top_k = 8 # 15

        # for counting fps
        self.fps = 0
        self.last_reset_time = rospy.Time()
        self.frame_counter = 0
        self.arrs = []
        self.headers = []
        self.images = []

        # for select request type
        self.is_image = True
        
    def handel_server(self,req):
        
        if self.is_image:
            frame = req.rgb.data
            frame = np.frombuffer(frame, np.uint8)
            frame = frame.reshape((req.rgb.height,req.rgb.width, -1))
            print('Receive the request of Image!')
        else:
            image_path = req.rgb_path
            print('Receive the request of Path!')
            frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

        size = frame.shape
        print(size)
        arr = frame.reshape((size[0],size[1], -1)) # [:,:,::-1] # BGR ==> RGB
  
        classes, scores, boxes, masks = self.pre_pub(arr) # 使用网络进行预测
        dets_srv,dets_msg = self.generate_detections_msg(classes, scores, boxes, masks, frame) # 包装待发布消息
        self.detections_pub.publish(dets_msg) # 将预测结果 publish 出去

        print('**********************')

        return dets_srv

    # 主线程循环执行，只要收到了服务请求，就对其进行网络预测，并将结果包装好发布出去
    def server(self):
        # 创建一个名为/detect_image的server，注册回调函数handel_server
        s = rospy.Service('/detect_image', Req_detct, self.handel_server)

        # 循环等待回调函数
        print("Ready to process images.")
        print('**********************')
        rospy.spin()

    # 用于将预测信息包装成待 pub 的 message
    def generate_detections_msg(self,classes, scores, boxes, masks,image):
        # 实例化待发布的消息及服务resonse
        dets_msg = Detections()
        dets_srv = Req_detctResponse()

        # 将图片包装到Image消息中等待发布
        stamp = rospy.Time.now()
        image_header = Header(stamp = stamp)
        image_header.frame_id = "Camera"
        ros_frame = Image()
        ros_frame.header = image_header
        ros_frame.width = 640
        ros_frame.height = 480
        ros_frame.encoding = "bgr8" # 图像编码，可见　http://docs.ros.org/en/jade/api/sensor_msgs/html/image__encodings_8h.html
        ros_frame.step = 1920
        ros_frame.data = np.array(image).tobytes() #图片格式转换

        # 将预测结果包装到消息中
        for detnum in range(len(classes)):
            det = Detection()
            det.class_name = COCO_CLASSES[classes[detnum]]
            det.score = float(scores[detnum])
            x1, y1, x2, y2 = boxes[detnum]
            det.box.x1 = int(x1)
            det.box.y1 = int(y1)
            det.box.x2 = int(x2)
            det.box.y2 = int(y2)
            mask = masks[detnum,y1:y2,x1:x2]
            det.mask.mask = np.array(mask.bool().cpu(),dtype=np.uint8).tobytes()
            det.mask.height = int(y2 - y1)
            det.mask.width = int(x2 - x1)
            dets_msg.detections.append(det)

        
        # msg
        dets_msg.header = image_header
        dets_msg.frame = ros_frame
        
        # srv_response
        dets_srv.detections = dets_msg.detections
        dets_srv.frame = ros_frame
        
        return dets_srv,dets_msg

    # 用于对某个图像 frame 预测获得 classes score boxes masks 等
    def pre_pub(self, frame):

        [net, top_k, score_threshold] = [self.net, self.top_k, self.score_threshold]

        t1 = time.time() # wxl
        frame = torch.from_numpy(frame).cuda().float()
        t2 = time.time() # wxl
        # print('Load image use {} ms'.format( round( (t2 - t1)*1000 ) ))

        batch = FastBaseTransform()(frame.unsqueeze(0))
        t3 = time.time() # wxl
        # print('Transform image use {} ms'.format( round( (t3 - t1)*1000 ) ))

        preds = net(batch)
        t4 = time.time() # wxl
        print('Predict image use {} ms'.format( round( (t4 - t1)*1000 ) ))

        t = prep_process(preds, frame, None, None, undo_transform=False)

        idx = t[1].argsort(0, descending=True)[:top_k] #　根据 score 排序，只取前top_k个
            
        masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        num_dets_to_consider = min(top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < score_threshold:
                num_dets_to_consider = j
                break
        classes = classes[:j]
        scores = scores[:j]
        boxes = boxes[:j]
        masks = masks[:j]


        return classes, scores, boxes, masks

    # 测试函数，用于查看预测效果，预测结果将保存为图片
    def evalimage(self,net, path:str, save_path:str=None):
        t1 = time.time() # wxl
        frame = torch.from_numpy(cv2.imread(path)).cuda().float()
        t2 = time.time() # wxl
        print('Load image use {} ms'.format( round( (t2 - t1)*1000 ) ))

        batch = FastBaseTransform()(frame.unsqueeze(0))
        t3 = time.time() # wxl
        print('Transform image use {} ms'.format( round( (t3 - t1)*1000 ) ))

        preds = net(batch)
        t4 = time.time() # wxl
        print('Predict image use {} ms'.format( round( (t4 - t1)*1000 ) ))

        # prep_process(preds, frame, None, None, undo_transform=False)

        img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
        t5 = time.time() # wxl
        print('Draw image use {} ms'.format( round( (t5 - t1)*1000 ) ))

        if save_path is None:
            img_numpy = img_numpy[:, :, (2, 1, 0)]

        if save_path is None:
            plt.imshow(img_numpy)
            plt.title(path)
            plt.show()
        else:
            label = 'Yolact on 1050Ti GPU, Inference time for a frame : %0.0f ms' % abs((t4-t3)*1000)
            cv2.putText(img_numpy, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            cv2.imwrite(save_path, img_numpy)
        return img_numpy




if __name__ == '__main__':
    rospy.init_node('yolact_ros_server')
    rospack = rospkg.RosPack()
    yolact_path = rospack.get_path('yolact_ros')
    model_path_str = rospy.get_param('~model_path', os.path.join(yolact_path, "scripts/weights/yolact_plus_resnet50_54_800000.pth"))

    with torch.no_grad():
        net = load_net(['--trained_model', model_path_str, '--config', 'yolact_plus_resnet50_config', '--score_threshold', '0.15', '--top_k', '15'])

        ic = YolactNode(net)

    try:
        ic.server()
    except rospy.ROSInterruptException:
        pass
    # rospy.spin()