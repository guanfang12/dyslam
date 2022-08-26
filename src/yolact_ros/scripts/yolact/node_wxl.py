#!/home/wxl/anaconda3/envs/torch/bin/python3
#-*-coding:utf-8-*-

# import sys
# print(sys.path)
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
from yolact_ros.srv import Req_detct
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

        self.image_sub = rospy.Subscriber(self.image_topic , Image, self.callback)

    # 订阅图像帧的回调函数，讲图像帧的 array 和　header 存在列表中
    def callback(self, data):
        self.images.append(data)
        np_arr = np.frombuffer(data.data, np.uint8)
        self.arr = np_arr.reshape((data.height,data.width, -1)) # [:,:,::-1] # BGR ==> RGB
        self.arrs.append(self.arr)
        self.headers.append(data.header)

        # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # plt.clf()  #清除上一幅图像
        # plt.imshow(self.arr)
        # plt.show(block = False)
        # plt.pause(0.001) 
        # plt.ioff()
        
    def handel_server(self,req):
        image_name = req.rgb_name

    # 主线程循环执行，只要订阅到了图像帧，就对其进行网络预测，并将结果包装好发布出去，随后删除该图像帧以释放资源
    def publisher(self):
        rate = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():
            # 证明有未处理的图像帧
            if self.frame_counter < len(self.arrs): 
                print("There is {} frames!!".format(len(self.arrs)))
                if len(self.arrs) > 4:
                    del self.arrs[:4]
                    del self.headers[:4]
                    del self.images[:4]
                elif len(self.arrs) > 3:
                    del self.arrs[:3]
                    del self.headers[:3]
                    del self.images[:3]
                elif len(self.arrs) > 2:
                    del self.arrs[:2]
                    del self.headers[:2]
                    del self.images[:2]

                classes, scores, boxes, masks = self.pre_pub(self.arrs[self.frame_counter]) # 使用网络进行预测
                dets_msg = self.generate_detections_msg(classes, scores, boxes, masks, self.headers[self.frame_counter],self.images[self.frame_counter]) # 包装待发布消息
                self.detections_pub.publish(dets_msg) # 将预测结果 publish 出去
                self.image_pub.publish(self.images[self.frame_counter])
                # self.frame_counter = self.frame_counter + 1
                # 预测完当前帧后删除列表中元素以释放资源
                del self.images[self.frame_counter]
                del self.arrs[self.frame_counter]
                del self.headers[self.frame_counter]
                gc.collect()
            rate.sleep()

    # 用于将预测信息包装成待 pub 的 message
    def generate_detections_msg(self,classes, scores, boxes, masks, image_header,image):
        dets_msg = Detections()
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

        dets_msg.header = image_header
        dets_msg.frame = image

        
        return dets_msg

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
        print('**********************')

        t = prep_process(preds, frame, None, None, undo_transform=False)

        '''
        用于可视化网络输出，包括　classes scores boxes masks

        print(t[0:3])
        print('classes: {}'.format(t[0].shape))
        print('scores: {}'.format(t[1].shape))
        print('boxes: {}'.format(t[2].shape))
        print('masks: {}'.format(t[3].shape))

        out_img = np.zeros(t[3][0].shape)
        num = t[0].shape[0]
        r = int((num+1)//3)
        if (num+1)%33 >1:
            r = r + 1
        for i in range(num):
            temp = t[3][i].cpu().numpy()
            plt.subplot(r,3,i+1)
            plt.title(COCO_CLASSES[t[0][i]]) 
            plt.imshow(temp)
            out_img = out_img + temp

        用于查看像素直方图，了解 mask 组成
        # fig = plt.figure()
        # plt.subplot(2,1,1)
        # temp = t[3][0].cpu().numpy()
        # plt.hist(temp.reshape(temp.shape[0]*temp.shape[1]))
        # plt.subplot(2,1,2)
        plt.subplot(r,3,num+1)
        plt.imshow(out_img)
        # plt.show()
        '''

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
    rospy.init_node('yolact_ros')
    rospack = rospkg.RosPack()
    yolact_path = rospack.get_path('yolact_ros')
    model_path_str = rospy.get_param('~model_path', os.path.join(yolact_path, "scripts/weights/yolact_plus_resnet50_54_800000.pth"))

    with torch.no_grad():
        net = load_net(['--trained_model', model_path_str, '--config', 'yolact_plus_resnet50_config', '--score_threshold', '0.15', '--top_k', '15'])

        ic = YolactNode(net)

        # path = r'/home/wxl/graduate/dyslam_ws/src/yolact_ros/scripts/4.png'
        # t1 = time.time() # wxl
        # frame = cv2.imread(path)
        # t2 = time.time() # wxl
        # print('Load image use {} ms'.format( round( (t2 - t1)*1000 ) ))
        # classes, scores, boxes, masks = ic.pre_pub(frame)
        # dets_msg = ic.generate_detections_msg(classes, scores, boxes, masks, ic.headers[ic.frame_counter])
        # ic.detections_pub.publish(dets_msg)

        # img = evalimage(net, r'/home/wxl/graduate/dyslam_ws/src/yolact_ros/scripts/1.jpg', r'1-wow.jpg')
        # plt.imshow(img)
        # plt.show()
        # plt.figure()
        # for i in range(100):
        #     plt.imshow(ic.arrs[i])
        #     plt.pause(0.001)
    try:
        ic.publisher()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()