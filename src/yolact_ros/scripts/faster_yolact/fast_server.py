#!/home/wxl/anaconda3/envs/torch/bin/python3
#-*-coding:utf-8-*-

# import sys
# print(sys.path)
# from asyncio.windows_events import NULL
# from email import header
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
from eval import parse_args,FastBaseTransform,prep_process,set_cfg,prep_display
from eval import Yolact,convert_to_tensorrt,SavePath,BaseTransform,logging
from yolact_edge.utils.logging_helper import setup_logger
from yolact_edge.data import cfg
from yolact_edge.data.config import COCO_CLASSES
from yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.utils import timer
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
        self.image_topic = rospy.get_param('~image_topic', default='/image_view/image_raw')

        self.detections_pub = rospy.Publisher("/prediction/detections", Detections, queue_size=10)

        self.image_sub = rospy.Subscriber(self.image_topic , Image, self.callback)
        

        # set parameter default values (will be overwritten by dynamic reconfigure callback)
        self.image_topic = rospy.get_param('~image_topic', default='/image_view/image_raw')
        self.use_compressed_image = False
        self.publish_detections = True

        self.display_fps = False
        self.score_threshold = args.score_threshold # 0.0
        self.crop_masks = True
        self.top_k = 100 # 15

        # for counting fps
        self.fps = 0
        self.last_reset_time = rospy.Time()
        self.frame_counter = 0
        self.arrs = []
        self.headers = []
        self.images = []
        self.arr = None

        # for select request type
        self.is_image = True
        self.transform = FastBaseTransform()
        
    # 订阅图像帧的回调函数，讲图像帧的 array 和　header 存在列表中
    def callback(self, data):
        # self.images.append(data)
        self.image = data
        np_arr = np.frombuffer(data.data, np.uint8)
        self.arr = np_arr.reshape((data.height,data.width, -1)) # [:,:,::-1] # BGR ==> RGB
        # self.arrs.append(self.arr)
        # self.headers.append(data.header)
        self.header = data.header

        # cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # plt.clf()  #清除上一幅图像
        # plt.imshow(self.arr)
        # plt.show(block = False)
        # plt.pause(0.001) 
        # plt.ioff()

    # 主线程循环执行，只要订阅到了图像帧，就对其进行网络预测，并将结果包装好发布出去，随后删除该图像帧以释放资源
    def publisher(self):
        print("publisher Ready to process images.")
        print('**********************')
        rate = rospy.Rate(30) # 30hz
        while not rospy.is_shutdown():
            # 有未处理的图像帧
            if self.arr is not None:
                t1 = time.time() # wxl
                classes, scores, boxes, masks = self.pre_pub(self.arr) # 使用网络进行预测
                t2 = time.time() # wxl
                _,dets_msg = self.generate_detections_msg(classes, scores, boxes, masks, self.arr) # 包装待发布消息
                t3 = time.time() # wxl
                self.detections_pub.publish(dets_msg) # 将预测结果 publish 出去
                t4 = time.time() # wxl

                self.arr = None
                print(np.around(np.array([t2-t1,t3-t2,t4-t3])*1000,1))
                print('**********************')
 
            # rate.sleep()

    def handel_server(self,req):
        t1 = time.time() # wxl
        if req.rgb_path == "None":
            print(req.rgb_path)
            dets_srv = Req_detctResponse()
            return dets_srv
        if self.is_image:
            frame = req.rgb.data
            frame = np.frombuffer(frame, np.uint8)
            frame = frame.reshape((req.rgb.height,req.rgb.width, -1))
            print('Receive the request of Image!')
        else:
            image_path = req.rgb_path
            frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
            print('Receive the request of Path!')
        t2 = time.time() # wxl

        size = frame.shape
        arr = frame.reshape((size[0],size[1], -1)) # [:,:,::-1] # BGR ==> RGB
        t3 = time.time() # wxl

        print(size)
  
        classes, scores, boxes, masks = self.pre_pub(arr) # 使用网络进行预测
        t4 = time.time() # wxl
        dets_srv,dets_msg = self.generate_detections_msg(classes, scores, boxes, masks, frame) # 包装待发布消息
        t5 = time.time() # wxl
        self.detections_pub.publish(dets_msg) # 将预测结果 publish 出去
        t6 = time.time() # wxl
        print(np.around(np.array([t2-t1,t3-t2,t4-t3,t5-t4,t6-t5,t6-t1])*1000,1))
        print('**********************')

        return dets_srv

    # 主线程循环执行，只要收到了服务请求，就对其进行网络预测，并将结果包装好发布出去
    def server(self):
        # 创建一个名为/detect_image的server，注册回调函数handel_server 
        s = rospy.Service('/detect_image', Req_detct, self.handel_server) 

        # 循环等待回调函数
        print("server Ready to process images.")
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
            #####
            # mask expend
            #####
            factor = 1.1
            x1, y1, x2, y2 = (boxes[detnum]*factor).astype(np.int16)
            mask = masks[detnum].cpu().numpy()            
            de_factor = (factor-1)/2
            mask2 = cv2.resize(mask,(0,0),fx=factor,fy=factor)
            mask2 = mask2[y1:y2, x1:x2]>0
            x1, y1 = x1-de_factor*(boxes[detnum][0]+boxes[detnum][2]), y1-de_factor*(boxes[detnum][1]+boxes[detnum][3])
            if x1 >= 0:
                x1 = int(x1)
                x2 = x1 + mask2.shape[1]
            else:
                x2 = mask2.shape[1] + int(x1)
                mask2 = mask2[:,-int(x1):]
                x1 = 0
            if x2 >= 640:
                mask2 = mask2[:,:639-x2]
                x2 = 639
            if y1 >= 0:
                y1 = int(y1)
                y2 = y1 + mask2.shape[0]
            else:
                y2 = mask2.shape[0] + int(y1)
                mask2 = mask2[-int(y1):,:]
                y1 = 0
            if y2 >= 480:
                mask2 = mask2[:479-y2,:]
                y2 = 479
            #####
            # mask expend
            #####
            det = Detection()
            det.class_name = COCO_CLASSES[classes[detnum]]
            det.score = float(scores[detnum])
            # x1, y1, x2, y2 = boxes[detnum]
            det.box.x1 = int(x1)
            det.box.y1 = int(y1)
            det.box.x2 = int(x2)
            det.box.y2 = int(y2)
            # mask = masks[detnum,y1:y2,x1:x2]
            det.mask.mask = mask2.tobytes()
            # det.mask.mask = np.array(mask.bool().cpu(),dtype=np.uint8).tobytes()
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
        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False,
          "moving_statistics": None}

        [net, top_k, score_threshold] = [self.net, self.top_k, self.score_threshold]

        t1 = time.time() # wxl
        frame = torch.from_numpy(frame).cuda().float()
        t2 = time.time() # wxl
        # print('Load image use {} ms'.format( round( (t2 - t1)*1000 ) ))

        batch = self.transform(frame.unsqueeze(0))
        t3 = time.time() # wxl
        # print('Transform image use {} ms'.format( round( (t3 - t1)*1000 ) ))

        preds = net(batch, extras=extras)["pred_outs"]
        t4 = time.time() # wxl
        print('Predict image use {} ms'.format( round( (t4 - t1)*1000 ) ))

        # t = prep_process(preds, frame, None, None, undo_transform=False)
        h, w, _ = frame.shape
        t = postprocess(preds, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        t5 = time.time() # wxl
        print('postprocess Predict use {} ms'.format( round( (t5 - t4)*1000 ) ))

        # idx = t[1].argsort(0, descending=True)[:top_k] #　根据 score 排序，只取前top_k个
        # masks = t[3][:top_k]
        # classes, scores, boxes = [x[:top_k].cpu().numpy() for x in t[:3]]

        ##
        print(t[0].shape)
        idx = torch.nonzero(t[0]==0).squeeze()  # 筛选所有是人的 index
        t6 = time.time() # wxl
        print('Get idx use {} ms'.format( round( (t6 - t5)*1000 ) ))
        # masks = t[3][idx,:,:]
        masks = torch.index_select(t[3], 0, idx)
        classes, scores, boxes = [torch.index_select(x, 0, idx).cpu().numpy() for x in t[:3]]
        t7 = time.time() # wxl
        print('select Predict use {} ms'.format( round( (t7 - t6)*1000 ) ))
        
        ##

        # num_dets_to_consider = min(top_k, classes.shape[0])
        # for j in range(num_dets_to_consider):
        #     if scores[j] < score_threshold:
        #         num_dets_to_consider = j
        #         break
        # classes = classes[:num_dets_to_consider]
        # scores = scores[:num_dets_to_consider]
        # boxes = boxes[:num_dets_to_consider]
        # masks = masks[:num_dets_to_consider]
        


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
    parse_args([])
    from eval import args

    args.trained_model = "/home/wxl/graduate/dyslam_ws/src/yolact_ros/scripts/faster_yolact/weights/yolact_edge_mobilenetv2_54_800000.pth"
    if args.config is not None:
        set_cfg(args.config)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    setup_logger(logging_level=logging.INFO)
    logger = logging.getLogger("yolact.eval")

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    with torch.no_grad():
        logger.info('Loading model...')
        net = Yolact(training=False)
        net.load_weights(args.trained_model, args=args)
        net.eval()
        logger.info('Model loaded.')

        net.detect.use_fast_nms = args.fast_nms
        cfg.mask_proto_debug = args.mask_proto_debug

        args.score_threshold = 0.012
        args.top_k = 100

        convert_to_tensorrt(net, cfg, args, transform=BaseTransform())

        ic = YolactNode(net)

    try:
        ic.server()
        # ic.publisher()
    except rospy.ROSInterruptException:
        pass
    # rospy.spin()