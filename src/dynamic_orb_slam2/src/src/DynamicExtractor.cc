//
// Created by horacehxw on 4/10/19.
//

#include "DynamicExtractor.h"

using namespace std;
using namespace cv;
using namespace dnn;

namespace ORB_SLAM2 {
    DynamicExtractor::DynamicExtractor(const string &strModelPath, int maxUsage, bool useOpticalFlow) : maxUsage(maxUsage),
                                                                                   useOpticalFlow(useOpticalFlow){
        maskUsage = 0;
        ros::NodeHandle n;
        mclient = n.serviceClient<yolact_ros::Req_detct>("/detect_image");

        // string temp;
        // ros::param::get("/RGB_D/confThreshold",temp);
        // cout<<"/RGB_D/confThreshold: "<<temp<<endl;
        // confThreshold = std::stof(temp);
        // ros::param::get("/RGB_D/maskThreshold",temp);
        // cout<<"/RGB_D/maskThreshold: "<<temp<<endl;
        // maskThreshold = std::stof(temp);

        confThreshold = 0.08;
        maskThreshold = 0.3;

        string textGraph = strModelPath + "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
        string modelWeights = strModelPath +  "frozen_inference_graph.pb";
        string classesFile = strModelPath + "mscoco_labels.names";
        string dynamicClassFile = strModelPath + "dynamic.txt";

        // Load names of classes
        ifstream ifs(classesFile.c_str());
        string line;
        while (getline(ifs, line)) classes.push_back(line);

        // load names of dynamic classes
        ifstream ifs2(dynamicClassFile.c_str());
        while (getline(ifs2, line)) dynamicClasses.insert(line);

        // Load the network
        // net = readNetFromTensorflow(modelWeights, textGraph);
        cout << "load model!!!" << endl;
        // net.setPreferableBackend(DNN_BACKEND_OPENCV);
        // should be able to use intel GPU
        // net.setPreferableTarget(DNN_TARGET_OPENCL);
        cout << "Out!!" << endl;

    }

    Mat DynamicExtractor::extractMask(const Mat &frame) {
        Mat blob;
        cout << "1###########" << endl;
        // Create 4D blob from a frame as input
        blobFromImage(frame, blob, 1.0, Size(frame.cols, frame.rows), Scalar(), true, false);
        cout << "2###########" << endl;
        net.setInput(blob);
        cout << "3###########" << endl;

        // Runs the forward pass to get output from the output layers
        std::vector<String> outNames{"detection_out_final", "detection_masks"};
        vector<Mat> outs;
        cout << "4###########" << endl;
        net.forward(outs, outNames);
        cout << "5##########---#" << endl;
        Mat outDetections = outs[0];
        Mat outMasks = outs[1];
        

        // Output size of masks is NxCxHxW where
        // N - number of detected boxes
        // C - number of classes (excluding background)
        // HxW - segmentation shape
        const int numClasses = outMasks.size[1];

        // Output size of Detection size is 1 * 1 * numDetections * 7
        // 7 refers to classId, score, and detection box information
        const int numDetections = outDetections.size[2];
        // reshape to channel = 1, row = num of detections
        // now outDetection size is numDetections * 7
        outDetections = outDetections.reshape(1, outDetections.total() / 7);

        // aggregate binary mask of dynamic objects into dynamic_mask
        // dynamic part should be zero
        Mat dynamic_mask = Mat(frame.size(), CV_8U, Scalar(255));
        Mat mat_zeros = Mat::zeros(frame.size(), CV_8U);
        for (int i = 0; i < numDetections; ++i) {
            float score = outDetections.at<float>(i, 2);
            if (score > confThreshold) {
                // Extract class id
                int classId = static_cast<int>(outDetections.at<float>(i, 1));

                // Extract bounding box
                int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
                int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
                int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
                int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));

                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                right = max(0, min(right, frame.cols - 1));
                bottom = max(0, min(bottom, frame.rows - 1));
                Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

                // Extract the mask for the object
                Mat objectMask(outMasks.size[2], outMasks.size[3], CV_32F, outMasks.ptr<float>(i, classId));
                // Resize the mask, threshold, color and apply it on the image
                resize(objectMask, objectMask, Size(box.width, box.height));
                // threshold mask into binary 255/0 mask
                Mat mask = (objectMask > maskThreshold);
                mask.convertTo(mask, CV_8U);

                if (is_dynamic(classId)) {
                    // copy ones into the corresponding mask region
                    mat_zeros(box).copyTo(dynamic_mask(box), mask);
                }
            }
        }
        return dynamic_mask;
    }

    // void DynamicExtractor::extractMask(const Mat &frame, Mat &dynamic_mask, const std::string RGB_path) {
    //     // if maskUsage <= masUsage, resuse prevMask
    //     cout << "1+++++++++++++" << endl;
    //     if (prevMask.empty() || maskUsage >= maxUsage) {
    //         prevFrame = frame.clone();
    //         prevMask = extractMask(frame, RGB_path);
    //         dynamic_mask = prevMask.clone();
    //         maskUsage = 0;
    //     } else if (useOpticalFlow) {
    //         cv::Mat flow;
    //         calcOpticalFlowFarneback(frame, prevFrame, flow, 0.5, 3, 20, 3, 5, 1.2, 0);
    //         propagate_mask(prevMask, dynamic_mask, flow);
    //     } else {
    //         dynamic_mask = prevMask.clone();
    //     }
    //     maskUsage++;
    //     cout << "2+++++++++++++" << endl;
    // }
    void DynamicExtractor::extractMask(const Mat &frame, Mat &dynamic_mask) {
        // if maskUsage <= masUsage, resuse prevMask
        cout << "1+++++++++++++" << endl;
        if (prevMask.empty() || maskUsage >= maxUsage) {
            prevFrame = frame.clone();
            prevMask = extractMask(frame,string("****"));
            dynamic_mask = prevMask.clone();
            maskUsage = 0;
        } else if (useOpticalFlow) {
            cv::Mat flow;
            calcOpticalFlowFarneback(frame, prevFrame, flow, 0.5, 3, 20, 3, 5, 1.2, 0);
            propagate_mask(prevMask, dynamic_mask, flow);
        } else {
            dynamic_mask = prevMask.clone();
        }
        maskUsage++;
    }

    Mat DynamicExtractor::extractMask(const Mat &frame, const string &RGB_path) {
        // ros::NodeHandle n;
        // ros::ServiceClient client = n.serviceClient<yolact_ros::Req_detct>("/detect_image");
        yolact_ros::Req_detct srv;
        srv.request.rgb_path = RGB_path;

        sensor_msgs::ImagePtr msg= cv_bridge::CvImage(std_msgs::Header(), "8UC1", frame).toImageMsg();
        srv.request.rgb = *msg;

        Mat dynamic_mask = Mat(frame.size(), CV_8U, Scalar(255));

        if (mclient.call(srv))
        {
            ROS_INFO("There are %d objects being detected!", (int)srv.response.detections.size());
        }
        else
        {
            ROS_ERROR("Failed to call service detect_image");
            return dynamic_mask;
        }

        Mat mat_zeros = Mat::zeros(frame.rows,frame.cols, CV_8U);
        
        for (int i = 0; i < (int)srv.response.detections.size(); i++) {
            float score = srv.response.detections[i].score;
            if (score > confThreshold) {
                // Extract class id
                string class_name = srv.response.detections[i].class_name;

                // Extract bounding box
                int left = srv.response.detections[i].box.x1;
                int top = srv.response.detections[i].box.y1;

                left = max(0, min(left, 640 - 1));
                top = max(0, min(top, 480 - 1));
                Rect box = Rect(left, top, srv.response.detections[i].mask.width, srv.response.detections[i].mask.height);

                // Extract the mask for the object
                Mat objectMask = Mat(srv.response.detections[i].mask.height,srv.response.detections[i].mask.width, CV_8U, Scalar(0));
                for (int j=0; j < srv.response.detections[i].mask.height; j++)
                {
                    for (int k=0; k < srv.response.detections[i].mask.width; k++)
                    {
                        objectMask.at<uchar>(j,k) = (int)srv.response.detections[i].mask.mask[j*srv.response.detections[i].mask.width + k];
                    }
                }
                
                // Resize the mask, threshold, color and apply it on the image
                // resize(objectMask, objectMask, Size(box.width, box.height));
                // threshold mask into binary 255/0 mask
                Mat mask = (objectMask > maskThreshold);
                mask.convertTo(mask, CV_8U);
                // namedWindow("output2");
                // imshow("output2",mask);
                // waitKey(0);

                if (is_dynamic_name(srv.response.detections[i].class_name)) {
                    // copy ones into the corresponding mask region
                    mat_zeros(box).copyTo(dynamic_mask(box), mask);
                }
                
            }
        }

        return dynamic_mask;
    }
}