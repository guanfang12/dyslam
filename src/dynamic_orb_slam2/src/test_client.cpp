#include "ros/ros.h"
#include "yolact_ros/Req_detct.h"
#include <cstdlib>
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace cv;
void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test_client");
  string strAssociationFilename;
  string root_path;
  if (argc != 3)
  {
    ROS_INFO("usage: rosrun test_client path_to_sequence path_to_association");
    strAssociationFilename = string("/home/wxl/Data/TUM/rgbd_dataset_freiburg3_walking_xyz/associations.txt"); 
    root_path = string("/home/wxl/Data/TUM/rgbd_dataset_freiburg3_walking_xyz");
  }
  else
  {
    strAssociationFilename = string(argv[2]);
    root_path = string(argv[1]);
    
  }

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<yolact_ros::Req_detct>("/detect_image");
  yolact_ros::Req_detct srv;

  // Retrieve paths to images
  vector <string> vstrImageFilenamesRGB;
  vector <string> vstrImageFilenamesD;
  vector<double> vTimestamps;
  
  LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

  // Check consistency in the number of images and depthmaps
  int nImages = vstrImageFilenamesRGB.size();
  if (vstrImageFilenamesRGB.empty()) {
      cerr << endl << "No images found in provided path." << endl;
      return 1;
  } else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size()) {
      cerr << endl << "Different number of images for rgb and depth." << endl;
      return 1;
  }

  string RGB_path;
  cv::Mat imRGB;
  sensor_msgs::ImagePtr msg;

  float confThreshold = 0.5; // Confidence threshold
  float maskThreshold = 0.3; // Mask threshold

  for (int ni = 0; ni < nImages; ni++){
    RGB_path = root_path + "/" + vstrImageFilenamesRGB[ni];
    imRGB = cv::imread(RGB_path, CV_LOAD_IMAGE_UNCHANGED);

    srv.request.rgb_path = RGB_path;
    msg= cv_bridge::CvImage(std_msgs::Header(), "bgr8", imRGB).toImageMsg();
    // srv.request.rgb.data = msg.get()->data;
    srv.request.rgb = *msg;
    if (client.call(srv))
    {
      ROS_INFO("There are %d objects being detected!", (int)srv.response.detections.size());
    }
    else
    {
      ROS_ERROR("Failed to call service detect_image");
      return 1;
    }
    Mat dynamic_mask = Mat(480,640, CV_8U, Scalar(255));
    Mat mat_zeros = Mat::zeros(480,640, CV_8U);
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

          // copy ones into the corresponding mask region
          mat_zeros(box).copyTo(dynamic_mask(box), mask);
      }
    }
    Mat Image = Mat(480,640, CV_8UC3, Scalar(255));
    for (int j=0; j < (int)srv.response.frame.height; j++)
    {
      for (int k=0; k < (int)srv.response.frame.width; k++)
      {
        Image.at<Vec3b>(j,k)[0] = (int)srv.response.frame.data[j*srv.response.frame.width*3 + k*3];
        Image.at<Vec3b>(j,k)[1] = (int)srv.response.frame.data[j*srv.response.frame.width*3 + k*3 + 1];
        Image.at<Vec3b>(j,k)[2] = (int)srv.response.frame.data[j*srv.response.frame.width*3 + k*3 + 2];
      }
    }
    namedWindow("output1");
    imshow("output1",Image);

    namedWindow("output2");
    imshow("output2",dynamic_mask);
    waitKey(10);
  }

  return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    if (!fAssociation) {
        cout << "can't open the association file: " << strAssociationFilename << endl;
    }
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

