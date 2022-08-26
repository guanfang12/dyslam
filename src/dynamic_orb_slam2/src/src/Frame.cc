/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

using namespace cv;

namespace ORB_SLAM2
{

cv::Scalar colorTab[] =     //10个颜色  
{
	cv::Scalar(255, 0, 0),
	cv::Scalar(0, 255, 0),
	cv::Scalar(0, 0, 255),
	cv::Scalar(255, 100, 100),
	cv::Scalar(255, 0, 255),
	cv::Scalar(0, 255, 255),
	cv::Scalar(255, 255, 0),
	cv::Scalar(255, 0, 100),
	cv::Scalar(100, 100, 100),
	cv::Scalar(50, 125, 125),

	cv::Scalar(255, 100, 0),
	cv::Scalar(0, 125, 100),
	cv::Scalar(100, 0, 255),
	cv::Scalar(155, 211, 255),      // Burlywood1
	cv::Scalar(106, 106, 255),    // IndianRed1
	cv::Scalar(144, 128, 112),    //  SlateGrey
	cv::Scalar(105, 105, 105),   // DimGrey
	cv::Scalar(79, 79, 47),      // DarkSlateGray
	cv::Scalar(0, 0, 0),        // Black
	cv::Scalar(255, 250, 250), // 	Snow 

	cv::Scalar(55, 155, 0),
	cv::Scalar(55, 0, 100),
	cv::Scalar(100, 0, 50),
	cv::Scalar(50, 25, 25)
};

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    string RGB_path;
    thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
    thread threadRight(&Frame::ExtractORB, this, 0, imRight);
    
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mDepth(imDepth)
{
    // std::thread* mptGrow = new thread(&ORB_SLAM2::Frame::Run, this);

    // {
    //     unique_lock<mutex> lock(mMutexAccept);
    //     mbNewDepth = true;
    // }
    // mbNewDepth = true;
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    cout << "1------------" << endl;

    // ORB extraction
    ExtractORB(0,imGray);
    cout << "2------------" << endl;

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    string RGB_path;
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    cv::Mat mask = cv::Mat(im.size(), CV_8U, cv::Scalar(255));
    if(flag==0) {
        // if (mpORBextractorLeft->remove_dynamic)
        //     mpORBextractorLeft->extractDynamicMask(im, mask);

        
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        (*mpORBextractorLeft)(im, mask, mvKeys, mDescriptors);
        // Run();
        // cout << "mbNewDepth: " << mbNewDepth << endl; 
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        {            
            unique_lock<mutex> lock(mpORBextractorLeft->mMutexDynamic);
            mMmask = mpORBextractorLeft->mMmask;
            // namedWindow("semantic mask");
            imshow("semantic mask",mMmask);
        }        
        // {
        //     unique_lock<mutex> lock(mMutexGrow);
        // }
        // RequestFinish();
        // 使用区域生长分割结果补全语义分割掩码
        Mat temp, mask;
        Mat mat_zeros = Mat::zeros(mDepth.rows,mDepth.cols, CV_8U);
        for(int i = 2; i < mclusters_num.size(); i++)
        {
            if(mclusters_num[i] < 1200) continue;
            temp = Mat(mDepth.size(), CV_8U, Scalar(255));
            mask = (mshowLabel == i);
            mMmask.copyTo(temp, mask);
            // 统计语义掩码与区域生长分割交集所占比率
            int count=mDepth.rows*mDepth.cols-countNonZero(temp);
            int count_semantic=mDepth.rows*mDepth.cols-countNonZero(mMmask);
            // cout << "count: " << (float)count/(float)clusters_num[2] << endl;
            if((float)count/(float)mclusters_num[2] >= 0.4 || (float)count/(float)count_semantic >= 0.4)
            mat_zeros.copyTo(mMmask, mask);
        }
        imshow("completetion_mask",mMmask);

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();

        // wxl write
        // Mat temp;
        temp = im.clone();
        const float r = 5;
        cv::cvtColor(temp,temp,CV_GRAY2BGR);

        //　根据mask删除动态特征点，需要讲特征点和描述子均删去
        //　特征点为vector向量，描述子为cv::Mat
        int i = 0;
        set<int> del_rows;
        for(vector<cv::KeyPoint>::iterator it=mvKeys.begin();it!=mvKeys.end();)
        {
            int x = it->pt.x;
            int y = it->pt.y;
            if(mMmask.at<uchar>(y,x) == 0)
            {
                // auto iter = mDescriptors.erase(std::begin(mDescriptors)+i);
                cv::Point2f pt1,pt2;
                pt1.x=x-r;
                pt1.y=y-r;
                pt2.x=x+r;
                pt2.y=y+r;

                // cv::rectangle(temp,pt1,pt2,cv::Scalar(0,0,255));
                cv::circle(temp,it->pt,2,cv::Scalar(0,0,255),-1);

                it=mvKeys.erase(it);
                del_rows.insert(i);
            }
            else
            {                
                // cv::rectangle(temp,pt1,pt2,cv::Scalar(0,255,0));
                cv::circle(temp,it->pt,2,cv::Scalar(0,255,0),-1);

                it++;
            }
            i++;
            
        }

        // cv::namedWindow("semantic cull");
        cv::imshow("semantic cull",temp);
        // cv::waitKey(0);

        Mat temp2;
        for(int i = 0; i < mDescriptors.rows; i++)
        {
            if(!del_rows.count(i))
            {
                temp2.push_back(mDescriptors.row(i));
            }
        }
        mDescriptors = temp2;


        #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        #else
                std::chrono::monotonic_clock::time_point t4 = std::chrono::monotonic_clock::now();
        #endif
        cout << "ORB extractor use: " << std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count() << "s" << endl;
        cout << "Get mask use: " << std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count() << "s" << endl;
        cout << "Cull features use: " << std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count() << "s" << endl;
        cout << "All use: " << std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t1).count() << "s" << endl;
        cout<<"After Key size: "<<mvKeys.size()<<"; and Mat Height: " << mDescriptors.size().height<<endl;
    }
    else {
        if (mpORBextractorRight->remove_dynamic)
            mpORBextractorRight->extractDynamicMask(im, mask);
        (*mpORBextractorRight)(im, mask, mvKeysRight, mDescriptorsRight);
    }
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

double get_time(std::chrono::steady_clock::time_point t1, std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration_cast<std::chrono::duration<double> >(t1 - t0).count();
}

float getDistance2(cv::Vec3f t1, cv::Vec3f t2)
{
	double dx = powf((t1[0] - t2[0]), 2);
	double dy = powf((t1[1] - t2[1]), 2);
	double dz = powf((t1[2] - t2[2]), 2);
	return sqrt(dx + dy + dz);
}

void area_grow(Mat depth,cv::Mat &showLabel, std::vector<int> &clusters, float th)
{	
    /**********
     * depth: 待分割深度图
     * th: 区域生长阈值
    **********/

    cout << "begin!" << endl;   
	std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    // std::vector<int> clusters;
    clusters.push_back(1);
    clusters.push_back(1);
	// cv::Mat depth = cv::imread("../python/images/hope_depth.png",-1);	
	int imgHeight = depth.rows;
	int imgWidth = depth.cols;
	int imgSize = imgHeight*imgWidth;
    int sig = 0;
    int occupy[imgSize];
    int new_seed_index = 0;

    // 相机内参
	float fx = 535.4;
	float fy = 539.2;
	float cx = 320.1;
	float cy = 247.6;

    Mat pcs = Mat::zeros(imgHeight,imgWidth,CV_32FC3);
    float *pInput = (float*)pcs.data;
    // cv::Mat showLabel = Mat::ones(imgHeight,imgWidth,CV_8UC1);
    // showLabel.setTo(-1);
    Point seed;
	for(int v=0; v < imgHeight; v++)
	{
		for(int u = 0; u<imgWidth; u++)
		{
			if(depth.at<float>(v,u) == 0 || depth.at<float>(v,u) >= 3.6) {occupy[v*imgWidth + u] = 1; clusters[1]++; continue;}
            int idx = (v*imgWidth + u)*3;
            pInput[idx + 2] = depth.at<float>(v,u);
			pInput[idx + 1] = (v - cy) / fy*pcs.at<cv::Vec3f>(v,u)[2];
			pInput[idx + 0] = (u - cx) / fx*pcs.at<cv::Vec3f>(v,u)[2];
            showLabel.at<uchar>(v,u) = 0;
            if(sig == 0)
            {
                seed.x = u;
                seed.y = v;
                new_seed_index = v*imgWidth + u;
                sig = 1;
            }            
		}
	}    
    // cout << seed << endl; 	
  
    int label = 2;  // 下一个新类的标签，背景标签为 0,未分类标签为 1
    Point waitSeed; // 潜在新 seed
    int waitSeed_label; // 潜在新 seed 的标签
    int num_cluster = 1; // 用于统计每一个簇内点的个数
    vector<Point> seedVector; // 待探索 seed
    int direct[4][2] = { {0,-1},{1,0}, {0,1}, {-1,0} };   //4邻域,应该用4邻域减小时间复杂度
    int count = 0; // 统计循环次数
    while(1)
    {
        seedVector.push_back(seed);
        showLabel.at<uchar>(Point(seed.x, seed.y)) = label;
        occupy[seed.y*imgWidth + seed.x] = 1;
        while (!seedVector.empty())     //种子栈不为空则生长，即遍历栈中所有元素后停止生长
        {
            // cout<< "seedVector: " << seedVector.size() << endl;
            seed = seedVector.back();     //取出最后一个元素
            seedVector.pop_back();         //删除栈中最后一个元素,防止重复扫描           

            for (int i = 0; i < 4; i++)    //遍历种子点的4邻域
            {
                waitSeed.x = seed.x + direct[i][0];    //第i个坐标0行，即x坐标值
                waitSeed.y = seed.y + direct[i][1];    //第i个坐标1行，即y坐标值

                //检查是否是边缘点
                if (waitSeed.x < 0 || waitSeed.y < 0 ||
                    waitSeed.x >(imgWidth - 1) || (waitSeed.y > imgHeight - 1))
                    continue;
                waitSeed_label = showLabel.at<uchar>(Point(waitSeed.x, waitSeed.y));   // 待生长种子点现有标签，判断是否可以生长
                // cout<< "waitSeed_label: " << waitSeed_label << endl;
                if (waitSeed_label == 0)     //判断waitSeed是否已经被生长，避免重复生长造成死循环
                {
                    float dx_dy = abs( direct[i][1]/ fy)*pcs.at<cv::Vec3f>(seed.y, seed.x)[2]*0.7; 
                    float dist = getDistance2(pcs.at<cv::Vec3f>(Point(seed.x, seed.y)), pcs.at<cv::Vec3f>(Point(waitSeed.x, waitSeed.y)));

                    if (dist <= th + dx_dy)     //区域生长条件
                    {
                        showLabel.at<uchar>(Point(waitSeed.x, waitSeed.y)) = label;
                        occupy[waitSeed.y*imgWidth + waitSeed.x] = 1;
                        seedVector.push_back(waitSeed);    //将满足生长条件的待生长种子点放入种子栈中
                        num_cluster++;
                    }
                }
            }
        }
        if(num_cluster >= 700)  {clusters.push_back(num_cluster); label++; num_cluster = 1;}
        sig = 0; 
        for( ; new_seed_index<imgSize; new_seed_index++)
        {
            if(occupy[new_seed_index] == 0)
            {
                seed.x = new_seed_index%imgWidth;
                seed.y = new_seed_index/imgWidth;
                sig = 1;
                break;
            }
        }
        // cout << num_cluster << endl;
        // cout << label << endl;
        // cout << seed << endl;
        // if(label == 25) break;
        if(sig == 0) {cout<<"OUT!!!" << endl; break;}
        
    }
 
    // cout << seed << endl;
    cout << "There are "<< label-1 << " clusters!" << endl;

    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    cout << "All grow use time: " << get_time(t3, t0) << endl;
}

void Frame::Run()
{
    // mbFinishRequested =false;
    mbNewDepth = true;
    while(1)
    {
        if(CheckAccept())
        {
            // unique_lock<mutex> lock(mMutexGrow);
            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            mshowLabel = Mat::ones(mDepth.rows,mDepth.cols,CV_8UC1);
            area_grow(mDepth, mshowLabel, mclusters_num, 0.016);

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            // 绘制区域生长分割结果，用于可视化
            cv::Mat showColor = Mat::zeros(mDepth.rows,mDepth.cols,CV_8UC3);
            for (int i = 0; i < mDepth.rows; i++)
            {
                for (int j = 0; j < mDepth.cols; j++)
                {
                    //标记像素点的类别，颜色区分
                    int k = mshowLabel.at<uchar>(Point(j, i))%24 - 1;
                    if(k==0 && mshowLabel.at<uchar>(Point(j, i))!=1)
                    k = 10;              
                    if(mclusters_num[mshowLabel.at<uchar>(Point(j, i))] < 200) // 1200
                    cv::circle(showColor, cv::Point(j, i), 1, colorTab[0]);            
                    else
                    cv::circle(showColor, cv::Point(j, i), 1, colorTab[k]); 
                }
            }
        
            cout << "ALL time: " << get_time(t1, t0) << endl;

            cv::imshow("showColor_show",showColor);
            // cv::waitKey(0);
            mbNewDepth = false;
            // cout << "mbNewDepth = false? " << mbNewDepth << endl;
            break;
        }
        // if(CheckFinish())
        //     break;
    } 
}

bool Frame::CheckAccept()
{
    // unique_lock<mutex> lock(mMutexAccept);
    return mbNewDepth;
}
bool Frame::CheckFinish()
{
    // unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Frame::RequestFinish()
{
    // unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

} //namespace ORB_SLAM
