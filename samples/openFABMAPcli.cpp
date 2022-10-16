/*//////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// This file originates from the openFABMAP project:
// [http://code.google.com/p/openfabmap/] -or-
// [https://github.com/arrenglover/openfabmap]
//
// For published work which uses all or part of OpenFABMAP, please cite:
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6224843]
//
// Original Algorithm by Mark Cummins and Paul Newman:
// [http://ijr.sagepub.com/content/27/6/647.short]
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942]
// [http://ijr.sagepub.com/content/30/9/1100.abstract]
//
//                           License Agreement
//
// Copyright (C) 2012 Arren Glover [aj.glover@qut.edu.au] and
//                    Will Maddern [w.maddern@qut.edu.au], all rights reserved.
//
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistribution's of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * The name of the copyright holders may not be used to endorse or promote
//    products derived from this software without specific prior written
///   permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability,or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//////////////////////////////////////////////////////////////////////////////*/
#include <iostream>//
#include <openfabmap.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION == 2 and CV_MINOR_VERSION == 3

#elif CV_MAJOR_VERSION == 2 and CV_MINOR_VERSION == 4
#if USENONFREE
#include <opencv2/nonfree/nonfree.hpp>
#endif
#elif CV_MAJOR_VERSION == 3
#ifdef USENONFREE
#include <opencv2/xfeatures2d.hpp>
#endif
#endif

#include <fstream>
#include <iostream>
using namespace std;//
/*
openFABMAP procedural functions
*/
int help(void);
int showFeatures(std::string trainPath,
                 cv::Ptr<cv::FeatureDetector> &detector);
int generateVocabTrainData(std::string trainPath,
                           std::string vocabTrainDataPath,
                           cv::Ptr<cv::FeatureDetector> &detector,
                           cv::Ptr<cv::DescriptorExtractor> &extractor);
int trainVocabulary(std::string vocabPath,
                    std::string vocabTrainDataPath,
                    double clusterRadius);

int generateBOWImageDescs(std::string dataPath,
                          std::string bowImageDescPath,
                          std::string vocabPath,
                          cv::Ptr<cv::FeatureDetector> &detector,
                          cv::Ptr<cv::DescriptorExtractor> &extractor,
                          int minWords);

int trainChowLiuTree(std::string chowliutreePath,
                     std::string fabmapTrainDataPath,
                     double lowerInformationBound);

int openFABMAP(std::string testPath,
               of2::FabMap *openFABMAP,
               std::string vocabPath,
               std::string resultsPath,
               bool addNewOnly);

/*
helper functions
*/
of2::FabMap *generateFABMAPInstance(cv::FileStorage &settings);
cv::Ptr<cv::FeatureDetector> generateDetector(cv::FileStorage &fs);
cv::Ptr<cv::DescriptorExtractor> generateExtractor(cv::FileStorage &fs);

/*
Advanced tools for keypoint manipulation. These tools are not currently in the
functional code but are available for use if desired.
*/
void drawRichKeypoints(const cv::Mat& src, std::vector<cv::KeyPoint>& kpts,
                       cv::Mat& dst);
void filterKeypoints(std::vector<cv::KeyPoint>& kpts, int maxSize = 0,
                     int maxFeatures = 0);
void sortKeypoints(std::vector<cv::KeyPoint>& keypoints);


/*
The openFabMapcli accepts a YML settings file, an example of which is provided.
Modify options in the settings file for desired operation
*/
int main(int argc, char * argv[])
{
    //load the settings file
    std::string settfilename;
    if (argc == 1) {
        //assume settings in working directory
        settfilename = "settings.yml";  //载入设置文件
    } else if (argc == 3) {
        if(std::string(argv[1]) != "-s") {
            //incorrect option
            return help();
        } else {
            //settings provided as argument
            settfilename = std::string(argv[2]);
        }
    } else {
        //incorrect arguments
        return help();
    }

    cv::FileStorage fs;
    fs.open(settfilename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open settings file: " << settfilename <<
                     std::endl;
        return -1;
    }

    cv::Ptr<cv::FeatureDetector> detector = generateDetector(fs); //特征检测，由设置文件给定算法参数
    if(!detector) {
        std::cerr << "Feature Detector error" << std::endl;
        return -1;
    }

    cv::Ptr<cv::DescriptorExtractor> extractor = generateExtractor(fs); //计算描述子，由设置文件给定算法参数
    if(!extractor) {
        std::cerr << "Feature Extractor error" << std::endl;
        return -1;
    }

    //run desired function
    int result = 0;
    std::string function = fs["Function"];
    if (function == "ShowFeatures") {
        result = showFeatures(
                    fs["FilePaths"]["TrainPath"],
                detector); //显示在训练视频中检测到的特征.将设置文件中的Function设置为ShowFeatures即可运行这一步,下面同理。

    } else if (function == "GenerateVocabTrainData") {
        result = generateVocabTrainData(fs["FilePaths"]["TrainPath"],
                fs["FilePaths"]["TrainFeatDesc"],
                detector, extractor);  //获得生成词典所需训练数据(多个描述子)

    } else if (function == "TrainVocabulary") {
        result = trainVocabulary(fs["FilePaths"]["Vocabulary"],
                fs["FilePaths"]["TrainFeatDesc"],
                fs["VocabTrainOptions"]["ClusterSize"]); //利用上述数据生成词典

    } else if (function == "GenerateFABMAPTrainData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TrainPath"],
                fs["FilePaths"]["TrainImagDesc"],
                fs["FilePaths"]["Vocabulary"], detector, extractor,
                fs["BOWOptions"]["MinWords"]);  //计算每张训练图片的BOW特征描述子

    } else if (function == "TrainChowLiuTree") {
        result = trainChowLiuTree(fs["FilePaths"]["ChowLiuTree"],
                fs["FilePaths"]["TrainImagDesc"],
                fs["ChowLiuOptions"]["LowerInfoBound"]);  //利用训练数据生成周柳树

    } else if (function == "GenerateFABMAPTestData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TestPath"],
                fs["FilePaths"]["TestImageDesc"],
                fs["FilePaths"]["Vocabulary"], detector, extractor,
                fs["BOWOptions"]["MinWords"]);  //计算每张测试图片的BOW特征描述子

    } else if (function == "RunOpenFABMAP") {
        std::string placeAddOption = fs["FabMapPlaceAddition"];
        bool addNewOnly = (placeAddOption == "NewMaximumOnly");
        of2::FabMap *fabmap = generateFABMAPInstance(fs);
        if(fabmap) {
            result = openFABMAP(fs["FilePaths"]["TestImageDesc"], fabmap,
                    fs["FilePaths"]["Vocabulary"],
                    fs["FilePaths"]["FabMapResults"], addNewOnly);  //进行位置估计，并将结果储存在一个矩阵里。（用matlab将结果打开，图中存在的斜线即为回环）
        }

    } else {
        std::cerr << "Incorrect Function Type" << std::endl;
        result = -1;
    }

    std::cout << "openFABMAP done" << std::endl;
    std::cin.sync(); std::cin.ignore();

    fs.release();
    return result;

}

/*
displays the usage message
*/
int help(void)
{
    std::cout << "Usage: openFABMAPexe -s settingsfile" << std::endl;
    return 0;
}

/*
shows the features detected on the training video
*/
int showFeatures(std::string trainPath, cv::Ptr<cv::FeatureDetector> &detector) //显示在训练视频中检测到的特征
{

    //open the movie
    cv::VideoCapture movie;
    movie.open(trainPath);

    if (!movie.isOpened()) {
        std::cerr << trainPath << ": training movie not found" << std::endl;
        return -1;
    }

    std::cout << "Press Esc to Exit" << std::endl;
    cv::Mat frame, kptsImg;

    movie.read(frame);
    std::vector<cv::KeyPoint> kpts;
    while (movie.read(frame)) {
        detector->detect(frame, kpts);

        std::cout << kpts.size() << " keypoints detected...         \r";
        fflush(stdout);

        cv::drawKeypoints(frame, kpts, kptsImg);

        cv::imshow("Features", kptsImg);
        if(cv::waitKey(5) == 27) {
            break;
        }
    }
    std::cout << std::endl;

    cv::destroyWindow("Features");
    return 0;
}

/*
generate the data needed to train a codebook/vocabulary for bag-of-words methods
*/ //获得生成词典所需训练数据(多个描述子)
int generateVocabTrainData(std::string trainPath,
                           std::string vocabTrainDataPath,
                           cv::Ptr<cv::FeatureDetector> &detector,
                           cv::Ptr<cv::DescriptorExtractor> &extractor)
{

    //Do not overwrite any files
    std::ifstream checker;
    checker.open(vocabTrainDataPath.c_str());
    if(checker.is_open()) {
        std::cerr << vocabTrainDataPath << ": Training Data already present" <<
                     std::endl;
        checker.close();
        return -1;
    }

    //load training movie
    cv::VideoCapture movie;
    movie.open(trainPath);
    if (!movie.isOpened()) {
        std::cerr << trainPath << ": training movie not found" << std::endl;
        return -1;
    }

    //extract data
    std::cout << "Extracting Descriptors" << std::endl;
    cv::Mat vocabTrainData;
    cv::Mat frame, descs, feats;
    std::vector<cv::KeyPoint> kpts;

    std::cout.setf(std::ios_base::fixed); //设置固定的浮点位数显示
    std::cout.precision(0);//设置精度，在第n位四舍五入等。

    while(movie.read(frame)) {            //获得每张训练图片的描述子并将其压入矩阵中

        //detect & extract features
        detector->detect(frame, kpts);
        extractor->compute(frame, kpts, descs);

        //add all descriptors to the training data
        vocabTrainData.push_back(descs);

        //show progress
        cv::drawKeypoints(frame, kpts, feats);
        cv::imshow("Training Data", feats);

        std::cout << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
                            movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%. " <<
                     vocabTrainData.rows << " descriptors         \r";
        fflush(stdout);

        if(cv::waitKey(5) == 27) {
            cv::destroyWindow("Training Data");
            std::cout << std::endl;
            return -1;
        }

    }
    cv::destroyWindow("Training Data");
    std::cout << "Done: " << vocabTrainData.rows << " Descriptors" << std::endl;

    //save the training data
    cv::FileStorage fs;
    fs.open(vocabTrainDataPath, cv::FileStorage::WRITE);
    fs << "VocabTrainData" << vocabTrainData;
    fs.release();

    return 0;
}

/*
use training data to build a codebook/vocabulary
*/ //利用训练数据集生成词典
int trainVocabulary(std::string vocabPath,
                    std::string vocabTrainDataPath,
                    double clusterRadius)
{

    //ensure not overwriting a vocabulary　判断词典是否已经存在
    std::ifstream checker;
    checker.open(vocabPath.c_str());
    if(checker.is_open()) {
        std::cerr << vocabPath << ": Vocabulary already present" <<
                     std::endl;
        checker.close();
        return -1;
    }

    std::cout << "Loading vocabulary training data" << std::endl;

    cv::FileStorage fs;

    //load in vocab training data  载入训练数据（描述子矩阵）给vocabTrainData
    fs.open(vocabTrainDataPath, cv::FileStorage::READ);
    cv::Mat vocabTrainData;
    fs["VocabTrainData"] >> vocabTrainData;
    if (vocabTrainData.empty()) {
        std::cerr << vocabTrainDataPath << ": Training Data not found" <<
                     std::endl;
        return -1;
    }
    fs.release();

    std::cout << "Performing clustering" << std::endl;

    //uses Modified Sequential Clustering to train a vocabulary  使用改进的序列聚类来训练词典
    of2::BOWMSCTrainer trainer(clusterRadius);
    trainer.add(vocabTrainData);  //添加训练数据
    cv::Mat vocab = trainer.cluster();  //聚类，生成词典

    //save the vocabulary 储存词典
    std::cout << "Saving vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::WRITE);
    fs << "Vocabulary" << vocab;
    fs.release();

    return 0;
}

/*
generate FabMap bag-of-words data : an image descriptor for each frame
*/
//计算每张训练图片的BOW特征描述子
int generateBOWImageDescs(std::string dataPath,
                          std::string bowImageDescPath,
                          std::string vocabPath,
                          cv::Ptr<cv::FeatureDetector> &detector,
                          cv::Ptr<cv::DescriptorExtractor> &extractor,
                          int minWords)
{

    cv::FileStorage fs;

    //ensure not overwriting training data
    std::ifstream checker;
    checker.open(bowImageDescPath.c_str());
    if(checker.is_open()) {
        std::cerr << bowImageDescPath << ": FabMap Training/Testing Data "
                                         "already present" << std::endl;
        checker.close();
        return -1;
    }

    //load vocabulary 载入词典
    std::cout << "Loading Vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::READ);
    cv::Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
        return -1;
    }
    fs.release();

    //use a FLANN matcher to generate bag-of-words representations
    cv::Ptr<cv::DescriptorMatcher> matcher =
            cv::DescriptorMatcher::create("FlannBased");
    cv::BOWImgDescriptorExtractor bide(extractor, matcher); //直方图统计
    bide.setVocabulary(vocab);  //使用上一步得到的词典

    //load movie
    cv::VideoCapture movie;
    movie.open(dataPath);

    if(!movie.isOpened()) {
        std::cerr << dataPath << ": movie not found" << std::endl;
        return -1;
    }

    //extract image descriptors
    cv::Mat fabmapTrainData;
    std::cout << "Extracting Bag-of-words Image Descriptors" << std::endl;
    std::cout.setf(std::ios_base::fixed);
    std::cout.precision(0);

    std::ofstream maskw; //

    if(minWords) {
        maskw.open(std::string(bowImageDescPath + "mask.txt").c_str());
    }

    cv::Mat frame, bow;
    std::vector<cv::KeyPoint> kpts;

    while(movie.read(frame)) {
        detector->detect(frame, kpts);
        bide.compute(frame, kpts, bow); //计算当前图像的BOW特征描述子

        if(minWords) {
            //writing a mask file
            if(cv::countNonZero(bow) < minWords) {
                //frame masked
                maskw << "0" << std::endl;
            } else {
                //frame accepted
                maskw << "1" << std::endl;
                fabmapTrainData.push_back(bow);
            }
        } else {
            fabmapTrainData.push_back(bow);
        }

        std::cout << 100.0 * (movie.get(CV_CAP_PROP_POS_FRAMES) /
                              movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%    \r";
        fflush(stdout);
    }
    std::cout << "Done                                       " << std::endl;

    movie.release();

    //save training data
    fs.open(bowImageDescPath, cv::FileStorage::WRITE);
    fs << "BOWImageDescs" << fabmapTrainData;
    fs.release();

    return 0;
}

/*
generate a Chow-Liu tree from FabMap Training data
*/ //利用训练数据生成周柳树
int trainChowLiuTree(std::string chowliutreePath,
                     std::string fabmapTrainDataPath,
                     double lowerInformationBound)
{

    cv::FileStorage fs;

    //ensure not overwriting training data　判断周柳树文件是否已存在
    std::ifstream checker;
    checker.open(chowliutreePath.c_str());
    if(checker.is_open()) {
        std::cerr << chowliutreePath << ": Chow-Liu Tree already present" <<
                     std::endl;
        checker.close();
        return -1;
    }

    //load FabMap training data  载入训练数据
    std::cout << "Loading FabMap Training Data" << std::endl;
    fs.open(fabmapTrainDataPath, cv::FileStorage::READ);
    cv::Mat fabmapTrainData;
    fs["BOWImageDescs"] >> fabmapTrainData;
    if (fabmapTrainData.empty()) {
        std::cerr << fabmapTrainDataPath << ": FabMap Training Data not found"
                  << std::endl;
        return -1;
    }
    fs.release();

    //generate the tree from the data 生成周柳树
    std::cout << "Making Chow-Liu Tree" << std::endl;
    of2::ChowLiuTree tree; 
    tree.add(fabmapTrainData);//添加训练数据
    cv::Mat clTree = tree.make(lowerInformationBound);

    //save the resulting tree
    std::cout <<"Saving Chow-Liu Tree" << std::endl;
    fs.open(chowliutreePath, cv::FileStorage::WRITE);
    fs << "ChowLiuTree" << clTree;
    fs.release();

    return 0;

}


/*
Run FabMap on a test dataset
*/ //在测试数据集上进行位置估计
int openFABMAP(std::string testPath,
               of2::FabMap *fabmap,
               std::string vocabPath,
               std::string resultsPath,
               bool addNewOnly)
{

    cv::FileStorage fs;

    //ensure not overwriting results //判断结果文件是否已经存在
    std::ifstream checker;
    checker.open(resultsPath.c_str());
    if(checker.is_open()) {
        std::cerr << resultsPath << ": Results already present" << std::endl;
        checker.close();
        return -1;
    }

    //load the vocabulary　加载上一步训练好的词典
    std::cout << "Loading Vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::READ);
    cv::Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
        return -1;
    }
    fs.release();

    //load the test data　加载测试数据集
    fs.open(testPath, cv::FileStorage::READ);
    cv::Mat testImageDescs;
    fs["BOWImageDescs"] >> testImageDescs;
    if(testImageDescs.empty()) {
        std::cerr << testPath << ": Test data not found" << std::endl;
        return -1;
    }
    fs.release();

    //running openFABMAP
    std::cout << "Running openFABMAP" << std::endl;
    std::vector<of2::IMatch> matches;
    std::vector<of2::IMatch>::iterator l;



    cv::Mat confusion_mat(testImageDescs.rows, testImageDescs.rows, CV_64FC1);
    confusion_mat.setTo(0); // init to 0's


    if (!addNewOnly) 
    {

        //automatically comparing a whole dataset  自动比较整个数据集
        fabmap->localize(testImageDescs, matches, true); //进行定位，计算位置估计

        for(l = matches.begin(); l != matches.end(); l++) //将计算结果写入矩阵中
        {
            if(l->imgIdx < 0) {
                confusion_mat.at<double>(l->queryIdx, l->queryIdx) = l->match;

            } else {
                confusion_mat.at<double>(l->queryIdx, l->imgIdx) = l->match;
            }
        }

    } 
    else 
    {

        //criteria for adding locations used
        for(int i = 0; i < testImageDescs.rows; i++) {
            matches.clear();
            //compare images individually
            fabmap->localize(testImageDescs.row(i), matches);

            bool new_place_max = true;
            for(l = matches.begin(); l != matches.end(); l++) {

                if(l->imgIdx < 0) {
                    //add the new place to the confusion matrix 'diagonal'
                    confusion_mat.at<double>(i, (int)matches.size()-1) = l->match;

                } else {
                    //add the score to the confusion matrix
                    confusion_mat.at<double>(i, l->imgIdx) = l->match;
                }

                //test for new location maximum
                if(l->match > matches.front().match) {
                    new_place_max = false;
                }
            }

            if(new_place_max) {
                fabmap->add(testImageDescs.row(i));
            }
        }
    }

    //save the result as plain text for ease of import to Matlab 将结果保存为纯文本以便于导入到Matlab
    std::ofstream writer(resultsPath.c_str());
    for(int i = 0; i < confusion_mat.rows; i++)  //将结果矩阵写入文件
    {
        for(int j = 0; j < confusion_mat.cols; j++) 
        {
            writer << confusion_mat.at<double>(i, j) << " ";
        }
        writer << std::endl;
    }
    writer.close();

    return 0;
}

#if CV_MAJOR_VERSION == 2
/*
generates a feature detector based on options in the settings file
*/
cv::Ptr<cv::FeatureDetector> generateDetector(cv::FileStorage &fs) {

    //create common feature detector and descriptor extractor
    std::string detectorMode = fs["FeatureOptions"]["DetectorMode"];
    std::string detectorType = fs["FeatureOptions"]["DetectorType"];
    cv::Ptr<cv::FeatureDetector> detector = NULL;
    if(detectorMode == "ADAPTIVE") {

        if(detectorType != "STAR" && detectorType != "SURF" &&
                detectorType != "FAST") {
            std::cerr << "Adaptive Detectors only work with STAR, SURF "
                         "and FAST" << std::endl;
        } else {

            detector = new cv::DynamicAdaptedFeatureDetector(
                        cv::AdjusterAdapter::create(detectorType),
                        fs["FeatureOptions"]["Adaptive"]["MinFeatures"],
                    fs["FeatureOptions"]["Adaptive"]["MaxFeatures"],
                    fs["FeatureOptions"]["Adaptive"]["MaxIters"]);
        }

    } else if(detectorMode == "STATIC") {
        if(detectorType == "STAR") {

            detector = new cv::StarFeatureDetector(
                        fs["FeatureOptions"]["StarDetector"]["MaxSize"],
                    fs["FeatureOptions"]["StarDetector"]["Response"],
                    fs["FeatureOptions"]["StarDetector"]["LineThreshold"],
                    fs["FeatureOptions"]["StarDetector"]["LineBinarized"],
                    fs["FeatureOptions"]["StarDetector"]["Suppression"]);

        } else if(detectorType == "FAST") {

            detector = new cv::FastFeatureDetector(
                        fs["FeatureOptions"]["FastDetector"]["Threshold"],
                    (int)fs["FeatureOptions"]["FastDetector"]
                    ["NonMaxSuppression"] > 0);
        } else if(detectorType == "MSER") {

            detector = new cv::MserFeatureDetector(
                        fs["FeatureOptions"]["MSERDetector"]["Delta"],
                    fs["FeatureOptions"]["MSERDetector"]["MinArea"],
                    fs["FeatureOptions"]["MSERDetector"]["MaxArea"],
                    fs["FeatureOptions"]["MSERDetector"]["MaxVariation"],
                    fs["FeatureOptions"]["MSERDetector"]["MinDiversity"],
                    fs["FeatureOptions"]["MSERDetector"]["MaxEvolution"],
                    fs["FeatureOptions"]["MSERDetector"]["AreaThreshold"],
                    fs["FeatureOptions"]["MSERDetector"]["MinMargin"],
                    fs["FeatureOptions"]["MSERDetector"]["EdgeBlurSize"]);
#if USENONFREE
        } else if(detectorType == "SURF") {

#if CV_MINOR_VERSION == 4
            detector = new cv::SURF(
                        fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                    (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                    (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

#elif CV_MINOR_VERSION == 3
            detector = new cv::SurfFeatureDetector(
                        fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                    (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
#endif
        } else if(detectorType == "SIFT") {
#if CV_MINOR_VERSION == 4
            detector = new cv::SIFT(
                        fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                    fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                    fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                    fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                    fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#elif CV_MINOR_VERSION == 3
            detector = new cv::SiftFeatureDetector(
                        fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                    fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"]);
#endif
#endif //USENONFREE

        } else {
            std::cerr << "Could not create detector class. Specify detector "
                         "options in the settings file" << std::endl;
        }
    } else {
        std::cerr << "Could not create detector class. Specify detector "
                     "mode (static/adaptive) in the settings file" << std::endl;
    }

    return detector;

}
#elif CV_MAJOR_VERSION == 3
/*
generates a feature detector based on options in the settings file
*/
cv::Ptr<cv::FeatureDetector> generateDetector(cv::FileStorage &fs) {

    //create common feature detector and descriptor extractor
    std::string detectorType = fs["FeatureOptions"]["DetectorType"];

    if(detectorType == "BRISK") {
        return cv::BRISK::create(
                    fs["FeatureOptions"]["BRISK"]["Threshold"],
                fs["FeatureOptions"]["BRISK"]["Octaves"],
                fs["FeatureOptions"]["BRISK"]["PatternScale"]);
    } else if(detectorType == "ORB") {
        return cv::ORB::create(
                    fs["FeatureOptions"]["ORB"]["nFeatures"],
                fs["FeatureOptions"]["ORB"]["scaleFactor"],
                fs["FeatureOptions"]["ORB"]["nLevels"],
                fs["FeatureOptions"]["ORB"]["edgeThreshold"],
                fs["FeatureOptions"]["ORB"]["firstLevel"],
                2, cv::ORB::HARRIS_SCORE,
                fs["FeatureOptions"]["ORB"]["patchSize"]);

    } else if(detectorType == "MSER") {
        return cv::MSER::create(
                    fs["FeatureOptions"]["MSERDetector"]["Delta"],
                fs["FeatureOptions"]["MSERDetector"]["MinArea"],
                fs["FeatureOptions"]["MSERDetector"]["MaxArea"],
                fs["FeatureOptions"]["MSERDetector"]["MaxVariation"],
                fs["FeatureOptions"]["MSERDetector"]["MinDiversity"],
                fs["FeatureOptions"]["MSERDetector"]["MaxEvolution"],
                fs["FeatureOptions"]["MSERDetector"]["AreaThreshold"],
                fs["FeatureOptions"]["MSERDetector"]["MinMargin"],
                fs["FeatureOptions"]["MSERDetector"]["EdgeBlurSize"]);
    } else if(detectorType == "FAST") {
        return cv::FastFeatureDetector::create(
                    fs["FeatureOptions"]["FastDetector"]["Threshold"],
                (int)fs["FeatureOptions"]["FastDetector"]["NonMaxSuppression"] > 0);
    } else if(detectorType == "AGAST") {
        return cv::AgastFeatureDetector::create(
                    fs["FeatureOptions"]["AGAST"]["Threshold"],
                (int)fs["FeatureOptions"]["AGAST"]["NonMaxSuppression"] > 0);
#ifdef USENONFREE
    } else if(detectorType == "STAR") {

        return cv::xfeatures2d::StarDetector::create(
                    fs["FeatureOptions"]["StarDetector"]["MaxSize"],
                fs["FeatureOptions"]["StarDetector"]["Response"],
                fs["FeatureOptions"]["StarDetector"]["LineThreshold"],
                fs["FeatureOptions"]["StarDetector"]["LineBinarized"],
                fs["FeatureOptions"]["StarDetector"]["Suppression"]);

    } else if(detectorType == "SURF") {
        return cv::xfeatures2d::SURF::create(
                    fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

    } else if(detectorType == "SIFT") {
        return cv::xfeatures2d::SIFT::create(
                    fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#endif

    } else {
        std::cerr << "Could not create detector class. Specify detector "
                     "mode (static/adaptive) in the settings file" << std::endl;
    }

    return cv::Ptr<cv::FeatureDetector>(); //return the nullptr

}
#endif


/*
generates a feature detector based on options in the settings file
*/
#if CV_MAJOR_VERSION == 2
cv::Ptr<cv::DescriptorExtractor> generateExtractor(cv::FileStorage &fs)
{
    std::string extractorType = fs["FeatureOptions"]["ExtractorType"];
    cv::Ptr<cv::DescriptorExtractor> extractor = NULL;
    if(extractorType == "DUMMY") {

#ifdef USENONFREE
    } else if(extractorType == "SIFT") {
#if CV_MINOR_VERSION == 4
        extractor = new cv::SIFT(
                    fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#elif CV_MINOR_VERSION == 3
        extractor = new cv::SiftDescriptorExtractor();
#endif

    } else if(extractorType == "SURF") {

#if CV_MINOR_VERSION == 4
        extractor = new cv::SURF(
                    fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

#elif CV_MINOR_VERSION == 3
        extractor = new cv::SurfDescriptorExtractor(
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
#endif
#endif

    } else {
        std::cerr << "Could not create Descriptor Extractor. Please specify "
                     "extractor type in settings file" << std::endl;
    }

    return extractor;

}
#elif CV_MAJOR_VERSION == 3

cv::Ptr<cv::DescriptorExtractor> generateExtractor(cv::FileStorage &fs)
{
    std::string extractorType = fs["FeatureOptions"]["ExtractorType"];

    if(extractorType == "BRISK") {
        return cv::BRISK::create(
                    fs["FeatureOptions"]["BRISK"]["Threshold"],
                fs["FeatureOptions"]["BRISK"]["Octaves"],
                fs["FeatureOptions"]["BRISK"]["PatternScale"]);
    } else if(extractorType == "ORB") {
        return cv::ORB::create(
                    fs["FeatureOptions"]["ORB"]["nFeatures"],
                fs["FeatureOptions"]["ORB"]["scaleFactor"],
                fs["FeatureOptions"]["ORB"]["nLevels"],
                fs["FeatureOptions"]["ORB"]["edgeThreshold"],
                fs["FeatureOptions"]["ORB"]["firstLevel"],
                2, cv::ORB::HARRIS_SCORE,
                fs["FeatureOptions"]["ORB"]["patchSize"]);
#ifdef USENONFREE
    } else if(extractorType == "SURF") {
        return cv::xfeatures2d::SURF::create(
                    fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

    } else if(extractorType == "SIFT") {
        return cv::xfeatures2d::SIFT::create(
                    fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#endif

    } else {
        std::cerr << "Could not create Descriptor Extractor. Please specify "
                     "extractor type in settings file" << std::endl;
    }

    return cv::Ptr<cv::DescriptorExtractor>();

}

#endif



/*
create an instance of a FabMap class with the options given in the settings file 使用设置文件中提供的选项创建FabMap类的实例
*/
of2::FabMap *generateFABMAPInstance(cv::FileStorage &settings) 
{

    cv::FileStorage fs;

    //load FabMap training data
    std::string fabmapTrainDataPath = settings["FilePaths"]["TrainImagDesc"];
    std::string chowliutreePath = settings["FilePaths"]["ChowLiuTree"];

    std::cout << "Loading FabMap Training Data" << std::endl;
    fs.open(fabmapTrainDataPath, cv::FileStorage::READ);
    cv::Mat fabmapTrainData;
    fs["BOWImageDescs"] >> fabmapTrainData;
    if (fabmapTrainData.empty()) {
        std::cerr << fabmapTrainDataPath << ": FabMap Training Data not found"
                  << std::endl;
        return NULL;
    }
    fs.release();

    //load a chow-liu tree
    std::cout << "Loading Chow-Liu Tree" << std::endl;
    fs.open(chowliutreePath, cv::FileStorage::READ);
    cv::Mat clTree;
    fs["ChowLiuTree"] >> clTree;
    if (clTree.empty()) {
        std::cerr << chowliutreePath << ": Chow-Liu tree not found" <<
                     std::endl;
        return NULL;
    }
    fs.release();

    //create options flags 创建选项标志,选择SAMPLED/MEAN_FIELD，CHOW_LIU/NAIVE_BAYES等不同近似方式
    std::string newPlaceMethod =
            settings["openFabMapOptions"]["NewPlaceMethod"];
    std::string bayesMethod = settings["openFabMapOptions"]["BayesMethod"];
    int simpleMotionModel = settings["openFabMapOptions"]["SimpleMotion"];
    int options = 0;
    if(newPlaceMethod == "Sampled") {
        options |= of2::FabMap::SAMPLED;  //a|=b  =  a=a|b 按位做或运算
    } else {
        options |= of2::FabMap::MEAN_FIELD;
    }
    if(bayesMethod == "ChowLiu") {
        options |= of2::FabMap::CHOW_LIU;
    } else {
        options |= of2::FabMap::NAIVE_BAYES;
    }
    if(simpleMotionModel) {
        options |= of2::FabMap::MOTION_MODEL;
    }

    of2::FabMap *fabmap;

    //create an instance of the desired type of FabMap 按照设置文件中的选项创建所需类型FabMap的实例
    std::string fabMapVersion = settings["openFabMapOptions"]["FabMapVersion"];
    if(fabMapVersion == "FABMAP1") {
        fabmap = new of2::FabMap1(clTree,
                                  settings["openFabMapOptions"]["PzGe"], //检测器模型的假阴性概率值
                settings["openFabMapOptions"]["PzGne"],  //检测器模型的假阳性概率值
                options,
                settings["openFabMapOptions"]["NumSamples"]);//采样样本数
    } else if(fabMapVersion == "FABMAPLUT") {
        fabmap = new of2::FabMapLUT(clTree,
                                    settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options,
                settings["openFabMapOptions"]["NumSamples"],
                settings["openFabMapOptions"]["FabMapLUT"]["Precision"]);
    } else if(fabMapVersion == "FABMAPFBO") {
        fabmap = new of2::FabMapFBO(clTree,
                                    settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options,
                settings["openFabMapOptions"]["NumSamples"],
                settings["openFabMapOptions"]["FabMapFBO"]["RejectionThreshold"],
                settings["openFabMapOptions"]["FabMapFBO"]["PsGd"],
                settings["openFabMapOptions"]["FabMapFBO"]["BisectionStart"],
                settings["openFabMapOptions"]["FabMapFBO"]["BisectionIts"]);
    } else if(fabMapVersion == "FABMAP2") {
        fabmap = new of2::FabMap2(clTree,
                                  settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options); 
    } else {
        std::cerr << "Could not identify openFABMAPVersion from settings"
                     " file" << std::endl;
        return NULL;
    }

    //add the training data for use with the sampling method  添加用于采样方法的训练数据
    fabmap->addTraining(fabmapTrainData);

    return fabmap;

}



/*
draws keypoints to scale with coloring proportional to feature strength
*/
void drawRichKeypoints(const cv::Mat& src, std::vector<cv::KeyPoint>& kpts, cv::Mat& dst) {

    cv::Mat grayFrame;
    cvtColor(src, grayFrame, CV_RGB2GRAY);
    cvtColor(grayFrame, dst, CV_GRAY2RGB);

    if (kpts.size() == 0) {
        return;
    }

    std::vector<cv::KeyPoint> kpts_cpy, kpts_sorted;

    kpts_cpy.insert(kpts_cpy.end(), kpts.begin(), kpts.end());

    double maxResponse = kpts_cpy.at(0).response;
    double minResponse = kpts_cpy.at(0).response;

    while (kpts_cpy.size() > 0) {

        double maxR = 0.0;
        unsigned int idx = 0;

        for (unsigned int iii = 0; iii < kpts_cpy.size(); iii++) {

            if (kpts_cpy.at(iii).response > maxR) {
                maxR = kpts_cpy.at(iii).response;
                idx = iii;
            }

            if (kpts_cpy.at(iii).response > maxResponse) {
                maxResponse = kpts_cpy.at(iii).response;
            }

            if (kpts_cpy.at(iii).response < minResponse) {
                minResponse = kpts_cpy.at(iii).response;
            }
        }

        kpts_sorted.push_back(kpts_cpy.at(idx));
        kpts_cpy.erase(kpts_cpy.begin() + idx);

    }

    int thickness = 1;
    cv::Point center;
    cv::Scalar colour;
    int red = 0, blue = 0, green = 0;
    int radius;
    double normalizedScore;

    if (minResponse == maxResponse) {
        colour = CV_RGB(255, 0, 0);
    }

    for (int iii = (int)kpts_sorted.size()-1; iii >= 0; iii--) {

        if (minResponse != maxResponse) {
            normalizedScore = pow((kpts_sorted.at(iii).response - minResponse) / (maxResponse - minResponse), 0.25);
            red = int(255.0 * normalizedScore);
            green = int(255.0 - 255.0 * normalizedScore);
            colour = CV_RGB(red, green, blue);
        }

        center = kpts_sorted.at(iii).pt;
        center.x *= 16;
        center.y *= 16;

        radius = (int)(16.0 * ((double)(kpts_sorted.at(iii).size)/2.0));

        if (radius > 0) {
            circle(dst, center, radius, colour, thickness, CV_AA, 4);
        }

    }

}

/*
Removes surplus features and those with invalid size 删除多余的特征和尺寸无效的特征
*/
void filterKeypoints(std::vector<cv::KeyPoint>& kpts, int maxSize, int maxFeatures) {

    if (maxSize == 0) {
        return;
    }

    sortKeypoints(kpts);

    for (unsigned int iii = 0; iii < kpts.size(); iii++) {

        if (kpts.at(iii).size > float(maxSize)) {
            kpts.erase(kpts.begin() + iii);
            iii--;
        }
    }

    if ((maxFeatures != 0) && ((int)kpts.size() > maxFeatures)) {
        kpts.erase(kpts.begin()+maxFeatures, kpts.end());
    }

}

/*
Sorts keypoints in descending order of response (strength)
*/
void sortKeypoints(std::vector<cv::KeyPoint>& keypoints) {

    if (keypoints.size() <= 1) {
        return;
    }

    std::vector<cv::KeyPoint> sortedKeypoints;

    // Add the first one
    sortedKeypoints.push_back(keypoints.at(0));

    for (unsigned int i = 1; i < keypoints.size(); i++) {

        unsigned int j = 0;
        bool hasBeenAdded = false;

        while ((j < sortedKeypoints.size()) && (!hasBeenAdded)) {

            if (abs(keypoints.at(i).response) > abs(sortedKeypoints.at(j).response)) {
                sortedKeypoints.insert(sortedKeypoints.begin() + j, keypoints.at(i));

                hasBeenAdded = true;
            }

            j++;
        }

        if (!hasBeenAdded) {
            sortedKeypoints.push_back(keypoints.at(i));
        }

    }

    keypoints.swap(sortedKeypoints);

}
