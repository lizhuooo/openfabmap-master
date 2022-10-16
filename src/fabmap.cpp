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
//    and/or other Materials provided with the distribution.
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
#include<iostream>
#include "fabmap.hpp"
#include <time.h>
using namespace std;
/*
    Calculate the sum of two log likelihoods
*/
namespace of2 {

static double logsumexp(double a, double b) {   //计算两个似然的和的对数
    return a > b ? log(1 + exp(b - a)) + a : log(1 + exp(a - b)) + b;
}

FabMap::FabMap(const cv::Mat & _clTree, double _PzGe,
               double _PzGNe, int _flags, int _numSamples) :
    flags(_flags), numSamples(_numSamples),
    clTree(new cv::Mat(_clTree)),
    infer(new InferBinary(clTree, _PzGe, _PzGNe, (_flags & NAIVE_BAYES) != 0))
{
    CV_Assert(flags & MEAN_FIELD || flags & SAMPLED);
    CV_Assert(flags & NAIVE_BAYES || flags & CHOW_LIU);

    // TODO: Add default values for member variables
    Pnew = 0.9; //新位置的先验
    sFactor = 0.99; //平滑化
    mBias = 0.5;  //第一版论文未用此参数
}

FabMap::~FabMap() {
}

const std::vector<cv::Mat>& FabMap::getTrainingImgDescriptors() const {
    return trainingImgDescriptors;
}

const std::vector<cv::Mat>& FabMap::getTestImgDescriptors() const {
    return testImgDescriptors;
}

void FabMap::addTraining(const cv::Mat& queryImgDescriptor) {
    CV_Assert(!queryImgDescriptor.empty());
    std::vector<cv::Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }
    addTraining(queryImgDescriptors);
}

void FabMap::addTraining(const std::vector<cv::Mat>& queryImgDescriptors) {
    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == infer->vocabSize());
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);
        trainingImgDescriptors.push_back(queryImgDescriptors[i]);
    }
}

void FabMap::add(const cv::Mat& queryImgDescriptor) {
    CV_Assert(!queryImgDescriptor.empty());
    std::vector<cv::Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }
    add(queryImgDescriptors);
}

void FabMap::add(const std::vector<cv::Mat>& queryImgDescriptors) {
    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == infer->vocabSize());
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);
        testImgDescriptors.push_back(queryImgDescriptors[i]);
    }
}

void FabMap::compare(const cv::Mat& queryImgDescriptor,
                     const cv::Mat& testImgDescriptor, std::vector<IMatch>& matches,
                     const cv::Mat& mask) {
    CV_Assert(!queryImgDescriptor.empty());
    std::vector<cv::Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }

    CV_Assert(!testImgDescriptor.empty());
    std::vector<cv::Mat> _testImgDescriptors;
    for (int i = 0; i < testImgDescriptor.rows; i++) {
        _testImgDescriptors.push_back(testImgDescriptor.row(i));
    }
    compare(queryImgDescriptors,_testImgDescriptors,matches,mask);

}

void FabMap::compare(const cv::Mat& queryImgDescriptor,
                     const std::vector<cv::Mat>& _testImgDescriptors,
                     std::vector<IMatch>& matches, const cv::Mat& mask) {
    CV_Assert(!queryImgDescriptor.empty());
    std::vector<cv::Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }
    compare(queryImgDescriptors,_testImgDescriptors,matches,mask);
}

void FabMap::compare(const std::vector<cv::Mat>& queryImgDescriptors,
                     const std::vector<cv::Mat>& _testImgDescriptors,
                     std::vector<IMatch>& matches, const cv::Mat& /*mask*/) {

    CV_Assert(!(flags & MOTION_MODEL));
    for (size_t i = 0; i < _testImgDescriptors.size(); i++) {
        CV_Assert(!_testImgDescriptors[i].empty());
        CV_Assert(_testImgDescriptors[i].rows == 1);
        CV_Assert(_testImgDescriptors[i].cols == infer->vocabSize());
        CV_Assert(_testImgDescriptors[i].type() == CV_32F);
    }

    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == infer->vocabSize());
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);

        // TODO: add mask

        compareImgDescriptor(queryImgDescriptors[i],
                             (int)i, _testImgDescriptors, matches);
    }
}

// DEPRECATED, USE LOCALIZE BELOW
void FabMap::compare(const cv::Mat& queryImgDescriptor,
                     std::vector<IMatch>& matches, bool addQuery,
                     const cv::Mat& mask) {
    return localize(queryImgDescriptor,matches,addQuery,mask);
}
//进行定位，queryImgDescriptor 所有测试图片的BOW特征描述(在一个矩阵里)分开放进容器里
void FabMap::localize(const cv::Mat& queryImgDescriptor,
                      std::vector<IMatch>& matches, bool addQuery,
                      const cv::Mat& mask)
{
    CV_Assert(!queryImgDescriptor.empty());
    std::vector<cv::Mat> queryImgDescriptors;
    for (int i = 0; i < queryImgDescriptor.rows; i++) {
        queryImgDescriptors.push_back(queryImgDescriptor.row(i));
    }
    //compare(queryImgDescriptors,matches,addQuery,mask);
    localize(queryImgDescriptors,matches,addQuery,mask);
}

// DEPRECATED, USE LOCALIZE BELOW
void FabMap::compare(const std::vector<cv::Mat>& queryImgDescriptors,
                     std::vector<IMatch>& matches, bool addQuery, const cv::Mat& mask) {
    return localize(queryImgDescriptors,matches,addQuery,mask);
}
//进行定位，为每一个观测（这里即描述子）计算位置估计
void FabMap::localize(const std::vector<cv::Mat>& queryImgDescriptors,
                      std::vector<IMatch>& matches, bool addQuery, const cv::Mat& /*mask*/) {

    // TODO: add first query if empty (is this necessary)

    for (size_t i = 0; i < queryImgDescriptors.size(); i++) 
    {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == infer->vocabSize());
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);

        // TODO: add mask
//计算每次判断回环所需时间        
 clock_t start,finish;
 double totaltime;
 start = clock();

        compareImgDescriptor(queryImgDescriptors[i],
                             (int)i, testImgDescriptors, matches);

 finish = clock();
 totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
 cout << "\n查询图片 "<< i+1 << "的运行时间为" << totaltime*1000 << "毫秒!" << endl;

        if (addQuery)
            add(queryImgDescriptors[i]); //添加历史帧，新帧将与历史帧做匹配（位置估计）
    }
}
//计算当前观测的位置估计，输入参数：queryImgDescriptor 查询图片的描述子， queryIndex 查询图片序号，_testImgDescriptors 测试图片描述子(历史帧)
void FabMap::compareImgDescriptor(const cv::Mat& queryImgDescriptor,
                                  int queryIndex, const std::vector<cv::Mat>& _testImgDescriptors,
                                  std::vector<IMatch>& matches) 
{

    std::vector<IMatch> querymatches;
    querymatches.push_back(IMatch(queryIndex,-1,
                                  getNewPlaceLikelihood(queryImgDescriptor),0)); //计算未知位置的似然并压入容器
    getLikelihoods(queryImgDescriptor,_testImgDescriptors,querymatches); //计算地图中已知位置的似然
    normaliseDistribution(querymatches);  //计算归一化常量
    for (size_t j = 1; j < querymatches.size(); j++) 
    {
        querymatches[j].queryIdx = queryIndex;
    }
    matches.insert(matches.end(), querymatches.begin(), querymatches.end());
}
//计算新位置的似然
double FabMap::getNewPlaceLikelihood(const cv::Mat& queryImgDescriptor) {  
    if (flags & MEAN_FIELD) //使用平均场方法计算新位置的似然
    {
        double logP = 0.;
#pragma omp parallel for reduction(+:logP)
        for (int q = 0; q < infer->vocabSize(); q++)
        {
            bool zq = queryImgDescriptor.at<float>(0,q) > 0;
            bool zpq = queryImgDescriptor.at<float>(0,infer->pq(q)) > 0;
            logP += log(infer->PzGL(q, zq, zpq, false/*unused*/, true));
        }
        return logP;
    }

    if (flags & SAMPLED) {                                //使用采样(蒙特卡洛)方法计算新位置的似然
        CV_Assert(!trainingImgDescriptors.empty());
        CV_Assert(numSamples > 0);

        std::vector<cv::Mat> sampledImgDescriptors;

        // TODO: this method can result in the same sample being added
        // multiple times. Is this desired?

        for (int i = 0; i < numSamples; i++) {
            int index = rand() % trainingImgDescriptors.size();
            sampledImgDescriptors.push_back(trainingImgDescriptors[index]);
        } //训练样本图片描述子随机重排,(模拟采样？)

        std::vector<IMatch> matches;
        getLikelihoods(queryImgDescriptor,sampledImgDescriptors,matches);

        double averageLogLikelihood = -DBL_MAX + matches.front().likelihood + 1;  //这里没明白为何要加－DBL_MAX，但是数学上不加对下面求和也没影响
        
//cout << averageLogLikelihood << endl;  //测试代码

        for (int i = 0; i < numSamples; i++) //计算所有采样位置的似然相加
        {
            averageLogLikelihood =
                    logsumexp(matches[i].likelihood, averageLogLikelihood);
        }

        return averageLogLikelihood - log((double)numSamples); //新位置的似然
    }
    return 0;
}
//计算归一化常量
void FabMap::normaliseDistribution(std::vector<IMatch>& matches) {
    CV_Assert(!matches.empty());

    if (flags & MOTION_MODEL) //设定简单运动模型,看的论文中没有这一点
    {

        matches[0].match = matches[0].likelihood + log(Pnew);

        if (priormatches.size() > 2) {
            matches[1].match = matches[1].likelihood;
            matches[1].match += log(
                        (2 * (1-mBias) * priormatches[1].match +
                        priormatches[1].match +
                    2 * mBias * priormatches[2].match) / 3);
            for (size_t i = 2; i < priormatches.size()-1; i++) {
                matches[i].match = matches[i].likelihood;
                matches[i].match += log(
                            (2 * (1-mBias) * priormatches[i-1].match +
                            priormatches[i].match +
                            2 * mBias * priormatches[i+1].match)/3);
            }
            matches[priormatches.size()-1].match =
                    matches[priormatches.size()-1].likelihood;
            matches[priormatches.size()-1].match += log(
                        (2 * (1-mBias) * priormatches[priormatches.size()-2].match +
                        priormatches[priormatches.size()-1].match +
                    2 * mBias * priormatches[priormatches.size()-1].match)/3);

            for(size_t i = priormatches.size(); i < matches.size(); i++) {
                matches[i].match = matches[i].likelihood;
            }
        } else {
            for(size_t i = 1; i < matches.size(); i++) {
                matches[i].match = matches[i].likelihood;
            }
        }

        double logsum = -DBL_MAX + matches.front().match + 1;

        //calculate the normalising constant
        for (size_t i = 0; i < matches.size(); i++) {
            logsum = logsumexp(logsum, matches[i].match); 
        }

        //normalise
        for (size_t i = 0; i < matches.size(); i++) {
            matches[i].match = exp(matches[i].match - logsum);
        }

        //smooth final probabilities
        for (size_t i = 0; i < matches.size(); i++) {
            matches[i].match = sFactor*matches[i].match +
                    (1 - sFactor)/matches.size();
        }

        //update our location priors
        priormatches = matches;

    } 
 else //无运动模型,论文第一版中采用的这里
    {

        double logsum = -DBL_MAX + matches.front().likelihood + 1;  //

        for (size_t i = 0; i < matches.size(); i++) //计算归一化常量
        {
            logsum = logsumexp(logsum, matches[i].likelihood); //计算两个似然的和的对数
        }
        for (size_t i = 0; i < matches.size(); i++) //对每个历史帧计算后验概率
        {
            matches[i].match = exp(matches[i].likelihood - logsum);
        }
        for (size_t i = 0; i < matches.size(); i++) //平滑化
        {
            matches[i].match = sFactor*matches[i].match +
                    (1 - sFactor)/matches.size();
        }
    }
}

FabMap1::FabMap1(const cv::Mat& _clTree, double _PzGe, double _PzGNe, int _flags,
                 int _numSamples) : FabMap(_clTree, _PzGe, _PzGNe, _flags,
                                           _numSamples) {
}

FabMap1::~FabMap1() {
}
//计算地图中已知位置的似然,注意论文第一版用的都是FabMap1
void FabMap1::getLikelihoods(const cv::Mat& queryImgDescriptor,
                             const std::vector<cv::Mat>& testImgDescriptors, std::vector<IMatch>& matches)
{
    // Preallocate matches
    size_t startOfNewMatches = matches.size();
    matches.resize(startOfNewMatches+testImgDescriptors.size()); //扩容

#pragma omp parallel for if (testImgDescriptors.size() > 100)
    for (int i = 0; i < (int)testImgDescriptors.size(); i++) //遍历地图中所有位置
    {
        bool zq, zpq, Lzq;
        double logP = 0;
        for (int q = 0; q < infer->vocabSize(); q++) //遍历所有单词，求得一个位置的似然
        {
            zq = queryImgDescriptor.at<float>(0,q) > 0; 
            zpq = queryImgDescriptor.at<float>(0,infer->pq(q)) > 0;
            Lzq = testImgDescriptors[i].at<float>(0,q) > 0;
            logP += log(infer->PzGL(q, zq, zpq, Lzq, false));
        }
        matches[startOfNewMatches+(size_t)i] = IMatch(0,i,logP,0);
    }
}

FabMapLUT::FabMapLUT(const cv::Mat& _clTree, double _PzGe, double _PzGNe,
                     int _flags, int _numSamples, int _precision) :
    FabMap(_clTree, _PzGe, _PzGNe, _flags, _numSamples), precision(_precision) {

    int nWords = infer->vocabSize();
    double precFactor = (double)pow(10.0, precision);

    table = new int[nWords][8];

    for (int q = 0; q < nWords; q++) {
        for (unsigned char i = 0; i < 8; i++) {

            bool Lzq = (bool) ((i >> 2) & 0x01);
            bool zq = (bool) ((i >> 1) & 0x01);
            bool zpq = (bool) (i & 1);

            table[q][i] = -(int)(log(infer->PzGL(q, zq, zpq, Lzq, false))
                                 * precFactor);
        }
    }
}

FabMapLUT::~FabMapLUT() {
    delete[] table;
}

void FabMapLUT::getLikelihoods(const cv::Mat& queryImgDescriptor,
                               const std::vector<cv::Mat>& testImageDescriptors, std::vector<IMatch>& matches) {

    double precFactor = (double)pow(10.0, -precision);

    for (size_t i = 0; i < testImageDescriptors.size(); i++) {
        unsigned long long int logP = 0;
        for (int q = 0; q < infer->vocabSize(); q++) {
            logP += table[q][(queryImgDescriptor.at<float>(0,infer->pq(q)) > 0) +
                    ((queryImgDescriptor.at<float>(0, q) > 0) << 1) +
                    ((testImageDescriptors[i].at<float>(0,q) > 0) << 2)];
        }
        matches.push_back(IMatch(0,(int)i,-precFactor*(double)logP,0));
    }
}

FabMapFBO::FabMapFBO(const cv::Mat& _clTree, double _PzGe, double _PzGNe,
                     int _flags, int _numSamples, double _rejectionThreshold,
                     double _PsGd, int _bisectionStart, int _bisectionIts) :
    FabMap(_clTree, _PzGe, _PzGNe, _flags, _numSamples), PsGd(_PsGd),
    rejectionThreshold(_rejectionThreshold), bisectionStart(_bisectionStart),
    bisectionIts(_bisectionIts) {
}


FabMapFBO::~FabMapFBO() {
}

void FabMapFBO::getLikelihoods(const cv::Mat& queryImgDescriptor,
                               const std::vector<cv::Mat>& testImageDescriptors, std::vector<IMatch>& matches) {

    std::multiset<WordStats> wordData;
    setWordStatistics(queryImgDescriptor, wordData);

    std::vector<int> matchIndices;
    std::vector<IMatch> querymatches;

    for (size_t i = 0; i < testImageDescriptors.size(); i++) {
        querymatches.push_back(IMatch(0,(int)i,0,0));
        matchIndices.push_back((int)i);
    }

    double currBest  = -DBL_MAX;
    double bailedOut = DBL_MAX;

    for (std::multiset<WordStats>::iterator wordIter = wordData.begin();
         wordIter != wordData.end(); wordIter++) {
        bool zq = queryImgDescriptor.at<float>(0,wordIter->q) > 0;
        bool zpq = queryImgDescriptor.at<float>(0,infer->pq(wordIter->q)) > 0;

        currBest = -DBL_MAX;

        for (size_t i = 0; i < matchIndices.size(); i++) {
            bool Lzq =
                    testImageDescriptors[matchIndices[i]].at<float>(0,wordIter->q) > 0;
            querymatches[matchIndices[i]].likelihood +=
                    log(infer->PzGL(wordIter->q,zq,zpq,Lzq,false));
            currBest =
                    std::max(querymatches[matchIndices[i]].likelihood, currBest);
        }

        if (matchIndices.size() == 1)
            continue;

        double delta = std::max(limitbisection(wordIter->V, wordIter->M),
                                -log(rejectionThreshold));

        std::vector<int>::iterator matchIter = matchIndices.begin();
        while (matchIter != matchIndices.end()) {
            if (currBest - querymatches[*matchIter].likelihood > delta) {
                querymatches[*matchIter].likelihood = bailedOut;
                matchIter = matchIndices.erase(matchIter);
            } else {
                matchIter++;
            }
        }
    }

    for (size_t i = 0; i < querymatches.size(); i++) {
        if (querymatches[i].likelihood == bailedOut) {
            querymatches[i].likelihood = currBest + log(rejectionThreshold);
        }
    }
    matches.insert(matches.end(), querymatches.begin(), querymatches.end());

}

void FabMapFBO::setWordStatistics(const cv::Mat& queryImgDescriptor,
                                  std::multiset<WordStats>& wordData) {
    //words are sorted according to information = -ln(P(zq|zpq))
    //in non-log format this is lowest probability first
    for (int q = 0; q < infer->vocabSize(); q++) {
        wordData.insert(WordStats(q,
                                  infer->PzqGzpq(q, queryImgDescriptor.at<float>(0,q) > 0,
                                                 queryImgDescriptor.at<float>(0,infer->pq(q)) > 0)));
    }

    double d = 0, V = 0, M = 0;
    bool zq, zpq;

    for (std::multiset<WordStats>::reverse_iterator wordIter =
         wordData.rbegin();
         wordIter != wordData.rend(); wordIter++) {

        zq = queryImgDescriptor.at<float>(0,wordIter->q) > 0;
        zpq = queryImgDescriptor.at<float>(0,infer->pq(wordIter->q)) > 0;

        d = log(infer->PzGL(wordIter->q, zq, zpq, true, false)) -
                log(infer->PzGL(wordIter->q, zq, zpq, false, false));

        V += pow(d, 2.0) * 2 *
                (infer->Pzq(wordIter->q, true) - pow(infer->Pzq(wordIter->q, true), 2.0));
        M = std::max(M, fabs(d));

        wordIter->V = V;
        wordIter->M = M;
    }
}

double FabMapFBO::limitbisection(double v, double m) {
    double midpoint, left_val, mid_val;
    double left = 0, right = bisectionStart;

    left_val = bennettInequality(v, m, left) - PsGd;

    for(int i = 0; i < bisectionIts; i++) {

        midpoint = (left + right)*0.5;
        mid_val = bennettInequality(v, m, midpoint)- PsGd;

        if(left_val * mid_val > 0) {
            left = midpoint;
            left_val = mid_val;
        } else {
            right = midpoint;
        }
    }

    return (right + left) * 0.5;
}

double FabMapFBO::bennettInequality(double v, double m, double delta) {
    double DMonV = delta * m / v;
    double f_delta = log(DMonV + sqrt(pow(DMonV, 2.0) + 1));
    return exp((v / pow(m, 2.0))*(cosh(f_delta) - 1 - DMonV * f_delta));
}

bool FabMapFBO::compInfo(const WordStats& first, const WordStats& second) {
    return first.info < second.info;
}

FabMap2::FabMap2(const cv::Mat& _clTree, double _PzGe, double _PzGNe,
                 int _flags) :
    FabMap(_clTree, _PzGe, _PzGNe, _flags) {
    CV_Assert(flags & SAMPLED);

    children.resize(infer->vocabSize()); 

    for (int q = 0; q < infer->vocabSize(); q++) {
        d1.push_back(log(infer->PzGL(q, false, false, true, false) /
                         infer->PzGL(q, false, false, false, false)));
        d2.push_back(log(infer->PzGL(q, false, true, true, false) /
                         infer->PzGL(q, false, true, false, false)) - d1[q]);
        d3.push_back(log(infer->PzGL(q, true, false, true, false) /
                         infer->PzGL(q, true, false, false, false))- d1[q]);
        d4.push_back(log(infer->PzGL(q, true, true, true, false) /
                         infer->PzGL(q, true, true, false, false))- d1[q]);
        children[infer->pq(q)].push_back(q);
    }

}

FabMap2::~FabMap2() {
}


void FabMap2::addTraining(const std::vector<cv::Mat>& queryImgDescriptors) {
    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == infer->vocabSize());
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);
        trainingImgDescriptors.push_back(queryImgDescriptors[i]);
        addToIndex(queryImgDescriptors[i], trainingDefaults, trainingInvertedMap);
    }
}


void FabMap2::add(const std::vector<cv::Mat>& queryImgDescriptors) {
    for (size_t i = 0; i < queryImgDescriptors.size(); i++) {
        CV_Assert(!queryImgDescriptors[i].empty());
        CV_Assert(queryImgDescriptors[i].rows == 1);
        CV_Assert(queryImgDescriptors[i].cols == infer->vocabSize());
        CV_Assert(queryImgDescriptors[i].type() == CV_32F);
        testImgDescriptors.push_back(queryImgDescriptors[i]);
        addToIndex(queryImgDescriptors[i], testDefaults, testInvertedMap);
    }
}

void FabMap2::getLikelihoods(const cv::Mat& queryImgDescriptor,
                             const std::vector<cv::Mat>& testImageDescriptors, std::vector<IMatch>& matches) {

    if (&testImageDescriptors == &testImgDescriptors) {
        getIndexLikelihoods(queryImgDescriptor, testDefaults, testInvertedMap,
                            matches);
    } else {
        CV_Assert(!(flags & MOTION_MODEL));
        std::vector<double> defaults;
        std::map<int, std::vector<int> > invertedMap;
        for (size_t i = 0; i < testImageDescriptors.size(); i++) {
            addToIndex(testImageDescriptors[i],defaults,invertedMap);
        }
        getIndexLikelihoods(queryImgDescriptor, defaults, invertedMap, matches);
    }
}

double FabMap2::getNewPlaceLikelihood(const cv::Mat& queryImgDescriptor) {

    CV_Assert(!trainingImgDescriptors.empty());

    std::vector<IMatch> matches;
    getIndexLikelihoods(queryImgDescriptor, trainingDefaults,
                        trainingInvertedMap, matches);

    double averageLogLikelihood = -DBL_MAX + matches.front().likelihood + 1;
    for (size_t i = 0; i < matches.size(); i++) {
        averageLogLikelihood =
                logsumexp(matches[i].likelihood, averageLogLikelihood);
    }

    return averageLogLikelihood - log((double)trainingDefaults.size());

}

void FabMap2::addToIndex(const cv::Mat& queryImgDescriptor,
                         std::vector<double>& defaults,
                         std::map<int, std::vector<int> >& invertedMap) {
    defaults.push_back(0);
    for (int q = 0; q < infer->vocabSize(); q++) {
        if (queryImgDescriptor.at<float>(0,q) > 0) {
            defaults.back() += d1[q];
            invertedMap[q].push_back((int)defaults.size()-1);
        }
    }
}

void FabMap2::getIndexLikelihoods(const cv::Mat& queryImgDescriptor,
                                  std::vector<double>& defaults,
                                  std::map<int, std::vector<int> >& invertedMap,
                                  std::vector<IMatch>& matches) {

    std::vector<int>::iterator LwithI, child;

    std::vector<double> likelihoods = defaults;

    for (int q = 0; q < infer->vocabSize(); q++) {
        if (queryImgDescriptor.at<float>(0,q) > 0) {
            for (LwithI = invertedMap[q].begin();
                 LwithI != invertedMap[q].end(); LwithI++) {

                if (queryImgDescriptor.at<float>(0,infer->pq(q)) > 0) {
                    likelihoods[*LwithI] += d4[q];
                } else {
                    likelihoods[*LwithI] += d3[q];
                }
            }
            for (child = children[q].begin(); child != children[q].end();
                 child++) {

                if (queryImgDescriptor.at<float>(0,*child) == 0) {
                    for (LwithI = invertedMap[*child].begin();
                         LwithI != invertedMap[*child].end(); LwithI++) {

                        likelihoods[*LwithI] += d2[*child];
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < likelihoods.size(); i++) {
        matches.push_back(IMatch(0,(int)i,likelihoods[i],0));
    }
}

} // namespace of2
