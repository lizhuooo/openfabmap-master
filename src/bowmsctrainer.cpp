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

#include "bowmsctrainer.hpp"

#include <iostream>
#include <vector>
#include <list>

namespace of2 {

BOWMSCTrainer::BOWMSCTrainer(double _clusterSize) :
    clusterSize(_clusterSize) {
}

BOWMSCTrainer::~BOWMSCTrainer() {
}

cv::Mat BOWMSCTrainer::cluster() const {
    CV_Assert(!descriptors.empty());
    int descCount = 0;
    for(size_t i = 0; i < descriptors.size(); i++)
        descCount += descriptors[i].rows;

    cv::Mat mergedDescriptors(descCount, descriptors[0].cols,
            descriptors[0].type());
    for(size_t i = 0, start = 0; i < descriptors.size(); i++)
    {
        cv::Mat submut = mergedDescriptors.rowRange((int)start,
                                                    (int)(start + descriptors[i].rows));
        descriptors[i].copyTo(submut);
        start += descriptors[i].rows;
    }
    return cluster(mergedDescriptors);
}
//使用改进的序列聚类来训练词典
cv::Mat BOWMSCTrainer::cluster(const cv::Mat& descriptors) const {

    CV_Assert(!descriptors.empty());

    // TODO: sort the descriptors before clustering.

    // Start timing
    int64 start_time = cv::getTickCount();

    // Used for Mahalanobis distance calculation, identity covariance
    cv::Mat icovar = cv::Mat::eye(descriptors.cols,descriptors.cols,descriptors.type());

    // Create initial centres guaranteeing a centre distance < minDist //

    // Loop through all the descriptors
    std::vector<cv::Mat> initialCentres; //此容器中放入代表簇中心的描述子
    initialCentres.push_back(descriptors.row(0)); //将第一个描述子作为簇中心压入initialCentres

    for (int i = 1; i < descriptors.rows; i++)
    {
        double minDist = DBL_MAX;
#pragma omp parallel for if (initialCentres.size() > 100)
        for (int j = 0; j < (int)initialCentres.size(); j++)
        {
            // Our covariance is identity, just use the norm, it's faster.
            // cv::Mahalanobis(descriptors.row(i),initialCentres[j], icovar);
            double myDist = cv::norm(descriptors.row(i),initialCentres[j]);  //求两向量的欧氏距离（对应L2范数,差的平方和再开方）
#pragma omp critical
            minDist = std::min(minDist, myDist);
        }
        // Add new cluster if outside of range
        if (minDist > clusterSize)
            initialCentres.push_back(descriptors.row(i));  //若最小距离大于阈值，则当前描述子作为簇中心压入initialCentres

        // Status
        if ((i-1)%(descriptors.rows/10) == 0)
            std::cout << "." << std::flush;
    }
    // Status
    std::cout << "\nFinished initial clustering for "
              << descriptors.rows << " descriptors. "
              << initialCentres.size() << " initial clusters. "
              << std::endl;

    // Assign each descriptor to its closest centre //

    // Loop through all the descriptors again
    // TODO: Consider a kd-tree for this search
    std::vector<std::list<cv::Mat> > clusters; //簇容器，将所有描述子分类为各个簇
    clusters.resize(initialCentres.size()); 
#pragma omp parallel for schedule(dynamic, 200)
    for (int i = 0; i < descriptors.rows; i++) 
    {
        size_t index; double dist, minDist = DBL_MAX;
        for (size_t j = 0; j < initialCentres.size(); j++) 
        {
            dist = cv::norm(descriptors.row(i),initialCentres[j]);
            if (dist < minDist) //找到当前描述子离哪一个簇中心最近
            {
                minDist = dist;
                index = j;
            }
        }
#pragma omp critical // Order doesn't matter here
        clusters[index].push_back(descriptors.row(i)); //将当前描述子压入属于它的簇中

        // Status (could be off because of parallelism, but a guess
        if ((i-1)%(descriptors.rows/10) == 0)
            std::cout << "." << std::flush;
    }
    // Status
    std::cout << "\nFinished re-assignment. "
              << std::endl;

    // Calculate the centre mean for each cluster //

    // Loop through all the clusters
    cv::Mat vocabulary;
#pragma omp parallel for schedule(static, 1) ordered
    for (int i = 0; i < (int)clusters.size(); i++) 
    {
        // TODO: Throw away small clusters
        // TODO: Make this configurable
        // TODO: Re-assign?
        // if (clusters[i].size() < 3) continue;

        cv::Mat centre = cv::Mat::zeros(1,descriptors.cols,descriptors.type());
        for (std::list<cv::Mat>::iterator Ci = clusters[i].begin(); Ci != clusters[i].end(); Ci++) 
        {
            centre += *Ci; //将本簇里的描述子相加
        }
        centre /= (double)clusters[i].size(); //平均相加后的描述子，得到单词
#pragma omp ordered // Ordered so it's identical to non omp.
            vocabulary.push_back(centre); //将当前得到的单词压入词典中

        // Status (could be off because of parallelism, but a guess
        if ((i-1)%(clusters.size()/10) == 0)
            std::cout << "." << std::flush;
    }

    // Finish timing
    int64 end_time = cv::getTickCount();

    // Status
    std::cout << "\nFinished finding the mean. "
              << vocabulary.rows << " words. "
              << (end_time-start_time)/cv::getTickFrequency() << " s. "
              << std::endl;

    return vocabulary;
}

}

