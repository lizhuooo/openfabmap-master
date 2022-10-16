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
//    and/or other cv::Materials provided with the distribution.
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

#include "chowliutree.hpp"

#include <iostream>
#include <map>

namespace of2 {

ChowLiuTree::ChowLiuTree() {
}

ChowLiuTree::~ChowLiuTree() {
}
//载入生成周柳树需要的训练数据
void ChowLiuTree::add(const cv::Mat& imgDescriptor) {
    CV_Assert(!imgDescriptor.empty());
    if (!imgDescriptors.empty()) {
        CV_Assert(imgDescriptors[0].cols == imgDescriptor.cols);
        CV_Assert(imgDescriptors[0].type() == imgDescriptor.type());
    }

    imgDescriptors.push_back(imgDescriptor);

}

void ChowLiuTree::add(const std::vector<cv::Mat>& imgDescriptors) {
    for (size_t i = 0; i < imgDescriptors.size(); i++) {
        add(imgDescriptors[i]);
    }
}

const std::vector<cv::Mat>& ChowLiuTree::getImgDescriptors() const {
    return imgDescriptors;
}
//生成周柳树
cv::Mat ChowLiuTree::make(double infoThreshold) 
{
    CV_Assert(!imgDescriptors.empty());

    unsigned int descCount = 0;
    for (size_t i = 0; i < imgDescriptors.size(); i++)
        descCount += imgDescriptors[i].rows;

    mergedImgDescriptors = cv::Mat(descCount, imgDescriptors[0].cols,
        imgDescriptors[0].type());
    for (size_t i = 0, start = 0; i < imgDescriptors.size(); i++)
    {
        cv::Mat submut = mergedImgDescriptors.rowRange((int)start,
            (int)(start + imgDescriptors[i].rows));
        imgDescriptors[i].copyTo(submut);
        start += imgDescriptors[i].rows;
    }

    std::list<info> edges;
    createBaseEdges(edges, infoThreshold); //得到满足互信息阈值要求的二维分布

    // TODO: if it cv_asserts here they really won't know why.

    CV_Assert(reduceEdgesToMinSpan(edges));

    return buildTree(edges.front().word1, edges); //建立周柳树
}

double ChowLiuTree::P(int a, bool za) {  //计算边缘概率p(zi=0/1)

    if(za) {
        return (0.98 * cv::countNonZero(mergedImgDescriptors.col(a)) /
            mergedImgDescriptors.rows) + 0.01;
    } else {
        return 1 - ((0.98 * cv::countNonZero(mergedImgDescriptors.col(a)) /
            mergedImgDescriptors.rows) + 0.01);
    }

}
double ChowLiuTree::JP(int a, bool za, int b, bool zb) {   //计算二维分布的联合概率p(xi=0/1，xj=0/1)

    double count = 0;
    for(int i = 0; i < mergedImgDescriptors.rows; i++) 
    {
        if((mergedImgDescriptors.at<float>(i,a) > 0) == za &&(mergedImgDescriptors.at<float>(i,b) > 0) == zb) 
        {
                count++;
        }
    }
    return count / mergedImgDescriptors.rows;

}
double ChowLiuTree::CP(int a, bool za, int b, bool zb){  //计算二维分布的条件概率p(xi=0/1 | xj=0/1)

    int count = 0, total = 0;
    for(int i = 0; i < mergedImgDescriptors.rows; i++) {
        if((mergedImgDescriptors.at<float>(i,b) > 0) == zb) {
            total++;
            if((mergedImgDescriptors.at<float>(i,a) > 0) == za) {
                count++;
            }
        }
    }
    if(total) {
        return (double)(0.98 * count)/total + 0.01;
    } else {
        return (za) ? 0.01 : 0.99;
    }
}
//建立周柳树
cv::Mat ChowLiuTree::buildTree(int root_word, std::list<info> &edges) 
{

    int q = root_word;
    cv::Mat cltree(4, (int)edges.size()+1, CV_64F);

    cltree.at<double>(0, q) = q;             //第一排存放变量的父级索引
    cltree.at<double>(1, q) = P(q, true);    //第二排存放变量的概率P(q)
    cltree.at<double>(2, q) = P(q, true);    //第三排存放变量与父级的条件概率P(q|p)
    cltree.at<double>(3, q) = P(q, true);    //第四排存放变量与父级的条件概率P(q|～p)
    //setting P(zq|zpq) to P(zq) gives the root node of the chow-liu
    //independence from a parent node.

    //find all children and do the same
    std::vector<int> nextqs = extractChildren(edges, q);//找到当前节点的子节点（变量）

    int pq = q;
    std::vector<int>::iterator nextq;
    for(nextq = nextqs.begin(); nextq != nextqs.end(); nextq++) {
        recAddToTree(cltree, *nextq, pq, edges);//将当前节点（变量）与其相关信息（即父节点、概率等）加入到周柳树结构中
    }

    return cltree;


}
//将当前节点（变量）与其相关信息（即父节点、概率等）加入到周柳树结构中
void ChowLiuTree::recAddToTree(cv::Mat &cltree, int q, int pq,
                               std::list<info>& remaining_edges) {

    cltree.at<double>(0, q) = pq;
    cltree.at<double>(1, q) = P(q, true);
    cltree.at<double>(2, q) = CP(q, true, pq, true);
    cltree.at<double>(3, q) = CP(q, true, pq, false);

    //find all children and do the same
    std::vector<int> nextqs = extractChildren(remaining_edges, q);

    pq = q;
    std::vector<int>::iterator nextq;
    for(nextq = nextqs.begin(); nextq != nextqs.end(); nextq++) {
        recAddToTree(cltree, *nextq, pq, remaining_edges);
    }
}
//找到当前节点的子节点（变量）
std::vector<int> ChowLiuTree::extractChildren(std::list<info> &remaining_edges, int q) {

    std::vector<int> children;
    std::list<info>::iterator edge = remaining_edges.begin();

    while(edge != remaining_edges.end()) {
        if(edge->word1 == q) {
            children.push_back(edge->word2);
            edge = remaining_edges.erase(edge);
            continue;
        }
        if(edge->word2 == q) {
            children.push_back(edge->word1);
            edge = remaining_edges.erase(edge);
            continue;
        }
        edge++;
    }

    return children;
}
//按照互信息由高到低排序
bool ChowLiuTree::sortInfoScores(const info& first, const info& second) {
    return first.score > second.score;
}

//计算互信息
double ChowLiuTree::calcMutInfo(int word1, int word2) {   
    double accumulation = 0;

    double P00 = JP(word1, false, word2, false);
    if(P00) accumulation += P00 * log(P00 / (P(word1, false)*P(word2, false)));

    double P01 = JP(word1, false, word2, true);
    if(P01) accumulation += P01 * log(P01 / (P(word1, false)*P(word2, true)));

    double P10 = JP(word1, true, word2, false);
    if(P10) accumulation += P10 * log(P10 / (P(word1, true)*P(word2, false)));

    double P11 = JP(word1, true, word2, true);
    if(P11) accumulation += P11 * log(P11 / (P(word1, true)*P(word2, true)));

    return accumulation;
}
//得到满足互信息阈值要求的二维分布
void ChowLiuTree::createBaseEdges(std::list<info>& edges, double infoThreshold) 
{

    int nWords = imgDescriptors[0].cols;

#pragma omp parallel for schedule(dynamic, 500)  //并行运算
    for(int word1 = 0; word1 < nWords; word1++) {
        std::list<info> threadEdges;
        info mutInfo;
        for(int word2 = word1 + 1; word2 < nWords; word2++)
         {
            mutInfo.word1 = word1;
            mutInfo.word2 = word2;
            mutInfo.score = (float)calcMutInfo(word1, word2); //计算互信息
            if(mutInfo.score >= infoThreshold)  //只取满足阈值要求的二维分布
            threadEdges.push_back(mutInfo);
         }
#pragma omp critical
        {
            edges.splice(edges.end(), threadEdges);//将threadEdges剪接到edges.end()后
        }

        // Status
        if (nWords >= 10 && (word1+1)%(nWords/10) == 0)
            std::cout << "." << std::flush;
    }
    edges.sort(sortInfoScores);//按照互信息从大到小排序
}

//在众多二维分布中找出n-1个最佳选择
bool ChowLiuTree::reduceEdgesToMinSpan(std::list<info>& edges) {

    std::map<int, int> groups; std::map<int, int>::iterator groupIt;
    for(int i = 0; i < imgDescriptors[0].cols; i++) groups[i] = i;
    int group1, group2;

    std::list<info>::iterator edge = edges.begin();
    while(edge != edges.end())  //去除造成循环的分支
    {
        if(groups[edge->word1] != groups[edge->word2]) 
        {
            group1 = groups[edge->word1];
            group2 = groups[edge->word2];
            for(groupIt = groups.begin(); groupIt != groups.end(); groupIt++)
            if(groupIt->second == group2) groupIt->second = group1;
            edge++;
        }
        else 
        {
            edge = edges.erase(edge);
        }
    }

    if(edges.size() != (unsigned int)imgDescriptors[0].cols - 1) {
        return false;
    } else {
        return true;
    }

}

}

