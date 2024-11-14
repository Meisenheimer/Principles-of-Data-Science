#ifndef MW_CONFIG_H
#define MW_CONFIG_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <string>

#include "Eigen/Eigen"

using Int = long long int;
using Real = long double;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Vector<Real, Eigen::Dynamic>;

inline Matrix loadTimeSeries(const Int &node, const std::string &id);

inline Real covariance(const Vector &ts1, const Vector &ts2);
inline Matrix covariance(const Matrix &ts);

inline Matrix fc_pc(const Matrix &ts, const Real &rho);

inline Matrix loadTimeSeries(const Int &node, const std::string &id)
{
    std::vector<Real> data;
    std::string filename = "../Data/node_timeseries/3T_HCP1200_MSMAll_d" +
                           std::to_string(node) + "_ts2/" + id + ".txt";
    std::ifstream fp(filename, std::ios::in);
    while (!fp.eof())
    {
        Real x;
        fp >> x;
        data.push_back(x);
    }
    fp.close();
    const Int t = data.size() / node;
    Matrix res = Matrix::Zero(node, t);
    for (Int j = 0, k = 0; j < t; j++)
    {
        for (Int i = 0; i < node; i++, k++)
        {
            res(i, j) = data.at(k);
        }
    }
    return res;
}

inline Real covariance(const Vector &ts1, const Vector &ts2)
{
    return (ts1 - ts1.mean() * Vector::Ones(ts1.rows())).dot(ts2 - ts2.mean() * Vector::Ones(ts2.rows())) / (Real)(ts1.rows());
}

inline Matrix covariance(const Matrix &ts)
{
    const Int node = ts.rows();
    Matrix res = Matrix::Zero(node, node);
    for (Int i = 0; i < node; i++)
    {
        for (Int j = i; j < node; j++)
        {
            res(i, j) = res(j, i) = covariance(ts.row(i), ts.row(j));
        }
    }
    return res;
}

inline Matrix fc_pc(const Matrix &ts, const Real &rho)
{
    const Int node = ts.rows();
    Matrix pcm = covariance(ts);
    pcm /= std::sqrt((pcm.diagonal().cwiseAbs2()).mean());
    pcm = (pcm + std::abs(rho) * Matrix::Identity(node, node)).inverse();
    if (rho >= 0)
    {
        pcm = -pcm;
        Vector tmp = pcm.diagonal().cwiseAbs().cwiseSqrt();
        for (Int i = 0; i < node; i++)
        {
            for (Int j = 0; j < node; j++)
            {
                pcm(i, j) /= (tmp(i) * tmp(j));
            }
        }
    }
    const Int n = std::min(pcm.rows(), pcm.cols());
    for (Int i = 0; i < n; i++)
    {
        pcm(i, i) = 0.0;
    }
    return pcm;
}

#endif