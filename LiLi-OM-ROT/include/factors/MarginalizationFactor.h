#ifndef MARGINALIZATIONFACTOR_H_
#define MARGINALIZATIONFACTOR_H_

#include <ceres/ceres.h>
#include <Eigen/Eigen>
#include <unordered_map>

#include "utils/common.h"
#include "utils/math_tools.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo {
    ResidualBlockInfo(ceres::CostFunction *_cost_function,
                      ceres::LossFunction *_loss_function,
                      std::vector<double *> _parameter_blocks, 
                      std::vector<int> _drop_set) 
        : cost_function(_cost_function),
          loss_function(_loss_function),
          parameter_blocks(_parameter_blocks),
          drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;
    
    //注意: 窗口内的位姿是imu在map下的位姿,不是laser在map下的位姿

    //marg位姿的imu预积分残差, P V Q Ba Bg, 15*1 , 产生一个ResidualBlockInfo对象
    //parameter_blocks: 6个,  tmpTrans[0], tmpQuat[0], tmpSpeedBias[0], tmpTrans[1], tmpQuat[1], tmpSpeedBias[1]
    //drop_set:               vector<int>{0, 1, 2}


    //marg位姿自己帧的laser残差, 每个有效point产生一个ResidualBlockInfo对象
    //parameter_blocks: 2个,  tmpTrans[0], tmpQuat[0]
    //drop_set:              vector<int>{0, 1}


    //窗口内其他帧的laser残差,  0 < i <= 滑窗size - 1, 每个有效point产生一个ResidualBlockInfo对象
    //parameter_blocks: 2个,  tmpTrans[i], tmpQuat[i]
    //drop_set:         空





    double **raw_jacobians; //大小：由该残差块有多少个参数块决定；每个存放的是：残差对每个参数块的雅克比
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians; //vector形式存放
    Eigen::VectorXd residuals;
};


//＝＝＝＝＝＝＝＝＝＝＝======================||
//                                        ||
//          和lio-mapping一样              ||
//                                        ||
//＝＝＝＝＝＝＝＝＝＝＝======================||


class MarginalizationInfo {
public:
    ~MarginalizationInfo();
    int LocalSize(int size) const;
    void AddResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void PreMarginalize();
    void Marginalize();
    std::vector<double *> GetParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; //当前marg位姿相关联的residuals

    int m, n;
    std::unordered_map<long, int> parameter_block_size; //global size,  <每个参数块的地址, 每个参数块的大小>
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size,   <每个参数块的地址, 地址偏移量> 
    std::unordered_map<long, double *> parameter_block_data;//         <每个参数块的地址, 指向每个参数块的raw指针>


    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};




class MarginalizationFactor : public ceres::CostFunction {
public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};


struct ThreadsStruct {
    std::vector<ResidualBlockInfo *> sub_factors; //每个ThreadsStruct对象, 只计算marg对象的所有ResidualBlockInfo 1/4个
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size   <每个参数块的地址, 每个参数块的大小>
    std::unordered_map<long, int> parameter_block_idx; //local size     <每个参数块的地址, 地址偏移量> 
};

#endif //MARGINALIZATIONFACTOR_H_
