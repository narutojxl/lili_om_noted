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
    
    //注意: 窗口内的位姿是imu在map下的位姿, 不是laser在map下的位姿
    //如下4类的ResidualBlockInfo, 被添加到本次marg对象

    //<1>marg位姿的imu预积分残差, P V Q Ba Bg, 15*1 , 产生一个ResidualBlockInfo
    //parameter_blocks: 6个,  tmpTrans[0], tmpQuat[0], tmpSpeedBias[0], tmpTrans[1], tmpQuat[1], tmpSpeedBias[1]
    //每个参数的大小为:              3           4               9            3             4              9
    //drop_set:               vector<int>{0, 1, 2}


    //<2>marg位姿自己帧的laser残差
    //parameter_blocks: 2个,  tmpTrans[0], tmpQuat[0]
    //每个参数的大小为:               3          4
    //drop_set:              vector<int>{0, 1}


    //<3>窗口内其他帧的laser残差
    //第2帧
    //parameter_blocks: 2个,  tmpTrans[1], tmpQuat[1] 
    //drop_set:                    空

    //第3帧
    //parameter_blocks: 2个,  tmpTrans[2], tmpQuat[2] 
    //drop_set:                    空


    //<4>上一帧边缘化产生的残差
    //parameter_blocks: last_marginalization_parameter_blocks
    //drop_set:          ?


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
    std::unordered_map<long, int> parameter_block_idx; //local size,   <每个参数块的地址, 在矩阵中的id>  
    std::unordered_map<long, double *> parameter_block_data;         //<每个参数块的地址, 指向每个参数块的raw指针>


    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians; //边缘化得到的雅可比矩阵
    Eigen::VectorXd linearized_residuals; //边缘化得到的残差
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
    std::unordered_map<long, int> parameter_block_idx; //local size     <每个参数块的地址, 在矩阵中的id> 
};

#endif //MARGINALIZATIONFACTOR_H_
