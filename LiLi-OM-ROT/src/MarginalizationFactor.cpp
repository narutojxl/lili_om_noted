#include "factors/MarginalizationFactor.h"



//和liom基本一样,不一样的地方已经标注了出来



void *ThreadsConstructA(void *threadsstruct) {
    ThreadsStruct *p = ((ThreadsStruct *) threadsstruct);
    for (auto it : p->sub_factors) {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++) {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 4)
                size_i = 3;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].rightCols(size_i); //liom中为leftCols
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++) {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 4)
                    size_j = 3;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].rightCols(size_j); //liom中为leftCols
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}


void ResidualBlockInfo::Evaluate() {
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians); //计算残差, 雅克比
    //哪种ResidualBlockInfo,就执行哪种Evaluate()
    //ImuFactor::Evaluate()
    //LidarPlaneNormFactor::Evaluate()
    //LidarEdgeFactor::Evaluate()
    //MarginalizationFactor::Evaluate()


    //marg位姿的imu预积分残差的loss function = nullptr, 上次滑窗marg的残差loss function = nullptr, 其他的not nullptr
    if (loss_function) {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho); //ceres::CauchyLoss::Evaluate();
        //ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        } else {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));

        }

        residuals *= residual_scaling_;
    }
}




MarginalizationInfo::~MarginalizationInfo() {
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int) factors.size(); i++) {

        delete[] factors[i]->raw_jacobians;

        delete factors[i]->cost_function;

        delete factors[i];
    }
}

void MarginalizationInfo::AddResidualBlockInfo(ResidualBlockInfo *residual_block_info) {
    factors.emplace_back(residual_block_info);

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++) {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    if(residual_block_info->drop_set.size() == 0) return;

    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++) {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;  //<marg位姿自己的参数块的地址, 0>
    }
}


void MarginalizationInfo::PreMarginalize() {
    for (auto it : factors) {//factors: marg对象的所有ResidualBlockInfo
        it->Evaluate(); //计算每个残差项，以及残差对优化变量的雅克比

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end()) {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data; //找到每个参数块地址对应的raw指针
            }
        }
    }
}


int MarginalizationInfo::LocalSize(int size) const {
    return size == 4 ? 3 : size;
}


void MarginalizationInfo::Marginalize() {
    int pos = 0;
    for (auto &it : parameter_block_idx) { 
        it.second = pos;
        pos += LocalSize(parameter_block_size[it.first]);
    }
    //tmpTrans[0]地址:      offset=0
    //tmpQuat[0]地址:       offset=3
    //tmpSpeedBias[0]地址:  offset=6

    m = pos; //m=15=3+3+9

    for (const auto &it : parameter_block_size) { //<参数块的地址, 每个参数块的大小>
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) {
            parameter_block_idx[it.first] = pos;
            pos += LocalSize(it.second);
        }
    }
    //计算窗口内非marg帧参数块的偏移量

    n = pos - m; //前m个是需要marg掉的, 剩下n个是需要保留的, pos: 滑窗内所有参数块维数之和
    //n = 30 ? 

    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();

    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    for (auto it : factors) {//factors: marg对象的所有ResidualBlockInfo
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos, pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;  //<每个参数块的地址, 每个参数块的大小>
        threadsstruct[i].parameter_block_idx = parameter_block_idx;    //<每个参数块的地址, 在矩阵中的id> 
        int ret = pthread_create(&tids[i], NULL, ThreadsConstructA, (void *) &(threadsstruct[i]));
        if (ret != 0) {
            ROS_DEBUG("pthread_create error");
            ROS_BREAK();
        }
    }
    for (int i = NUM_THREADS - 1; i >= 0; i--) {
        pthread_join(tids[i], NULL);
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }

    //见FEJ(First Estimate Jacobians) paper: Consistency Analysis for Sliding-Window Visual Odometry
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose()); //A的左上角矩阵，保证Amm是对称矩阵
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);  //特征值分解计算实对称矩阵的逆

    Eigen::MatrixXd Amm_inv = saes.eigenvectors()
            * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal()
            * saes.eigenvectors().transpose(); //特征值 > thresh, 取特征值的逆；否则，取0

    Eigen::VectorXd bmm = b.segment(0, m); //b最上面m*1
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;
    //舒尔补, 消去X_m，剩下Xn 
    //得到 A * Xn =  b，消去marg帧的所有参数块(维数: m)，得到窗口内剩余状态的normal equation
    //其中 A = J^T * J, A是一个实对称矩阵。 J = Σ每个残差项ri对X_n的雅克比
    //其中 b = J^T * r, r = Σ每个残差项。  r = (J^T).inv() * b
    //marg掉滑窗内的第一个位姿, 会产生一个残差,下次优化时要把上次边缘化产生的残差考虑进来.

    //FEJ:
    //由于剩余状态Xn还在滑窗内,在下次迭代优化时会被继续优化.
    //当引入新的观测后，这些Xn状态量是会发生变化的，此时就和边缘化时的Xn状态量不一样了，
    //从而导致边缘化时产生的关于Xn雅可比和残差项都发生了变化,导致H矩阵的N(H)空间秩发生了变化,引入了错误的信息.
    //在每次优化迭代时：
    //对于状态Xn，残差对Xn的雅克比一直是边缘化时Xn的值带入到雅克比表达式中
    //但是状态Xn每次都在更新，只是说Xn的线性化点是固定不变的。
    //更新后的残差 r_new = r_old + J*dx, dx为Xn当前状态与边缘化时状态量的差.
   

    //对marg后的H矩阵进行特征值分解:
    //对A进行特征值分解 A = J^T * J = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose() 
    //J = sqrt(eigenvalues.asDiagonal()) * eigenvectors.transpose()

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A); //实对称矩阵特征值分解 A = V [\] V^T, V.inv() =V.trans() 
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0)); //特征值
    Eigen::VectorXd
            S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0)); //特征值逆

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    
    // FEJ, 以后计算关于先验的残差和Jacobian都在边缘化的这个线性点展开
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose(); 
    //J = [sqrt(λ1) ... sqrt(λi) ... sqrt(λn)].diag() * V^T , J可逆
    
    //边缘化产生的残差r
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
     //r = [1/sqrt(λ1) ... 1/sqrt(λi) ... 1/sqrt(λn)].diag() * V^T * b
     //r = (J^T).inv() * b
}



std::vector<double *> MarginalizationInfo::GetParameterBlocks(std::unordered_map<long, double *> &addr_shift) {
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx) {//<每个参数块的地址, 在矩阵中的id> 
        if (it.second >= m) {//marg后剩余状态的维数
            keep_block_size.push_back(parameter_block_size[it.first]); //本次滑窗第二个,第三个位姿参数块的大小
            keep_block_idx.push_back(parameter_block_idx[it.first]);   //本次滑窗第二个,第三个位姿参数块的偏移量
            keep_block_data.push_back(parameter_block_data[it.first]); //本次滑窗第二个,第三个位姿参数块的raw data
            keep_block_addr.push_back(addr_shift[it.first]); //raw指针分别指向本次滑窗中第一个位姿(被marg掉)参数块的raw data,第二个位姿参数块的raw data
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0); //not used 

    return keep_block_addr; 
}






MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info) : marginalization_info(
                                                                                               _marginalization_info) {
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size) { //上次滑窗第二个,第三个位姿参数块的大小
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    set_num_residuals(marginalization_info->n);
};


//利用边缘化得到的雅可比和残差，构造先验信息进入到非线性优化中
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n); //dx表示当前的状态量和边缘化时的状态量之间的差值
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) { //上次滑窗第二个,第三个位姿参数块的大小
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;

        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);  //上次边缘化时被保留状态Xn在当前优化中的值

        //keep_block_data: 上次滑窗第二个,第三个位姿参数块的raw data
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size); //边缘化时，被保留的状态量的值

        if (size != 4) {
            dx.segment(idx, size) = x - x0;
        }
        else {
            dx.segment<3>(idx) = 2.0 * (Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse()
                                        * Eigen::Quaterniond(x(0), x(1), x(2), x(3))).normalized().vec();
            if ((Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse() * Eigen::Quaterniond(x(0), x(1), x(2), x(3))).w()
                    < 0) {//shortest quaternion
                dx.segment<3>(idx) = -2.0 * (Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse()
                                             * Eigen::Quaterniond(x(0), x(1), x(2), x(3))).normalized().vec();
            }
        }
    }

    Eigen::Map<Eigen::VectorXd>(residuals, n) =
            marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    //更新边缘化留下的残差


    if (jacobians) {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++) { //上次滑窗第二个,第三个位姿参数块的大小
            if (jacobians[i]) {

                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->LocalSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
				
                //liom中没有，新添加
                Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
                Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
                //
				
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        jacobian(jacobians[i], n, size);
                jacobian.setZero();
				
                if(size != 4)
                    jacobian.rightCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size); //liom中为left
                else {//liom中没有，新添加
                    if ((Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse() * Eigen::Quaterniond(x(0), x(1), x(2), x(3))).w() >= 0)
                        jacobian.rightCols(size) = 2.0 * marginalization_info->linearized_jacobians.middleCols(idx, local_size) *
                                Qleft(Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse()).bottomRightCorner<3, 4>();
                    else
                        jacobian.rightCols(size) = -2.0 * marginalization_info->linearized_jacobians.middleCols(idx, local_size) *
                                Qleft(Eigen::Quaterniond(x0(0), x0(1), x0(2), x0(3)).inverse()).bottomRightCorner<3, 4>();
                }
            }
        }
    }
    return true;
}
