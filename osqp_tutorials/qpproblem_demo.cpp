//******************************
// Data: 2022.07.14
// 
// 测试 QPProblem 封装
//
//********************

#include "QPProblem.hpp"

//
//               J = 1/2 x^TPx + q^Tx
//               s.t. l <= Ax <= u
//                  
//                P = [[4 1]     q= [[1]   l= [[1]    u= [[1.0]     A = [[1 1]
//                     [1 2]]        [1]]      [0]        [0.7]          [1 0]
//                                             [0]]       [0.7]]         [0 1]]


int main(){
    QPProblem<c_float> qp;
    
    int n = 2;
    int m = 3;

    //初始化
    qp.initialize(n ,m);

    // 加入数据
    qp.A_.addElement(0, 0, 1.0);
    qp.A_.addElement(0, 1, 1.0);
    qp.A_.addElement(1, 0, 1.0);
    qp.A_.addElement(2, 1, 1.0);

    qp.l_[0] = 1.0;
    qp.l_[1] = 0;
    qp.l_[2] = 0;

    qp.u_[0] = 1.0;
    qp.u_[1] = 0.7;
    qp.u_[2] = 0.7;

    qp.q_[0] = 1.0;
    qp.q_[0] = 1.0;

    qp.P_.addElement(0, 0, 4.0);
    qp.P_.addElement(0, 1, 1.0);
    qp.P_.addElement(1, 1, 2.0);

    //solve
    int error_code;
    OSQPSolution* solution = qp.solve(&error_code);

    std::cout << " solution x: { ";
    for(int i=0; i<n; i++){
        std::cout << solution->x[i] << " ";
    }
    std::cout << "}" << std::endl;




}