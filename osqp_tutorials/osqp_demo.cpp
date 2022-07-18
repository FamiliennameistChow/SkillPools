//****************************************************
// Date: 2022.7.14
// 
//
// osqp使用demo
// https://osqp.org/docs/interfaces/C.html#_CPPv413OSQPWorkspace
//****************************************************
#include "osqp.h"
#include <iostream>

//
//               J = 1/2 x^TPx + q^Tx
//               s.t. l <= Ax <= u
//                  
//                P = [[4 1]     q= [[1]   l= [[1]    u= [[1.0]     A = [[1 1]
//                     [1 2]]        [1]]      [0]        [0.7]          [1 0]
//                                             [0]]       [0.7]]         [0 1]]


int main(int argc, char **argv) {
    // Load problem data
    c_float P_x[3] = { 4.0, 1.0, 2.0, }; // P矩阵的上三角矩阵
    c_int   P_nnz  = 3;
    c_int   P_i[3] = { 0, 0, 1, };  //非零元素的行号
    c_int   P_p[3] = { 0, 1, 3, };
    c_float q[2]   = { 1.0, 1.0, };
    c_float A_x[4] = { 1.0, 1.0, 1.0, 1.0, }; //A的数据，第1列（0，1行）对应的非零元素为1.0，1.0，第2列（0，2行）对应的非零元素为1.0，1.0。
    c_int   A_nnz  = 4;    // 非零元素总数为4
    c_int   A_i[4] = { 0, 1, 0, 2, };   // 非零元素的行号, 第1列对应的非零元素在0,1行，第2列对应的非零元素在0,2行
    c_int   A_p[3] = { 0, 2, 4, };      // 记录每列的非零元素数量, 第i列的非零元素个数为A_p[i+1]-A_p[i], 如第0列非零元素个数为：2-0, 第1列非零元素个数为：4-2 
    c_float l[3]   = { 1.0, 0.0, 0.0, };
    c_float u[3]   = { 1.0, 0.7, 0.7, };
    c_int n = 2; //变量数
    c_int m = 3; //约束数

    // Exitflag
    c_int exitflag = 0;

    // Workspace structures
    OSQPWorkspace *work;
    OSQPSettings  *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    OSQPData      *data     = (OSQPData *)c_malloc(sizeof(OSQPData));

    // Populate data
    if (data) {
    data->n = n;
    data->m = m;
    data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
    data->q = q;
    data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
    data->l = l;
    data->u = u;
    }

    std::cout << "P { ";
    for(int i=0; i<P_p[2]; i++){
        std::cout << data->P->x[i] << " ";
    }
    std::cout << "}" << std::endl;

    // Define solver settings as default
    if (settings) osqp_set_default_settings(settings);

    // Setup workspace
    exitflag = osqp_setup(&work, data, settings);

    // Solve Problem
    osqp_solve(work);


    // Solution
    OSQPSolution* res = work->solution;
    std::cout << " soultion x: { ";
    for(int i=0; i<n; i++){
    std::cout << res->x[i] << " ";
    }
    std::cout << "}" << std::endl;

    std::cout << " soultion Y: { ";
    for(int i=0; i<m*m; i++){
    std::cout << res->y[i] << " ";
    }
    std::cout << "}" << std::endl;


    // Clean workspace
    osqp_cleanup(work);
    if (data) {
    if (data->A) c_free(data->A);
    if (data->P) c_free(data->P);
    c_free(data);
    }
    if (settings)  c_free(settings);

    return exitflag;
}
