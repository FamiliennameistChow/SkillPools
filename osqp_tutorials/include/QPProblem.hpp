//****************************************************
// Date: 2022.7.14
// 
// 使用osqp解QP问题的封装
// 
//
//****************************************************

#include <iostream>
#include <vector>
#include "osqp.h"
#include <algorithm>

template<typename T>
struct SparseMatrixElement {
    int r, c;
    T v;

    inline bool operator<(const SparseMatrixElement &rhs) {
        return (c == rhs.c) ? (r < rhs.r) : (c < rhs.c);
    }
};


// 稀疏矩阵类
template<typename T>
class SparseMatrix {
private:
    int m_, n_;

    std::vector< SparseMatrixElement<T> > elements_;

    std::vector<T> osqp_csc_data_;  //相当于稀疏矩阵的 x 表示所有非零元素的值， 按列统计
    std::vector<c_int> osqp_csc_row_idx_; //非零元素的行号, 这个需要借助osqp_csc_col_start_来确定
    std::vector<c_int> osqp_csc_col_start_; //记录每列的非零元素数量， osqp_csc_col_start_[0]=0, 第i列的非零元素个数为osqp_csc_col_start_[i+1]-osqp_csc_col_start_[i],
    csc *osqp_csc_instance = nullptr; //稀疏矩阵

    void freeOSQPCSCInstance();

public:
    SparseMatrix();
    ~SparseMatrix();

    void initialize(int m, int n);
    void addElement(int r, int c, T v);
    csc *toOSQPCSC();
};

template<typename T>
SparseMatrix<T>::SparseMatrix() {
    m_ = n_ = 0;
}

template<typename T>
SparseMatrix<T>::~SparseMatrix() {
    elements_.clear();
    freeOSQPCSCInstance();
}

// 相当于clear()
template<typename T>
void SparseMatrix<T>::freeOSQPCSCInstance() {
    osqp_csc_data_.clear();
    osqp_csc_row_idx_.clear();
    osqp_csc_col_start_.clear();

    if(osqp_csc_instance != nullptr) {
        c_free(osqp_csc_instance);
        osqp_csc_instance = nullptr;
    }
}

// r 行 c 列  v 值
// 这里加入的是非零元素的行列与值
template<typename T>
void SparseMatrix<T>::addElement(int r, int c, T v) {
    elements_.push_back({r, c, v});
}

template<typename T>
void SparseMatrix<T>::initialize(int m, int n) {
    m_ = m;
    n_ = n;
    elements_.clear();
}

// 将自定义的std::vector< SparseMatrixElement<T> > 转为  csc格式
template<typename T>
csc* SparseMatrix<T>::toOSQPCSC() {
    freeOSQPCSCInstance();

    sort(elements_.begin(), elements_.end());

    int idx = 0;
    int n_elem = elements_.size();

    osqp_csc_col_start_.push_back(0);
    for(int c = 0; c < n_; c++) {
        while((idx < n_elem) && elements_[idx].c == c) {
            osqp_csc_data_.push_back(elements_[idx].v);
            osqp_csc_row_idx_.push_back(elements_[idx].r);
            idx++;
        }

        osqp_csc_col_start_.push_back(osqp_csc_data_.size());
    }

    osqp_csc_instance = csc_matrix(m_, n_, osqp_csc_data_.size(), osqp_csc_data_.data(), osqp_csc_row_idx_.data(), osqp_csc_col_start_.data());
    return osqp_csc_instance;
}


// -----------------------------------------------------

// QP类
template<typename T>
class QPProblem {
private:
    OSQPWorkspace *osqp_workspace_ = nullptr;
    OSQPSettings  *osqp_settings_= nullptr;
    OSQPData      *osqp_data_ = nullptr;

public:
    

    //number of variables and constraints
    int n_, m_;

    //constraints
    SparseMatrix<T> A_;
    std::vector<T> l_, u_;

    //cost function
    SparseMatrix<T> P_;
    std::vector<T> q_;

    ~QPProblem();
    void initialize(int n, int m);
    OSQPSolution* solve(int *error_code);
};

template<typename T>
QPProblem<T>::~QPProblem() {
    if(osqp_workspace_ != nullptr) {
        osqp_workspace_->data->P = nullptr;
        osqp_workspace_->data->q = nullptr;

        osqp_workspace_->data->A = nullptr;
        osqp_workspace_->data->l = nullptr;
        osqp_workspace_->data->u = nullptr;

        //cleanup workspace
        osqp_cleanup(osqp_workspace_);
    }
}

// m 约束数   n 变量数
template<typename T>
void QPProblem<T>::initialize(int n, int m) {
    n_ = n;
    m_ = m;

    A_.initialize(m_, n_);
    l_.resize(m_);
    u_.resize(m_);

    std::fill(l_.begin(), l_.end(), 0);
    std::fill(u_.begin(), u_.end(), 0);

    P_.initialize(n_, n_);
    q_.resize(n_);

    std::fill(q_.begin(), q_.end(), 0);
}

template<typename T>
OSQPSolution* QPProblem<T>::solve(int *error_code) {
    //set up workspace
    if(osqp_workspace_ == nullptr) {
        osqp_settings_ = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
        osqp_data_     = (OSQPData *)    c_malloc(sizeof(OSQPData));

        //populate data
        osqp_data_->n = n_;
        osqp_data_->m = m_;

        osqp_data_->A = A_.toOSQPCSC();
        osqp_data_->l = l_.data();
        osqp_data_->u = u_.data();

        osqp_data_->P = P_.toOSQPCSC();
        osqp_data_->q = q_.data();

        osqp_set_default_settings(osqp_settings_);
        osqp_setup(&osqp_workspace_, osqp_data_, osqp_settings_);
    }
    else {
        csc *A_csc = A_.toOSQPCSC();
        osqp_update_A(osqp_workspace_, A_csc->x, NULL, A_csc->nzmax);
        osqp_update_bounds(osqp_workspace_, l_.data(), u_.data());

        csc *P_csc = P_.toOSQPCSC();
        osqp_update_P(osqp_workspace_, P_csc->x, NULL, P_csc->nzmax);
        osqp_update_lin_cost(osqp_workspace_, q_.data());
    }

    *error_code = osqp_solve(osqp_workspace_);

    return osqp_workspace_->solution;
}