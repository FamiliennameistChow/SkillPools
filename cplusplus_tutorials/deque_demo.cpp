/**************************************************************************
 * deque_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.04.06
 * 
 * @Description:
 *  deque容器为一个给定类型的元素进行线性处理，像向量一样，它能够快速地随机访问任一个元素，
 *  并且能够高效地插入和删除容器的尾部元素。但它又与vector不同，
 *  deque支持高效插入和删除容器的头部元素，因此也叫做双端队列
 ***************************************************************************/

#include <iostream>
#include <deque>

using namespace std;

/**
 * @brief 申明与初始化
 */
void init_deque(){
    std::cout << " ---------- init deque -----------" << std::endl;
    // 方式一: 创建一个空deque
    std::deque<int> a;
    std::cout << " size deque a: " << a.size() << std::endl;

    // 方式二: deque(int nSize):创建一个deque,元素个数为nSize
    std::deque<int> b(10);
    std::cout << " size deque b: " << b.size() << std::endl;

    // 方式三: deque(int nSize,const T& t):创建一个deque,元素个数为nSize,且值均为t
    std::deque<int> c(10 , 5);
    std::cout << " size deque c: " << c.size() << std::endl;

    // 方式四: deque(const deque &):复制构造函数
    std::deque<int> d(b);
    std::cout << " size deque d: " << d.size() << std::endl;

}

void cross_element(std::deque<int> de){
    std::cout << " deque : { ";
    for(size_t i = 0; i <de.size(); i++){
        std::cout << de[i] << " ";
    }
    std::cout<< "}" << std::endl;
}

/**
 * @brief 添加元素
 */
void add_element(){
    std::cout << " ---------- add element -----------" << std::endl;

    std::deque<int> a(2, 5);
    cross_element(a);

    // void push_front(const T& x):双端队列头部增加一个元素X
    a.push_front(8);
    cross_element(a);

    //void push_back(const T& x):双端队列尾部增加一个元素x
    a.push_back(7);
    cross_element(a);

    // iterator insert(iterator it,const T& x):双端队列中某一元素前增加一个元素x
    a.insert(a.begin()+1, 2);
    cross_element(a);

    // void insert(iterator it,int n,const T& x):双端队列中某一元素前增加n个相同的元素x
    a.insert(a.begin() + 1, 2, 3);
    cross_element(a);

    std::deque<int> b(3, 0);
    // void insert(iterator it,const_iterator first,const_iteratorlast):
    // 双端队列中某一元素前插入另一个相同类型向量的[forst,last)间的数据
    a.insert(a.begin() + 2, b.begin(), b.end());
    cross_element(a);
}

/**
 * @brief 删除元素
 */
void erase_element(){
    std::cout << " ---------- erase element -----------" << std::endl;

    std::deque<int> a;
    for (int i = 1; i < 10; ++i) {
        a.push_back(i);
    }
    cross_element(a);

    // Iterator erase(iterator it):删除双端队列中的某一个元素
    auto it = a.erase(a.begin() + 2);
    cross_element(a);
    std::cout << *it << std::endl; //  返回删除元素的下一个元素

    //Iterator erase(iterator first,iterator last):删除双端队列中[first,last）中的元素
    auto it2 = a.erase(a.begin()+1, a.begin()+4);
    cross_element(a);
    std::cout << *it2 << std::endl;

    //void pop_front():删除双端队列中最前一个元素
    a.pop_front();
    cross_element(a);

    //void pop_back():删除双端队列中最后一个元素
    a.pop_back();
    cross_element(a);

    //清空双端队列中的元素
    a.clear();
    cross_element(a);

}

/**
 * @brief 遍历元素、获取元素
 */
void cross_deque(){
    std::cout << " ---------- cross element -----------" << std::endl;
    std::deque<int> a;
    for (int i = 1; i < 10; ++i) {
        a.push_back(i);
    }


    // 遍历元素一：
    std::cout << " deque : { ";
    for(size_t i = 0; i <a.size(); i++){
        std::cout << a[i] << " ";
    }
    std::cout<< "}" << std::endl;

    //遍历元素二:
    std::cout << " deque : { ";
    for(int i : a){
        std::cout << i << " ";
    }
    std::cout<< "}" << std::endl;

    //遍历元素三:
    std::cout << " deque : { ";
    for(std::deque<int>::iterator it = a.begin(); it != a.end(); it++){
        std::cout << *it << " ";
    }
    std::cout<< "}" << std::endl;

    // 获取某一元素
    // reference at(int pos):返回pos位置元素的引用
    std::cout << "a.at(2) : " << a.at(2) << std::endl;

    // 下标
    std::cout << "a[2] : " << a[2] << std::endl;

    // reference front():返回首元素的引用
    std::cout << "a.front(): " << a.front() << std::endl;

    // reference back():返回尾元素的引用
    std::cout << "a.back(): " << a.back() << std::endl;

    // iterator begin():返回向量头指针，指向第一个元素
    std::cout << "*a.begin(): " << *a.begin() << std::endl;

    // iterator end():返回指向向量中最后一个元素下一个元素的指针（不包含在向量中）
    std::cout << "*a.end() : " << *a.end()<< std::endl;

    //reverse_iterator rbegin():反向迭代器，指向最后一个元素
    std::cout << "*a.rbegin(): " << *a.rbegin() << std::endl;

    //reverse_iterator rend():反向迭代器，指向第一个元素的前一个元素
    // std::cout << "*a.rend(): " << *a.rend() << std::endl;
    std::cout << "*a.rend():  is error" << std::endl;

}

/**
 * @brief 其他函数
 */
void other_func(){
    std::cout << " ---------- other fnuc -----------" << std::endl;
    std::deque<int> a;
    for (int i = 1; i < 10; ++i) {
        a.push_back(i);
    }

    // Int size() const:返回向量中元素的个数
    std::cout << " size: " << a.size() << std::endl;

    //int max_size() const:返回最大可允许的双端对了元素数量值
    std::cout << " max size: " << a.max_size() << std::endl;

    // bool empty() const:向量是否为空，若true,则向量中无元素
    std::cout << " empty(): " << a.empty() << std::endl;

    // void swap(deque&):交换两个同类型向量的数据
    std::deque<int> b(3, 5);

    std::cout << " a : ";
    cross_element(a);

    std::cout << " b : ";
    cross_element(b);

    a.swap(b);
    std::cout << " ---- after swap ----" << std::endl;

    std::cout << " a : ";
    cross_element(a);

    std::cout << " b : ";
    cross_element(b);

    //void assign(int n,const T& x):改变队列的内容 一共有n个元素，每个元素的值都是x
    // 注意这个不是指改变单个元素，而是指改变整个队列
    b.assign(2, 0);
    cross_element(b);

}

int main(){
    std::cout << " ----------- this is deque demo -----------" << std::endl;

    // init_deque();

    // add_element();

    // erase_element();

    // cross_deque();

    other_func();

}

