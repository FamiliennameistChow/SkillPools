/**************************************************************************
 * vector_demo.cpp
 *
 * @Author： bornchow
 * @Date: 2022.03.28
 *
 * @Description:
 * 向量的使用
 ***************************************************************************/
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

/**
 * @brief 向量的初始化
 */
void init_vector(){
    std::cout << "---- init methods ---- " << std::endl;

    //定义一个空向量
    std::vector<int> a;
    std::cout << " a: " << a.size() << std::endl;

    //定义一个10个元素的向量
    std::vector<int> b(10);
    std::cout << " b: " << b.size() << std::endl;

    //定义一个10个元素的向量， 并附初值
    std::vector<int> c(10, 1);
    std::cout << " c: " << c.size() << std::endl;

    //用向量c来创建向量d，并整体赋值
    std::vector<int> d(c);
    std::cout << " d: " << d.size() << std::endl;

    //用向量c的部分元素来创建向量e
    std::vector<int>e(c.begin(), c.begin()+3);
    std::cout << " e: " << e.size() << std::endl;

    //使用数值获得初值
    int nus[4] = {1, 2, 3, 4};
    std::vector<int> f(nus, nus+4);
    std::cout << " f: " << f.size() << std::endl;

    // 使用数组初始化
    std::vector<int> h {11, 12, 13, 14, 15};
    std::cout << " h: " << h.size() << std::endl;

    //错误的方法
    //    std::vector<int> g;
    //    for (int i = 0; i < 10; ++i) {
    //        g[i] = i;
    //    }
    //下标只能用于获取已存在的元素
}

/**
 * @brief 添加元素
 */
void add_element(){
    std::cout << "---- add element ---- " << std::endl;
    std::vector<int> a;

    //push_back()
    a.push_back(12);

    std::cout << " a vector: { ";
    for (int i = 0; i < a.size(); ++i) {
        std::cout << a[i] << " ";
    }
    std::cout <<"}" <<std::endl;

    //insert()
    //iterator insert(iterator it,const T& x):向量中迭代器指向元素前增加一个元素x
    a.insert(a.begin(), 13);

    std::cout << " a vector: { ";
    for (int i = 0; i < a.size(); ++i) {
        std::cout << a[i] << " ";
    }
    std::cout <<"}" <<std::endl;

    //iterator insert(iterator it,int n,const T& x):向量中迭代器指向元素前增加n个相同的元素x
    a.insert(a.begin(), 2, 22);

    std::cout << " a vector: { ";
    for (int i = 0; i < a.size(); ++i) {
        std::cout << a[i] << " ";
    }
    std::cout <<"}" <<std::endl;

    //iterator insert(iterator it,const_iterator first,const_iterator last)
    // :向量中迭代器指向元素前插入另一个相同类型向量的[first,last)间的数据
    std::vector<int> b {1, 2, 3};
    a.insert(a.begin(), b.begin(), b.end());

    std::cout << " a vector: { ";
    for (int i = 0; i < a.size(); ++i) {
        std::cout << a[i] << " ";
    }
    std::cout <<"}" <<std::endl;

}

/**
 * @brief 删除元素
 */
void erase_element(){
    std::cout << "---- erase element ---- " << std::endl;
    std::vector<int> a {1, 2, 3, 4, 5};

    // iterator erase(iterator it):删除向量中迭代器指向元素
    a.erase(a.begin());

    std::cout << " a vector: { ";
    for (int i = 0; i < a.size(); ++i) {
        std::cout << a[i] << " ";
    }
    std::cout <<"}" <<std::endl;

    // iterator erase(iterator first,iterator last):删除向量中[first,last)中元素
    a.erase(a.begin(), a.begin()+2);

    std::cout << " a vector: { ";
    for (int i = 0; i < a.size(); ++i) {
        std::cout << a[i] << " ";
    }
    std::cout <<"}" <<std::endl;

    // pop_back():删除向量中最后一个元素
    a.pop_back();

    std::cout << " a vector: { ";
    for (int i = 0; i < a.size(); ++i) {
        std::cout << a[i] << " ";
    }
    std::cout <<"}" <<std::endl;

    //clear():清空向量中所有元素
    a.clear();
    std::cout << " a size: " << a.size() << std::endl;
}

/**
 * @brief 获取元素
 */
void get_element(){
    std::cout << "---- get element ---- " << std::endl;

    std::vector<int> v {11, 21, 31, 41};

    //at(int pos):返回pos位置元素的引用
    std::cout <<" v at 1: " << v.at(1) << std::endl;

    //下标
    std::cout <<" v [1] : " << v[1] << std::endl;

    //front():返回首元素的引用
    std::cout << " v front() 0 : " << v.front() << std::endl;

    //back():返回尾元素的引用
    std::cout << " v back() -1 : " << v.back() <<std::endl;

    //begin() 返回向量头指针，指向第一个元素
    std::cout << " v begin() 0 : " << *(v.begin()) << std::endl;

    // end():返回向量尾指针，指向向量最后一个元素的下一个位置 注意这里不是指向最后一个元素
    std::cout << " v end() 0 : " << *(v.end()) << std::endl;

    // rbegin():反向迭代器，指向最后一个元素
    std::cout << " v rbegin() -1: " << *(v.rbegin()) << std::endl;

    // rend():反向迭代器，指向第一个元素之前的位置 注意不是指向第一个元素
    std::cout << " v rend() 0 : " << *(v.rend()) << std::endl;

}

/**
 * @brief 向量的大小
 */
void vector_size(){
    std::cout << "---- vector size ---- " << std::endl;

    std::vector<int> v {12, 13, 14, 15};

    std::cout << " v size : " << v.size() << std::endl;

    std::cout <<" v empty : " << v.empty() << std::endl;

}

/**
 * @brief 交换两个向量
 */
void swap_vector(){
    std::cout << " ---- vector swap ---- " << std::endl;
    std::vector<int> a {1, 2, 3};

    std::cout << " vector a: { ";
    for(size_t i = 0; i < a.size(); i++){
        std::cout << a[i] << " ";
    }
    std::cout << "} " << std::endl;

    std::vector<int> b {11, 12, 13};

    std::cout << " vector b: { ";
    for(size_t i = 0; i < b.size(); i++){
        std::cout << b[i] << " ";
    }
    std::cout << "} " << std::endl;

    std::cout << " -----after swap----" <<std::endl;

    // swap()
    a.swap(b);

    std::cout << " vector a: { ";
    for(size_t i = 0; i < a.size(); i++){
        std::cout << a[i] << " ";
    }
    std::cout << "} " << std::endl;

    std::cout << " vector b: { ";
    for(size_t i = 0; i < b.size(); i++){
        std::cout << b[i] << " ";
    }
    std::cout << "} " << std::endl;

}

/**
 * @brief 遍历vector
 */
void traversal(){
    std::cout << " ---- vector traver ---- " << std::endl;
    std::vector<int> vec {1, 2, 3, 4};

    //方式一:
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;

    //方式二:
    for(auto v: vec){
        std::cout << v << " ";
    }
    std::cout << std::endl;

    //方式三：
    for(std::vector<int>::iterator iter = vec.begin(); iter != vec.end(); iter++){
        std::cout << *(iter) << " ";
    }
    std::cout << std::endl;

    //方式四：
    for(auto iter = vec.begin(); iter != vec.end(); iter++){
        std::cout << *(iter) << " ";
    }
    std::cout << std::endl;

}

/**
 * @brief 排序
 */
#include <algorithm>
//自己定义比较函数，大的写在前面
bool camp_max(int x, int y){
    return x > y;
}
void sort_vector(){
    std::cout << " ---- vector sort ---- " << std::endl;
    std::vector<int> a {4, 2, 5, 6, 1};

    // 默认从小到大
    sort(a.begin(), a.end());

    std::cout << " sort up: ";
    for(auto v: a){
        std::cout << v << " ";
    }
    std::cout << std::endl;

    //从大到小
    //方法一：使用 greater<int>()函数
    std::vector<int> b {4, 2, 5, 6, 1};

    sort(b.begin(), b.end(), greater<int>());
    std::cout << " sort down 1: ";
    for(auto v: b){
        std::cout << v << " ";
    }
    std::cout << std::endl;

    //方法二：
    std::vector<int> c {4, 2, 5, 6, 1};
    sort(c.rbegin(), c.rend());
    std::cout << " sort down 2: ";
    for(auto v: c){
        std::cout << v << " ";
    }
    std::cout << std::endl;

    //方法三： 自定义规则函数camp_max() -- 这个函数需要是全局函数 大的写在前面
    std::vector<int> d {4, 2, 5, 6, 1};
    sort(d.begin(), d.end(), camp_max);

    std::cout << " sort down 3: ";
    for(auto v: d){
        std::cout << v << " ";
    }
    std::cout << std::endl;

    //方式四：使用sort排序后，使用reverse() 将元素倒置，但不排列
    std::vector<int> f {4, 2, 5, 6, 1};

    sort(f.begin(), f.end());
    reverse(f.begin(), f.end());

    std::cout << " sort down 4: ";
    for(auto v: f){
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief 对结构体vector排序
 * 以A-star算法中，节点结构体为例(f = g + h )，我们的排序规则是：
 * 1. f不等，按f从小到大排序
 * 2. 当f相等，按h从小到大排序
 * https://blog.csdn.net/xf_zhen/article/details/51272278?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&utm_relevant_index=2
 */
 typedef struct Node{
     int x;
     int y;
     float f;
     float g;
     float h;
     Node *father_node;
     Node(int x_, int y_) : x(x_), y(y_){
         this->f = 0;
         this->h = 0;
         this->g = 0;
         this->father_node = NULL;
     }

     //重载输出
     friend ostream  &operator << (ostream&, Node &n){
         if (n.father_node == NULL){
             cout << " [ " << n.x << " , " <<  n.y << " ] -- " << "GHF: " << n.g << " , " << n.h << " , " << n.f <<
                  " -- {   NULL   } ";
         } else{
             cout << " [ " << n.x << " , " << n.y << " ] -- " << "GHF: " << n.g << " , " << n.h << " , " << n.f <<
                  " -- { " << n.father_node->x << " , " << n.father_node->y << " } ";
         }
         return cout;
     }

     //方式一: 结构体内重载比较函数
     //升序排序 <
     // 加const强调函数里不会去修改这个引用，如果修改则会编译出错 传递引用是考虑到效率，不需要复制
     bool operator < (const Node &n) const
     {
         if (this->f == n.f){
             return this->h < n.h;
         } else{
             return this->f < n.f;
         }
     }

     //降序排序 >
     bool operator > (const Node &n) const
     {
         if(this->f == n.f){
             return this->h >n.h;
         } else{
             return this->f > n.f;
         }
     }

 }Node;

// 方式二：结构体外排序
// 全局函数
// 升序排序 <
bool less_camp(const Node &n1, const Node &n2){
    if(n1.f == n2.f){
        return n1.h < n2.h;
    }else{
        return n1.f < n2.f;
    }
}

//降序 >
bool greater_camp(const Node &n1, const Node &n2){
    if(n1.f == n2.f){
        return n1.h > n2.h;
    }else{
        return n1.f > n2.f;
    }
}


void sort_struct(){
    std::cout << " ----struct vector sort ---- " << std::endl;
    std::vector<Node> Nodes;
    Node n1(1, 1);
    n1.f = 10;
    n1.g = 0;
    n1.h = 10;
    Nodes.push_back(n1);

    Node n2(2, 2);
    n2.f = 10;
    n2.g = 1;
    n2.h = 9;
    Nodes.push_back(n2);

    Node n3(3,3);
    n3.f = 11;
    n3.g = 2;
    n3.h = 10;
    Nodes.push_back(n3);

    //升序排序
    //sort(Nodes.begin(), Nodes.end());
    sort(Nodes.begin(), Nodes.end(), less<Node>());

    std::cout << " ----up order--------" << std::endl;
    for(auto n : Nodes){
     std::cout << n << std::endl;
    }
    std::cout << " ------------" << std::endl;

    //降序排序
    sort(Nodes.begin(), Nodes.end(), greater<Node>());
    std::cout << " ----down order--------" << std::endl;
    for(auto n : Nodes){
     std::cout << n << std::endl;
    }
    std::cout << " ------------" << std::endl;

    std::cout << " ----- method outside struct ------" <<std::endl;

    sort(Nodes.begin(), Nodes.end(), less_camp);
    std::cout << " ----up order--------" << std::endl;
    for(auto n : Nodes){
    std::cout << n << std::endl;
    }
    std::cout << " ------------" << std::endl;

    sort(Nodes.begin(), Nodes.end(), greater_camp);
    std::cout << " ----down order--------" << std::endl;
    for(auto n : Nodes){
        std::cout << n << std::endl;
    }
    std::cout << " ------------" << std::endl;

 }

/**
 * @brief 查找vector中的最大最小值
 */
void find_max_min(){
    std::vector<double> v {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0};

    std::vector<double>::iterator biggest = std::max_element(std::begin(v), std::end(v));
    //or std::vector<double>::iterator biggest = std::max_element(v.begin(), v.end);

    std::cout << "Max element is " << *biggest<< " at position " <<std::distance(std::begin(v), biggest) << std::endl;
    //另一方面，取最大位置也可以这样来写：
    //int nPos = (int)(std::max_element(v.begin(), v.end()) - (v.begin());
    //效果和采用distance(...)函数效果一致
    //说明：max_element(v.begin(), v.end()) 返回的是vector<double>::iterator,
    //相当于指针的位置，减去初始指针的位置结果即为最大值得索引。

    auto smallest = std::min_element(std::begin(v), std::end(v));
    std::cout << "min element is " << *smallest<< " at position " <<std::distance(std::begin(v), smallest) << std::endl;
}

void remove_overlap(){
    std::vector<int> a {1, 1, 16, 3, 24, 5, 5, 6,};
    std::sort(a.begin(), a.end());

    std::cout << "{ ";
    for(auto i : a){
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;

    auto it = unique(a.begin(), a.end());

    std::cout << " ----- after unique ------" << std::endl;
    std::cout << "{ ";
    for(auto i : a){
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;
    std::cout << " ------it " << *it << std::endl;

    a.erase(it, a.end());
    std::cout << " ----- after erase ------" << std::endl;
    std::cout << "{ ";
    for(auto i : a){
        std::cout << i << " ";
    }
    std::cout << "}" << std::endl;

//    std::sort(a.begin(), a.end());
//    a.erase(unique(a.begin(),a.end()), a.end());


}


int main() {
    std::cout << "this is a vector use demo" << std::endl;

    //init_vector();

    // add_element();

    // erase_element();

    // get_element();

    // vector_size();

    // swap_vector();

    // traversal();

    //sort_vector();

    // sort_struct();

    //find_max_min();


     remove_overlap();



    return 0;
}
