#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

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



int main() {
    std::cout << "this is a vector use demo" << std::endl;

    find_max_min();

    return 0;
}
