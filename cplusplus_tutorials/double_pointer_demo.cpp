
/**************************************************************************
 * double_pointer_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.05.09
 * 
 * @Description:
 * 双指的用法
 ***************************************************************************/

//  快慢指针
// leetcode 27. 移除元素

#include <iostream>
#include <vector>
using namespace std;

int removeElement(vector<int>& nums, int val) {
    int slowInd = 0;
    for(int fastInd = 0; fastInd < nums.size(); fastInd++){
        if(val != nums[fastInd]){
            nums[slowInd++] = nums[fastInd];
        }
        std::cout << "fastInd: " << fastInd << " slowInd: " << slowInd;
        std::cout << " -- [ ";
        for (int i = 0; i < nums.size(); ++i) {
            std::cout << nums[i] << " ";
        }
        std::cout << "] " << std::endl;
    }
    return slowInd;
}

int removeDuplicates(vector<int>& nums) {
    int slowInd = 0;
    for(int fastInd = 0; fastInd< nums.size() -1; fastInd++){
        if(nums[fastInd] != nums[fastInd+1]){
            nums[slowInd] = nums[fastInd];
            slowInd++;
        }

        std::cout << "fastInd: " << fastInd << " slowInd: " << slowInd;
        std::cout << " -- [ ";
        for (int i = 0; i < nums.size(); ++i) {
            std::cout << nums[i] << " ";
        }
        std::cout << "] " << std::endl;
    }
//    slowInd++;
    std::cout << nums.back()<< std::endl;
    nums[slowInd] = nums.back();
    slowInd++;
    std::cout << " slowInd " <<slowInd <<std::endl;
    for (int i = 0; i < nums.size(); ++i) {
        std::cout << nums[i] << " ";
    }
    std::cout << "] " << std::endl;
    return slowInd;
}

int main(){

//    std::vector<int>input_nums {0,1,2,2,3,0,4,2};
//    int ind = removeElement(input_nums, 2);

    std::vector<int>input_nums {0,0,1,1,1,2,2,3,3,4,4};
//    std::vector<int>input_nums {1, 2, 3};
    int ind = removeDuplicates(input_nums);

    std::cout<< " [ " ;
    for(size_t i = 0; i < ind; i++){
        std::cout << input_nums[i] << " ";
    }
    std::cout << "]" << std::endl;
}

