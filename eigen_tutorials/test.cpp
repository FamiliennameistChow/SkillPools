/**************************************************************************
 * test.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.03.19
 * 
 * @Description:
 *  
 ***************************************************************************/
#include <iostream>
#include <math.h>
using namespace std;

int main(){

    float px = 19.9;
    float p_ori_x = -20.0;
    float re = 0.5;

    float map = floor((px-p_ori_x) / re);
    std::cout << "map " << map << std::endl;
}