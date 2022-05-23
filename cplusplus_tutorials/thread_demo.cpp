/**************************************************************************
 * thread_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.05.21
 * 
 * @Description:
 *  这里是多线程demo
 ***************************************************************************/

#include <thread>
#include <iostream>
#include <mutex>
#include <chrono>

int i;

std::mutex mut_i;
int cntr1 =0, cntr2 =0;

void main_func(){

    while (i < 100){
        mut_i.lock();
        std::cout << " thread 1: " << i << std::endl;
        i++;
        mut_i.unlock();
        cntr1++;
        std::this_thread::sleep_for(std::chrono::microseconds (1));
    }

}

void cout_func(){;

    while (i < 100){
        mut_i.lock();
        std::cout << " thread 2: " << i << std::endl;
        i++;
        mut_i.unlock();
        cntr2++;
        std::this_thread::sleep_for(std::chrono::microseconds (1));
    }


}

int main(){


    std::thread cout_thread(cout_func);
    cout_thread.detach();
    main_func();

}


