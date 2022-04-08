/**************************************************************************
 * struct_demo2.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.03.23
 * 
 * @Description:
 *  结构体的构造函数
 ***************************************************************************/
#include <iostream>
using namespace std;


typedef struct Student{
    string name;
    int age;
    float score;
    
    Student(string name, int age, float score){
        this->name = name;
        this->age = age;
        this->score = score;
    } //构造函数一

    Student(string _name, int _age) :
    name(_name),
    age(_age)
    {
        this->score = 0;
    } // 构造函数二

    Student(){ }//一旦自定义构造函数了，那么默认不可见的构造函数就被覆盖了，所以需要显示默认构造函数
}Student;


int main(){
    // 声明
    Student born("bron", 18);
    std::cout << " " << born.name << " " << born.age << " " << born.score << std::endl;

    //使用指针声明
    Student *kiity = new Student ("kiity", 19, 100);
    std::cout << " " << kiity->name << " " << kiity->age << " " << kiity->score << std::endl;

}

