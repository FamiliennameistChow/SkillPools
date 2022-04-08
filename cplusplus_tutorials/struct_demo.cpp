/**************************************************************************
 * struct_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.03.23
 * 
 * @Description:
 * 结构体的定义与使用
 ***************************************************************************/
#include <iostream>
using namespace std;

// ************** 1.结构体的定义:
// 方式一 推荐
struct Person{
    string name;
    int age;
};

// 方式二
struct Student{
    string name;
    int age;
}student;

//方式三
struct Man{
    string name;
    int age;
}man{"Bob", 33};

//方式四 推荐
typedef struct Women{
    string name;
    int age;
}Women;
// 该方法完成了两步
//第一步定义一个struct Women
/*
struct Women{
    string name;
    int age;
};
*/
//第二步
// typedef Struct Women Women


void useStruct(){
    // ********** 2.结构体的声明与初始化
    // 2.1 直接声明
    // 方式一 使用直接赋值初始化
    // Person zhang_san 定义变量
    // Person boys[5]
    Person zhang_san;
    zhang_san.name = "zhang san";
    zhang_san.age = 18;
    std::cout <<" " << zhang_san.name << " " << zhang_san.age << std::endl;

//    student.name = "bron";
//    student.age = 22;
//    std::cout <<" " << student.name << " " << student.age << std::endl;

    // 方式二 使用初始化列表初始化
    Person wang_er={"wang er", 12};
    std::cout << " " << wang_er.name << " " << wang_er.age << std::endl;

    Person classOne[2] = {{"lili", 12}, {"fangfang", 12}};
    std::cout << " " << classOne[0].name << " " << classOne[0].age << std::endl;

    student = {"born", 23};
    std::cout <<" " << student.name << " " << student.age << std::endl;

    std::cout <<" " << man.name << " " << man.age << std::endl;

    //2.2 使用指针声明
    // 错误初始化     这份代码会报一个错：空指针访问异常，这是因为li_si这个指针还没有初始化，因此他没有内存空间，自然就不存在有name和age这个参数
//    Person *li_si;
//    li_si->name = "li si";
//    li_si->age = 19;
//    std::cout << " " << li_si->name << " " << li_si->age << std::endl;
    Person p;
    p.name ="ming_ming";
    p.age = 13;
    Person *ming_ming;
    ming_ming = &p;
    std::cout << "指针： " << ming_ming->name << " " << ming_ming->age << std::endl;
    // 注意当p的值发生变化时，ming_ming的值也会变化
    p.age = 16;
    std::cout << "指针： after change " << ming_ming->name << " " << ming_ming->age << std::endl;

    Person *zhang = new Person();
    zhang->name = "zhang";
    zhang->age = 12;
    std::cout << "指针： 2" << zhang->name << " " << zhang->age << std::endl;

    // 3. 结构体的拷贝
    Person li = {"li", 20};
    std::cout <<" " << li.name << " " << li.age << std::endl;
    Person li_copy;
    li_copy = li;
    std::cout <<" copy " << li_copy.name << " " << li_copy.age << std::endl;

    // 如过li发生变化, copy_li不会变化
    li.age = 21;
    std::cout <<" copy after change" << li_copy.name << " " << li_copy.age << std::endl;

}

int main(){

    useStruct();
}
