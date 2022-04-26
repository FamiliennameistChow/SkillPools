/**************************************************************************
 * map_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.04.17
 * 
 * @Description:
 * 本程序示例map的使用
 ***************************************************************************/

#include <iostream>
#include <map>

using namespace std;

void init_map(){
    std::cout << " this is init of map " << std::endl;
    std::map<int, string> person;

    std::map<int, string> stu{{10, "lili"}, {12, "XiaoMing"}};

    std::map<float, string> players {std::make_pair(30.2, "zhang"), std::make_pair(20.1, "li")};
}

void insert_elements(){
    std::cout << " -------------- insert elements----------" << std::endl;

    std::map<int, string> Stus;

    // pair<iterator,bool> insert() 插入pair
    // 插入成功
    auto p = Stus.insert(pair<int, string>(19, "Jerk"));

    std::cout << " { ";
    for(auto stu : Stus){
        std::cout << "{ "<< stu.first << " " << stu.second << " },";
    }
    std::cout << " } " << std::endl;

    if (p.second){
        std::cout << " insert successful !" << std::endl;
        std::cout << " the insert pair is: {" << p.first->first << "," << p.first->second << "}" << std::endl;
    } else{
        std::cout << " insert unsuccessful !" << std::endl;
    }

    // key值一样，插入不成功
    auto p1 = Stus.insert(pair<int, string>(19, "zhang"));
    std::cout << " is insert? " << p1.second << std::endl;
    std::cout << " the key in map already is : " <<  p1.first->first << ", " << p1.first->second << std::endl;

    //用insert函数插入 make_pair
    Stus.insert(make_pair(15, "Bo"));

    std::cout << " { ";
    for(auto stu : Stus){
        std::cout << "{ "<< stu.first << " " << stu.second << " },";
    }
    std::cout << " } " << std::endl;

    //用insert函数插入value_type数据
    Stus.insert(map<int, string>::value_type(20, "Mehdi"));

    std::cout << " { ";
    for(auto stu : Stus){
        std::cout << "{ "<< stu.first << " " << stu.second << " },";
    }
    std::cout << " } " << std::endl;

    //使用数组
    Stus[14] = "Born ";

    std::cout << " { ";
    for(auto stu : Stus){
        std::cout << "{ "<< stu.first << " " << stu.second << " },";
    }
    std::cout << " } " << std::endl;

}

void get_elements(){
    std::cout << " ----------- get elements --------------- " << std::endl;

    std::map<string, int> Person {{"li", 19}, {"mehdi", 20}};

    // 通过 Key值访问
    std::cout <<" [ ] method:  " << Person["li"] << std::endl;

    // begin() 返回指向 map 头部的迭代器
    std::map<string,int>::iterator it = Person.begin();
    std::cout << it->first << " is " << it->second << std::endl;

    // end() 返回指向 map 末尾的迭代器 注意尾部迭代器不代表最后一个元素
    std::map<string, int>::iterator it2 = Person.end();
    std::cout << it2->first << " " << it2->second << std::endl;

    // rbegin() 返回最后一个元素的迭代器
    std::map<string, int>::reverse_iterator it3 = Person.rbegin();
    std::cout << it3->first << "  " << it3->second << std::endl;

    // rend() 返回尾部迭代器 error
    // std::map<string, int>::reverse_iterator it4 = Person.rend();
    // std::cout << it4->first << " " << it4->second << std::endl;

}

void cross_elements(){
    std::cout << "----------- cross_elements ------------" << std::endl;
    
    std::map<float, string> players {{12.1, "li"}, {13.4, "zhang"}};
    
    // 使用auto
    for (auto & player : players) {
        std::cout<< player.first << " " << player.second << std::endl;
    }

    // 使用迭代器
    for(std::map<float, string>::iterator it=players.begin(); it != players.end(); it++){
        std::cout << it->first << " " << it->second <<std::endl;
    }

}

void other_func(){

    std::map<int, string> stu{{10, "lili"}, {12, "XiaoMing"}, {14, "zhang"}};

    std::cout << " { ";
    for(auto s : stu){
        std::cout << "{ "<< s.first << " " << s.second << " },";
    }
    std::cout << " } " << std::endl;

    // iterator find() 查找元素
    auto it = stu.find(10);
    if (it == stu.end()){
        std::cout << " no key is [10] in the map " <<std::endl;
    } else{
        std::cout << " key " << it->first << " is : " << it->second << std::endl;
    }

    // int size()返回map容器中的个数
    std::cout << " the size of map is: " << stu.size() << std::endl;

    // bool empty() 返回容器是否为空
    std::cout << " is the map empty ? " << stu.empty() << std::endl;

    // int count() 返回指定Key元素出现的次数, 由于map中key不能重复，所以只能是0或1
    std::cout << " key == 12 : " << stu.count(12) << std::endl;
    std::cout << " key == 13 : " << stu.count(13) << std::endl;

    //iterator lower_bound() 返回键值>=给定元素的第一个位置
    auto it_lower = stu.lower_bound(12);
    std::cout <<" lower 12: " << it_lower->first << " " << it_lower->second << std::endl;

    //iterator upper_bound() 返回键值>给定元素的第一个位置
    auto it_upper = stu.upper_bound(12);
    std::cout << " upper 12: " << it_upper->first << " " << it_upper->second << std::endl;

    // 修改元素
    stu[10] = "li";

    std::cout << " { ";
    for(auto s : stu){
        std::cout << "{ "<< s.first << " " << s.second << " },";
    }
    std::cout << " } " << std::endl;

    //  clear() 删除所有元素
    stu.clear();
    std::cout << " size of map :" << stu.size() << std::endl;


}


void erase_elements(){
    std::map<int, string> stu{{10, "lili"}, {12, "XiaoMing"}, {14, "zhang"}, {15, "ming"}};

    std::cout << " { ";
    for(auto s : stu){
        std::cout << "{ "<< s.first << " " << s.second << " },";
    }
    std::cout << " } " << std::endl;

    // 使用Key值删除
    stu.erase(12);

    std::cout << " { ";
    for(auto s : stu){
        std::cout << "{ "<< s.first << " " << s.second << " },";
    }
    std::cout << " } " << std::endl;

    // 使用迭代器删除元素
    auto it = stu.find(14);
    if (it != stu.end()){
        stu.erase(it);
    }

    std::cout << " { ";
    for(auto s : stu){
        std::cout << "{ "<< s.first << " " << s.second << " },";
    }
    std::cout << " } " << std::endl;

    // 删除所有元素
    stu.erase(stu.begin(), stu.end());
    std::cout << " empty: " << stu.empty() << std::endl;

}

// 重载operator()的类
struct myComp{
    bool operator()(const string &s1, const string &s2) const {
        return s1.length() < s2.length(); //按string 长度升序
    }
};

void sort_map(){

    // 默认升序排列
    std::map<int, string> Stu{{11, "li"}, {10, "zhang"}, {12, "wang"}};
    std::cout << " { ";
    for (auto s:Stu) {
        std::cout << "{" << s.first << "," << s.second << "} ";
    }
    std::cout << " } " << std::endl;

    // 降序排列
    std::map<int, string, greater<int> >Person{{11, "li"}, {10, "zhang"}, {12, "wang"}};

    std::cout << " { ";
    for (auto s:Person) {
        std::cout << "{" << s.first << "," << s.second << "} ";
    }
    std::cout << " } " << std::endl;

    //自定义排序函数
    // 按String长度升序排列，这也意味着String的长度不能相等
    std::map<string, int, myComp> Men{{"wang", 13}, {"zhang", 11}, {"li", 12}};

    std::cout << " { ";
    for (auto s:Men) {
        std::cout << "{" << s.first << "," << s.second << "} ";
    }
    std::cout << " } " << std::endl;
}

int main(){

    std::cout << " this is map demo " << std::endl;

    //init_map();

    //get_elements();
    
    //cross_elements();

    // insert_elements();

    //other_func();

    // erase_elements();


    sort_map();
    
}

