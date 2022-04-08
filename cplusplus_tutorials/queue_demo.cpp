/**************************************************************************
 * queue_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.03.26
 * 
 * @Description:
 * 数列的使用
 ***************************************************************************/
#include <iostream>
#include <queue>
using namespace std;

typedef struct Node{
    int x;
    int y;
    Node(int _x, int _y) : x(_x), y(_y) {}
};

int main(){
    std::queue<int> numbers;

    //1.添加元素
    numbers.push(1);
    numbers.push(2);
    numbers.push(3);
    numbers.push(4);

    //2.获取第一个元素
    std::cout <<" queue number: " << numbers.front() << std::endl;

    //3.获取最后一个元素
    std::cout <<" queue number back : " << numbers.back() << std::endl;

    //4.队列大小
    std::cout <<" queue size : " << numbers.size() << std::endl;

    //4.删除第一个元素
    numbers.pop();
    std::cout <<" queue size after erase : " << numbers.size() << std::endl;

    //5.返回队列是否为空
    std::cout << " numbers if empty :" << numbers.empty() << std::endl;
    std::queue<int> nus;
    std::cout << " nus if empty : " << nus.empty() << std::endl;

    // 6.交换两个队列
    // swap()函数用于交换两个队列的内容，但是队列的类型必须相同，尽管大小可能有所不同
    std::cout << " numbers size: " << numbers.size() << " nus size: " << nus.size() << std::endl;
    numbers.swap(nus);
    std::cout << " after swap --> numbers size: " << numbers.size() << " nus size: " << nus.size() << std::endl;

    //7.遍历队列
    // queue 也没有迭代器。访问元素的唯一方式是遍历容器内容，并移除访问过的每一个元素。
    while (!nus.empty()){
        std::cout << nus.front() << std::endl;
        nus.pop();
    }

    //8.emplace 可以直接传入构造对象需要的元素， 然后自己调用其构造函数
    std::queue<Node> Nodes;
    Node this_node(1, 1);
    Node this_node2(2, 2);
    Nodes.push(this_node);
    Nodes.push(this_node2);
    Nodes.emplace(3, 3);
    std::cout <<"Nodes size: " << Nodes.size() << std::endl;


}
