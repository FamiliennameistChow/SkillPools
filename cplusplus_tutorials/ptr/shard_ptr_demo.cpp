
/**************************************************************************
 * shard_ptr_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.06.28
 * 
 * @Description:
 *  这里是智能指针shrad_ptr的使用方法
 ***************************************************************************/
#include <iostream>

struct Node{
    int x;
    int y;
    Node(int x_, int y_):x(x_), y(y_){}
    Node(){}
};

int main(){

    // 初始化
    // 1. 使用构造函数初始化
    std::shared_ptr<Node> nodePtr1 (new Node(10, 10));
    std::cout << nodePtr1->x << " " << nodePtr1->y << std::endl;

    // 2. 拷贝或移动 构造函数初始化
    std::shared_ptr<Node> nodePtr2 = nodePtr1;
    std::cout << nodePtr2->x << " " << nodePtr2->y << std::endl;

    // 拷贝初始化，引用会变
    nodePtr1->x = 20;
    std::cout <<"node1: " << nodePtr1->x << " " << nodePtr1->y << std::endl;
    std::cout <<"node2: " << nodePtr2->x << " " << nodePtr2->y << std::endl;

    std::shared_ptr<Node> nodePtr3(std::move(nodePtr1));
    std::cout <<"node3: " << nodePtr3->x << " " << nodePtr3->y << std::endl;

    // nodePtr1->x = 30; 运行报错
    // 使用std::move nodePtr1变 nodePtr3不变
    std::cout <<"node3: " << nodePtr3->x << " " << nodePtr3->y << std::endl;

    std::shared_ptr<Node> nodePtr4 = std::move(nodePtr1);

    // 3. 使用std::make_shard<>初始化
    std::shared_ptr<Node> nodePtr5 = std::make_shared<Node>(50, 50);
    std::cout <<"node5: " << nodePtr5->x << " " << nodePtr5->y << std::endl;

    // 4. 使用reset函数
    std::shared_ptr<Node> nodePtr6;
    nodePtr6.reset(new Node(60, 60));
    std::cout <<"node6: " << nodePtr6->x << " " << nodePtr6->y << std::endl;

}
