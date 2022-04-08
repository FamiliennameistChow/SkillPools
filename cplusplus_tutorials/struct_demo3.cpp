/**************************************************************************
 * struct_demo3.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.03.25
 * 
 * @Description:
 * 结构体的自引用
 * 以A-star算法中定义 节点为例， 节点数据类型需要储存节点位置信息(x,y), 节点代价信息(f, g, h), 节点的父节点(这也是一个节点数据类型)
 ***************************************************************************/
#include <iostream>
using namespace std;

typedef struct Node{
    int x;
    int y;
    float f;
    float h;
    float g;
    Node * father_node;
    Node(int x, int y){
        this->x = x;
        this->y = y;
        this->g = 0;
        this->h = 0;
        this->f = 0;
        this->father_node = NULL;
    }

    Node(){}

    Node(int x_, int y_, Node *father ): x(x_), y(y_), g(0.0), f(0.0), h(0.0), father_node(father){ }

    //重载输入
    friend istream &operator >>(istream&, Node &n){
        cin >> n.x >> n.y;
        // 顺便初始化其他值
        n.h = 0;
        n.g = 0;
        n.f = 0;
        n.father_node = NULL;
        return cin;
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
};

int main(){
    // 声明
    Node start_node(0, 0);
    std::cout <<start_node << std::endl;

    Node new_node(1, 1, &start_node);
    std::cout << "new " << new_node << std::endl;

    //如果star_node改变，new_node也会改变
    start_node.x = 2;
    start_node.y = 2;
    std::cout << "new  after change: " << new_node << std::endl;



    //使用指针声明
    Node *father_node = &start_node;
    std::cout << "指针 ：" << *father_node << std::endl;

    Node *this_node = new Node(5, 5, father_node);
    std::cout << "指针 ：" << *this_node << std::endl;

    //如果father node指向 的值改变, this_node 也会改变
    start_node.x = 8;
    start_node.y = 8;
    std::cout << "指针 after change ：" << *this_node << std::endl;

    //
    Node* father;
    father->x = start_node.x;
    father->y = start_node.y;
    father->father_node = start_node.father_node;
    Node new_node2(5, 5, father);
    std::cout << " " << new_node2.x << " , " << new_node2.y << std::endl;

    start_node.x = 10;
    start_node.y = 10;
    std::cout << "  after change: " << new_node2.x << " , " << new_node2.y << std::endl;










}

