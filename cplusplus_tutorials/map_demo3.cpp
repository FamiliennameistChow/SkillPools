/**************************************************************************
 * map_demo3.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.04.23
 * 
 * @Description:
 * 结构体指针作为Key
 ***************************************************************************/
#include <iostream>
#include <map>

using namespace std;

typedef struct Node{
    int x, y;
    float score;

    Node(int _x, int _y, float _score) : x(_x), y(_y), score(_score){}

    bool operator == (const Node &n) const
    {
        return this->x == n.x && this->y == n.y;
    }

}Node;

struct comp {
    bool operator()(const Node *p1, const Node *p2) const {
        return !(p1->x == p2->x && p1->y == p2->y);
    }
};


int main(){

    std::map<Node*, int, comp> NodeSet;
    Node *n1 = new Node(1, 1, 11.2);
    Node *n2 = new Node(2, 2, 22.1);

    NodeSet.insert(make_pair(n1, 1));
    NodeSet.insert(make_pair(n2, 2));

    std::cout << "----------------" << std::endl;
    for (auto n: NodeSet) {
        std::cout << "{" <<n.first->x << "," << n.first->y <<"," << n.first->score << "}"<<std::endl;
    }

    Node *n3 = new Node(2, 2, 22.1);
    auto p = NodeSet.find(n3);
    if (p != NodeSet.end()){
        std::cout << "find: {" << p->first->x << ","<< p->first->y << "," << p->first->score << "}" << std::endl;
    } else{
        std::cout << " NO FIND" << std::endl;
    }

    // 说明find并不能找到对应的指针


}
