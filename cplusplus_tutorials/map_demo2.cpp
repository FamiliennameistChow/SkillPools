/**************************************************************************
 * map_demo2.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.04.18
 * 
 * @Description:
 *  自定义结构体作为K值
 *  需要提供< 运算
 ***************************************************************************/
#include <iostream>
#include <map>

using namespace std;

typedef struct Node{
    int x, y;
    float score;

    Node(int _x, int _y, float _score) : x(_x), y(_y), score(_score){ }

//    //重载 < 运算
//    bool operator <(const Node &n) const{
//        if (this->x == n.x && this->y == n.y){
//            return false;
//
//        }else{
//            return this->score < n.score;
//        }
//        // return this->score < n.score;
//    }

}Node;

bool Mycomp(const Node &n1, const Node &n2){
    if (n1.x == n2.x && n1.y == n2.y){
        return false;
    } else{
        return n1.score < n2.score;
    }
}

// 重载operator()的类
struct MyCompare{
    bool operator()(const Node &n1, const Node &n2) const{
        if (n1.x == n2.x && n1.y == n2.y){
            return false;
        } else{
            return n1.score < n2.score;
        }
    }
};


// less函数的模板定制
template <>
    struct less<Node>{
    public:
        bool operator()(const Node &n1, const Node &n2) const{
            if (n1.x == n2.x && n1.y == n2.y){
                return false;
            } else{
                return n1.score < n2.score;
            }
        }
    };

int main(){

    std::map<Node, int> NodeSet;

    Node n1(1, 1, 23.5);
    Node n2(2,2, 20.5);

    NodeSet.insert(make_pair(n1, 1));

    auto p2 = NodeSet.insert(make_pair(n2, 2));

    std::cout << "----------------" << std::endl;
    for (auto n: NodeSet) {
        std::cout << "{" <<n.first.x << "," << n.first.y <<"," << n.first.score << "}"<<std::endl;
    }

    Node n3(3,2, 20.5);
    auto p3 = NodeSet.insert(make_pair(n3, 3));

    std::cout << "is insert:  " << p3.second << " { " << p3.first->first.x << " " << p3.first->first.y << " " <<
    p3.first->first.score << " }" << std::endl;

    std::cout << "----------------" << std::endl;
    for (auto n: NodeSet) {
        std::cout << "{" <<n.first.x << "," << n.first.y <<"," << n.first.score << "}"<<std::endl;
    }

    std::cout << NodeSet.size() <<std::endl;

    Node n4(2,2, 26.5);
    auto p4 = NodeSet.find(n4);
    if (p4 != NodeSet.end()){
        std::cout <<"{ "<< p4->first.x << ", " << p4->first.y << "," << p4->first.score << " }" << std::endl;
    } else{
        std::cout << " No find " << std::endl;
    }

    // 使用std::function
    std::map<Node, int, function<bool(const Node&, const Node&)>> NodeSet1(Mycomp);
    //map<Node, int, decltype(&Mycomp)> NodeSet1(Mycomp);
    NodeSet1.insert(make_pair(n1, 1));

    auto p12 = NodeSet1.insert(make_pair(n2,2));

    std::cout << "-------function---------" << std::endl;
    for (auto n: NodeSet1) {
        std::cout << "{" <<n.first.x << "," << n.first.y <<"," << n.first.score << "}"<<std::endl;
    }

    // 重载operator()的类
    std::map<Node, int, MyCompare> NodeSet2;

    NodeSet2.insert(make_pair(n1, 1));
    auto p22 = NodeSet2.insert(make_pair(n2, 2));

    std::cout << "-------重载operator()的类---------" << std::endl;
    for (auto n: NodeSet2) {
        std::cout << "{" <<n.first.x << "," << n.first.y <<"," << n.first.score << "}"<<std::endl;
    }

}

