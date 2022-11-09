
/**************************************************************************
 * Astar.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.10.06
 * 
 * @Description:
 *  
 ***************************************************************************/
//
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <queue>
#include <math.h>


using namespace std;

// f(n) = g(n) + h(n)
typedef struct Node
{
    int x;
    int y;

    float g;  // cost so far
    float h;  // cost to go
    float f;  // total cost
    int t; // 该节点的执行时间

    Node* fatherNode;
    Node() = default;
    Node(int x, int y){
        this->x = x;
        this->y = y;
        this->g = 0;
        this->h = 0;
        this->f = 0;
        this->t = 1;
        this->fatherNode = nullptr;
    }
    Node(int x, int y, Node* father){
        this->x = x;
        this->y = y;
        this->g = 0;
        this->h = 0;
        this->f = 0;
        this->t = 1;
        this->fatherNode = father;
    }

    // // 重载 <
    // bool operator < (const Node &n) const {
    //     if (f != n.f)
    //     {
    //         return f < n.f;
    //     }else
    //     {
    //         return h < n.h;
    //     }

    // }

    // 重载输出函数
    friend ostream& operator << (ostream& ostr, Node &n){
        if (n.fatherNode == nullptr)
        {
            cout << "[ " << n.x << " , " << n.y << " ] -- " << "GHF: " << n.g << " " << n.h << " " << n.f << " " << n.t
                 << " -- {  NULL  } " << std::endl;
        }else
        {
            cout << "[ " << n.x << " , " << n.y << " ] -- " << "GHF: " << n.g << " " << n.h << " " << n.f << " " << n.t
                 << " -- { " << (n.fatherNode)->x<< " , " << (n.fatherNode)->y << " } ";
        }
        return cout;
    }

}Node;

struct Compare
{
//    bool operator()(const Node *a,const Node *b)const
//    {
//        return a->f > b->f;
//    }
    bool operator()(const Node *a,const Node *b)const
    {
        if(a->f == b->f){
            return a->h > b->h;
        }else{
            return a->f > b->f;
        }
    }
};

typedef priority_queue<Node*, vector<Node*> ,Compare > PQ;



void txt2map(string fileName, vector<vector<int>>& mapData, vector<vector<Node*>>& startGoalPair, int &robotNums){
    ifstream txtFile;
    txtFile.open(fileName.data());

    if (!txtFile.is_open()){
        std::cout << "can not open file: " << fileName << std::endl;
    }

    string str;

    while(getline(txtFile, str)){
        cout << str << endl;
        istringstream iss(str);
        string token;
        vector<int> thisLine;
        while(getline(iss, token, ' ')){
            thisLine.push_back(stoi(token));
        }
        mapData.push_back(thisLine);
    }

    cout << "map size: " << mapData.size() << " " << mapData[0].size() << std::endl;

    for(int row=0; row<mapData.size(); row++){
        for(int col=0; col<mapData[0].size(); col++){
            if(mapData[row][col] == 0 || mapData[row][col] == 1) continue;

            int n = mapData[row][col];
            cout << n % 10 - 1 << " " << n / 10 -1<< " { " << row << " " << col << " } " << endl;
            Node* thisNode = new Node(col, row);
            startGoalPair[n%10-1][n/10-1] = thisNode;
            mapData[row][col] = 0;

            if(n%10 > robotNums) robotNums = n % 10;
        }
    }

//     ====== cout info
    cout << " robot nums: " << robotNums << endl;
    cout << " ========== map ==========" << endl;
    for(int row=0; row<mapData.size(); row++){
        for(int col=0; col<mapData[0].size(); col++){
            cout << mapData[row][col] << " ";
        }
        cout << endl;
    }

    cout << " ========== SG pair ==========" << endl;
    for(int i=0; i<robotNums; i++){
        cout << " robot num: " << i+1 << " star point:  " << *startGoalPair[i][0]  <<
        "goal point:  " << *startGoalPair[i][1] << endl;
    }

    txtFile.close();
}

bool setsEmpty(vector<PQ> sets){
    bool res = true;
    for(auto set : sets){
        res = res && set.empty();
    }

    return res;
}

bool searchOver(vector<bool> res){
    bool r = true;
    for(auto rs : res) r = r && rs;

    return r;
}
//vector<int> Dir = {0, 1, 0, -1, 1, 1, -1, -1, 0};
vector<int> Dir = {0, 1, 0, -1, 0};
void searchNei(Node* n, vector<Node*>& neis, vector<vector<int>> map){
    int x, y;
    for(int i=1; i<Dir.size(); i++){
        x = n->x + Dir[i-1];
        y = n->y + Dir[i];
        if(x >=0 && x < map[0].size() && y>=0 && y< map.size()){
            Node* neiNode = new Node(x, y, n);
            neis.push_back(neiNode);
        }
    }
}

float callDis(Node* n1, Node* n2){
    return sqrt(pow(n1->x - n2->x, 2) + pow(n1->y - n2->y, 2));
}

void callCost(Node* curNode, Node* neiNode, vector<vector<Node*>> SG, int rInd){
    neiNode->g = curNode->g + callDis(curNode, neiNode);

    // 启发函数使用 汉明距离
    neiNode->h = abs(neiNode->x - SG[rInd-2][1]->x) + abs(neiNode->y - SG[rInd-2][1]->y);

    neiNode->f = neiNode->g + neiNode->h;
}

bool inSet(Node* node, vector<Node*> set){
    for(auto n : set){
        if(n->x == node->x && n->y == node->y) return true;
    }
    return false;
}

void inOpenSetProcess(Node* node, PQ & openSet){
    PQ backSet;
    bool inSet = false;
    while(!openSet.empty()){
        Node* openNode = openSet.top();
        openSet.pop();
        if(node->x == openNode->x && node->y == openNode->y){ // in openSet
            if(node->f < openNode->f){ // 当前节点代价更优
                backSet.push(node);
            }else{
                backSet.push(openNode);
            }
            inSet = true;
        }else{
            backSet.push(openNode);
        }
    }

    if(!inSet) backSet.push(node);

    swap(openSet, backSet);
}

void backSet(vector<Node*>& path, Node* curN){
    int n = curN->t;
    while(n--){
        cout << "curNode " << *curN << endl;
        path.push_back(curN);
    }

    if(curN->fatherNode == nullptr) return;

    backSet(path, curN->fatherNode);
}

vector<vector<int>> COLORS = {{158, 168, 3, 255, 0, 0},
                              {35, 142, 107, 0, 255, 0},
                              {80, 127, 155, 0, 0, 255},
                              {132, 227, 255, 0,255,255},
                              {203, 192, 255, 255,0,255},
                              {192, 192, 192, 120, 0, 120},
                              {216, 235, 202, 8, 64, 84},
                              {33, 145, 237,0, 200, 140},
                              {255, 255, 0, 84, 46, 8},
                              {0, 215, 220, 88, 87, 86}};


void showRes(vector<vector<int>> map, vector<vector<Node*>> timePath){

    cv::Mat maxtrix(map.size(), map[0].size(), CV_8UC3, cv::Scalar(255, 255, 255));

    for(int row=0; row<map.size(); row++){
        for(int col=0; col<map[0].size(); col++){
            if(map[row][col] == 1) {
                maxtrix.at<cv::Vec3b>(row, col)[0] = 0;
                maxtrix.at<cv::Vec3b>(row, col)[1] = 0;
                maxtrix.at<cv::Vec3b>(row, col)[2] = 0;
            }
        }
    }
    int height = 200;
    int width = height * 1.0 / map.size() * map[0].size();

    int pathSize = 0;
    for(auto & r : timePath){
        if(r.size() > pathSize){
            pathSize = r.size();
        }
    }

    for(int i=0; i<pathSize; i++){
        for(int r=0; r<timePath.size(); r++){
            if(i < timePath[r].size()){
                Node* n = timePath[r][i];

                cv::rectangle(maxtrix, cv::Rect(n->x, n->y, 1, 1),
                              cv::Scalar(COLORS[r][0], COLORS[r][1], COLORS[r][2]),
                              -1, 1,0);
            }
        }
        cv::Mat showImg;
        cv::resize(maxtrix, showImg, cv::Size(width, height), 0, 0, cv::INTER_AREA);
        cv::copyMakeBorder(showImg, showImg, 5, 5, 5, 5,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0,0));
        cv::imshow("mat_"+ to_string(i), showImg);
        cv::waitKey();
    }

    vector<vector<int>> lastPoint(timePath.size(), vector<int>(2, 0));

    cv::Mat showImg;
    cv::resize(maxtrix, showImg, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    float s_x = height*1.0 / map.size();
    float s_y = width*1.0 / map[0].size();
    for(int r=0; r<timePath.size(); r++){
        std::cout << "robot " << r <<" ";
        for(int i=0; i<timePath[r].size(); i++){
            if(i == 0){
                lastPoint[r][0] = (timePath[r][i]->x+0.5)*s_x;
                lastPoint[r][1] = (timePath[r][i]->y+0.5)*s_y;
            }
            cv::line(showImg,
                     cv::Point(lastPoint[r][0], lastPoint[r][1]),
                     cv::Point((timePath[r][i]->x + 0.5)*s_x, (timePath[r][i]->y + 0.5)*s_y),
                     cv::Scalar(COLORS[r][3], COLORS[r][4], COLORS[r][5]),
                     1, 1, 0);
            lastPoint[r][0] = (timePath[r][i]->x+0.5)*s_x;
            lastPoint[r][1] = (timePath[r][i]->y+0.5)*s_y;
            cout <<"( " << timePath[r][i]->x << " , " << timePath[r][i]->y << " ) ";
        }
        std::cout << std::endl;
    }

    cv::copyMakeBorder(showImg, showImg, 5, 5, 5, 5,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0,0));
    cv::imshow("mat_path", showImg);
    cv::waitKey();

}

void printSet(PQ openSet){
    while(!openSet.empty()){
        cout << *openSet.top() << endl;
        openSet.pop();
    }
}

int main(){
    cout << "======>>> Built with OpenCV " << CV_VERSION << endl;
    cout << "======>>> this is multi a-star" << endl;
    vector<vector<int>> mapData;
    string filePath = "/Users/zhoubo/Documents/SkillPools/cplusplus_tutorials/data/map3.txt";
    vector<vector<Node*>> SGpair(10, vector<Node*>(2, new Node())); // 储存各个机器人起点与终点对
    int robotNums = 0;

    txt2map(filePath, mapData, SGpair, robotNums);
    cout << "=========================" << endl;

    // A star search
    vector<PQ> openSets;
    vector<vector<Node*>> closeSets(robotNums, vector<Node*>());

    for(int i=0; i<robotNums; i++){
        PQ openSet;
        openSet.push(SGpair[i][0]);

        mapData[SGpair[i][0]->y][SGpair[i][0]->x] = i+2; // 标记机器人当前位置 标号2表示第一个机器人
        openSets.push_back(openSet);
    }

    vector<vector<Node*>> resPaths(robotNums, vector<Node*>());
    vector<bool> searchRes(robotNums, false);


    int index = 0;


    // 循环
    while(!setsEmpty(openSets)){

        if(searchOver(searchRes)){
            break;
        }

        int robotInd;
        index++;
        cout << " ============= proccess index " << index << endl;
        for(int ind=0; ind<robotNums; ind++){
            robotInd = ind+2;
            cout << " ============= proccess robot " << robotInd << endl;

            if(searchRes[ind]){
                cout << " ==== search over ==== " << endl;
                continue;
            }

            // 弹出最优节点
            Node* curNode = openSets[ind].top();
            openSets[ind].pop();

            // 判断当前节点是否存在其他机器人
            if(mapData[curNode->y][curNode->x] > 1 && mapData[curNode->y][curNode->x] != robotInd){ // other robot there;
                cout << " curNode has other robot....." << endl;
                Node* tempNode = openSets[ind].top(); // 弹出次优节点
                if(tempNode->f <= curNode->f + 1){
                    curNode = tempNode; // 次优节点 比原地等待更优, 当前节点换为次优节点
                    openSets[ind].pop();
                    if(mapData[curNode->y][curNode->x] > 1 && mapData[curNode->y][curNode->x] != robotInd){ // 次优节点有其他机器人
                        closeSets[ind].back()->t += 1; // 当前机器人原地等待
                        openSets[ind].push(curNode);
                        continue;
                    }
                }else {
                    cout << " wait in this pos....." << endl;
                    closeSets[ind].back()->t += 1; // 当前机器人原地等待
                    openSets[ind].push(curNode);
                    continue;
                }
            }

            // 不存在其他机器人，go on...

            // 在地图上更新机器人位置
            if(!closeSets[ind].empty()){
                mapData[closeSets[ind].back()->y][closeSets[ind].back()->x] = 0;
                mapData[curNode->y][curNode->x] = robotInd;
            }

            cout << " ========== map ==========" << endl;
            for(int row=0; row<mapData.size(); row++){
                for(int col=0; col<mapData[0].size(); col++){
                    cout << mapData[row][col] << " ";
                }
                cout << endl;
            }
            cout << " ========== map ==========" << endl;

            // 如果当前位置为目标位置，搜索结束
            if(curNode->x == SGpair[ind][1]->x && curNode->y == SGpair[ind][1]->y){
                cout << " robot " << robotInd << "  search over !!! "<< resPaths[ind].size() <<endl;

                backSet(resPaths[ind], curNode);
                reverse(resPaths[ind].begin(), resPaths[ind].end());
                searchRes[ind] = true;

                continue;
            }

            // 搜索邻域
            vector<Node*> Neis;
            searchNei(curNode, Neis, mapData);
            cout << " neis size: " << Neis.size() << endl;
            //遍历邻近节点
            for(size_t i=0; i<Neis.size(); i++){
                Node* neiNode = Neis[i];
                callCost(curNode, neiNode, SGpair, robotInd);

                cout << " robot " << robotInd << " ==== neis: " << i << *neiNode << endl;
                // 1.节点是否被占据
                if(mapData[neiNode->y][neiNode->x] == 1) continue;

                // 2.节点是否在closeSet中
                if(inSet(neiNode, closeSets[robotInd-2])) continue;

                // 3.节点是否在openSet中
                inOpenSetProcess(neiNode, openSets[robotInd-2]);

            }
            cout << " openSet size " << openSets[ind].size() << endl;
            cout << "====== openSet ======= for robotInd " << robotInd << endl;
            printSet(openSets[ind]);
            cout << "====== openSet ======= " << endl;

            closeSets[ind].push_back(curNode);

        }

    }

    cout << " ALL SEARCH OVER !!!!!" << endl;

    showRes(mapData, resPaths);

    return 0;
}

