
/**************************************************************************
 * yaml_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.05.25
 * 
 * @Description:
 *  
 ***************************************************************************/
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <fstream>

using namespace std;

typedef struct MissionPoint{
    int id;
    float x;
    float y;
    bool isStop;

    //重载输出
    friend ostream  &operator << (ostream&, MissionPoint &p){
        cout << "{ id: " << p.id << " x: " << p.x << " y: " << p.y << " is stop: " << p.isStop << " }" << std::endl;
        return cout;
    }

}MissionPoint;

int main(){
    YAML::Node root_node;
    string yamlFile = "/Users/zhoubo/Documents/SkillPools/cplusplus_tutorials/data/yaml_test.yaml";
    try
    {
        root_node = YAML::LoadFile(yamlFile);
    }
    catch(const YAML::Exception& e)
    {
        // TODO:处理异常
        std::cout<< " NO such file " <<std::endl;
        return -1;
    }

//    struct NodeType {
//        enum value { Undefined, Null, Scalar, Sequence, Map };
//    };
    cout << "Node type " << root_node.Type() << endl;
    YAML::Node mission_list = root_node["misson_point"];
    std::vector<MissionPoint> missionPoints;
    if(mission_list.IsSequence()){
        for(auto && item : mission_list){
            MissionPoint p;
            p.id = item["id"].as<int>();
            p.x = item["point"][0].as<float>();
            p.y = item["point"][1].as<float>();
            p.isStop = item["is_stop"].as<bool>();
            missionPoints.push_back(p);
        }
    }else{
        std::cout << " No map" << std::endl;
    }

    for(auto mp: missionPoints){
        std::cout << mp << std::endl;
    }

    // assert(node.IsNull())

    // 写入yaml
    MissionPoint new_point;
    new_point.id = 3;
    new_point.x = 2;
    new_point.y = 2;
    new_point.isStop = false;

    YAML::Node newNode;
    newNode["id"] = new_point.id;
    newNode["point"].push_back(new_point.x);
    newNode["point"].push_back(new_point.y);
    newNode["is_stop"] = new_point.isStop;

    mission_list.push_back(newNode);

    std::cout << root_node << std::endl;

    ofstream fout(yamlFile);
    fout << root_node;
    fout.close();








}

