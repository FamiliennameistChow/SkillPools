/**************************************************************************
 * read_txt.cpp
 *
 * @Author： bornchow
 * @Date: 2021.10.19
 *
 * @Description:
 *  本程序演示 c++ 读写txt文件
 *  fstream 提供了三个类，用来实现c++对文件的操作
 *  ifstream ：从已有的文件读入
 *  ofstream ： 向文件写内容
 *  fstream ： 打开文件供读写
 *
    ios::in             只读
    ios::out            只写
    ios::app            从文件末尾开始写，防止丢失文件中原来就有的内容
    ios::binary         二进制模式
    ios::nocreate       打开一个文件时，如果文件不存在，不创建文件
    ios::noreplace      打开一个文件时，如果文件不存在，创建该文件
    ios::trunc          打开一个文件，然后清空内容
    ios::ate            打开一个文件时，将位置移动到文件尾
 ***************************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;



int main() {
    std::cout << "this is a txt file read and write demo " << std::endl;

    //////////////////////////////////////////////////
    ////                   读取txt文件             ////
    //////////////////////////////////////////////////
    string fileName = "/Users/zhoubo/Documents/SkillPools/cplusplus_tutorials/data/demo.txt";
    // 打开一个文件
    // ifstream read_file(fileName);
    ifstream read_file;
    read_file.open(fileName.data());

    if (!read_file.is_open()){
        std::cout << "can not open file: " << fileName << std::endl;
    }
    // or assert(read_file.is_open());

    // 按行读取文件信息
    std::cout << "--- read by line -----" << std::endl;
    string s;
    while(getline(read_file,s))
    {
        cout<<s<<endl;
    }

    // 逐个字符读取，忽略空格与回车
//    std::cout << "--- read by char -----" << std::endl;
//    char c;
//    while (!read_file.eof())
//    {
//        read_file >> c;
//        std::cout<< c <<std::endl;
//    }

    // 逐个字符读取，不空格与回车
//    std::cout << "--- read by char -----" << std::endl;
//    char c1;
//    read_file >> noskipws;
//    while (!read_file.eof())
//    {
//        read_file >> c1;
//        std::cout<< c1 <<std::endl;
//    }

    read_file.close();

    //////////////////////////////////////////////////
    ////                   写txt文件             ////
    //////////////////////////////////////////////////
    std::vector<int> v1 = {1,2,3,4,5,6,7};
    string fileName2 = "/Users/zhoubo/Documents/SkillPools/cplusplus_tutorials/data/demo2.txt";

    ofstream outfile(fileName2);

    for(auto v: v1){
        outfile << v << " ";
    }
    outfile << std::endl;

    return 0;
}
