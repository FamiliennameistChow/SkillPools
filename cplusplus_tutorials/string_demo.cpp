
/**************************************************************************
 * string_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.05.09
 * 
 * @Description:
 * 字符串string的使用
 ***************************************************************************/
#include <iostream>
#include <string>
using namespace std;

void init_string(){
    string s1;
    string s2 (3, 'a');
    string s3 ("value");
    string s4 (s3);
    string s5 = "hello world"; // 使用等号初始化
    string s6 (s5, 6); // s5切片  6 - end
    string s7 (s5, 0, 5); // s5 切片 0-5
    std::cout << " s2: " << s2 << std::endl;
    std::cout << " s3: " << s3 << std::endl;
    std::cout << " s4: " << s4 << std::endl;
    std::cout << " s5: " << s5 << std::endl;
    std::cout << " s6: " << s6 << std::endl;
    std::cout << " s7: " << s7 << std::endl;


}

void other_function(){
    string s ("values");
    // 字符串的长度
    std::cout <<" string size: " <<s.length() << " " <<s.size() << std::endl;
}

void cross_string(){
    string str ("values");

    std::cout << " { ";
    for(int i = 0; i < str.size(); i++){
        std::cout << str[i];
    }
    std::cout << " }"<<std::endl;

    std::cout << "-----" << std::endl;

    std::cout << " { ";
    for(auto s : str){
        std::cout << s;
    }
    std::cout << " }"<<std::endl;

}

void insert_element(){
    string str = "hello world";

    //s.insert(pos,n,ch) 在字符串s的pos位置上面插入n个字符ch
    str.insert(6, 4, 'z');
    std::cout << str << std::endl;

    //s.insert(pos,str) 在字符串s的pos位置插入字符串str
    string str1 = "hello world";
    string str2 = "hard ";
    str1.insert(6, str2);
    std::cout << str1 << std::endl;

    //s.insert(pos,str,a,n) 在字符串s的pos位置插入字符串str中位置a到后面的n个字符
    string str3 = "hello world";
    string str4 = "it is so happy wow";

    str3.insert(6, str4, 6, 9);
    std::cout << str3 << std::endl;

    //s.insert(pos,cstr,n)      在字符串s的pos位置插入字符数组cstr从开始到后面的n个字符
    string str5 = "hello world";
    str5.insert(6,"it is so happy wow",6);
    std::cout << str5 << std::endl;
}


void find_char(){
    // find函数
    string str = "The apple thinks apple is delicious";     //长度34
    string key = "apple";

    // s.find(str) 查找字符串str在当前字符串s中第一次出现的位置
    int pos = str.find(key);
    std::cout <<key << " is at " << pos << std::endl;

    //s.find(str,pos)        查找字符串str在当前字符串s的[pos,end]中第一次出现的位置
    int pos2 = str.find(key, 10);
    std::cout <<key << " outer 10 is at " << pos2 << std::endl;

    //s.find(cstr,pos,n)     查找字符数组cstr前n的字符在当前字符串s的[pos,end]中第一次出现的位置
    //此处不可将"delete"替换为str2（如果定义str2 = "delete"）
    int pos3 = str.find("delete", 0, 2);       // 26
    std::cout << pos3 << std::endl;

    //s.find(ch,pos)         查找字符ch在当前字符串s的[pos,end]中第一次出现的位置
    int pos4 = str.find('s', 0);               // 15
    std::cout << "s is at " << pos4 << std::endl;

    int pos5 = str.find('b', 0);
    std::cout << " b not in str: " << pos5 << std::endl;

}


void compare_char(){
    string s1 = "hello wolrd";
    string s2 = "e";
    std::cout << s1.compare(1,1,s2,0,1) << std::endl;

    string str1 = "small leaf";
    string str2 = "big leaf";
    std::cout << str1.compare(6,1,str2,4,1) << std::endl;
}

int main(){
    std::cout << " this is string demo ...." << std::endl;

//    init_string();

//    other_function();

//    cross_string();
//    insert_element();

//    find_char();

    compare_char();

}

