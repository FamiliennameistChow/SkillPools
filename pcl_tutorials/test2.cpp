//
///**************************************************************************
// * test2.cpp
// *
// * @Author： bornchow
// * @Date: 2022.08.23
// *
// * @Description:
// *
// ***************************************************************************/
//#include <iostream>
//#include <sstream>
//#include <set>
//#include <vector>
//using namespace std;
//class Mycomp{
//public:
//    bool operator()(string s1, string s2){
//        int n1 = s1.size();
//        int n2 = s2.size();
//        int n = min(n1, n2);
//
//        //逐位比较字典序
//        for (int i = 0; i < n; i++)
//        {
//            char c1 = s1[i];
//            char c2 = s2[i];
//            if (c1 == c2)
//                continue;
//            else if (c1 < c2)
//                return true;   //s1字典序更小
//            else
//                return false;  //s2字典序更小,交换位置
//        }
//        //到这一步说明两个字符串前n位相同,比较字符串长度
//        return n1 < n2;
//    }
//};
//
//
//int main(){
//    int n;
//    set<string, Mycomp> words;
//    cin >> n;
//    cin.ignore();
//    while(n--){
//        string s;
//        getline(cin, s);
//        stringstream ss;
//        ss << s;
//        string temp;
//        while(ss >> temp){
//            words.insert(temp);
//        }
//    }
//
//    int nums = 0;
//    string s;
//    for(auto iter=words.begin(); iter != words.end(); ){
//        s += *iter;
//        nums += (*iter).size();
//        if(nums >= 50){
//            cout << s << endl;
//            nums = 0;
//            s.clear();
//            continue;
//        }
//        s += " ";
//        nums += 1;
//        iter++;
//    }
//
//    cout << s << endl;
//}


#include <vector>
#include <map>
#include <iostream>
#include <sstream>

using namespace std;

void getConv(vector<int> a, vector<int>b){
    int m = a.size();
    int n = b.size();
    int k = a.size() + b.size() - 1;
    vector<int> c(k, 0);
    cout << k << ",";
    for(int i = 0; i < k; i++) {
        for(int j = max(0, i + 1 - n); j <= min(i, m - 1); j++) {
            c[i] += a[j] * b[i - j];
        }
        cout << c[i] << " ";
    }

}

void getInter(vector<int> a, vector<int> b){
    int m = a.size();
    int n = b.size();
    int k = a.size() + b.size();
    vector<int> c(k, 0);
    cout << k << ",";
    for(int i = 0; i < k; i++) {
        for(int j = 0; j <= m-1; j++) {
            c[i] += a[j] * b[i + j];
        }
        cout << c[i] << " ";
    }
}

int main(){
    string strA;
    getline(cin, strA);
    stringstream ss(strA);
    string numA, strNumsA;
    getline(ss, numA, ',');
    getline(ss, strNumsA, ',');
    stringstream s1(strNumsA);
    string num;
    vector<int> numsA;
    while(getline(s1, num, ' ')){
        numsA.push_back(stoi(num));
    }

    string strB;
    getline(cin, strB);
    stringstream ss1(strB);
    string numB, strNumsB;
    getline(ss1, numB, ',');
    getline(ss1, strNumsB, ',');
    stringstream s2(strNumsB);
    string nu;
    vector<int> numsB;
    while(getline(s2, nu, ' ')){
        numsB.push_back(stoi(nu));
    }

    getConv(numsA, numsB);

    getInter(numsA, numsB);

}


//int main(){
//    string stringLine;
//    getline(cin, stringLine);
//    for (auto it1 = stringLine.begin(); it1 != stringLine.end(); it1++)
//    {
//        *it1 = toupper(*it1);
//    }
//
//    string dictWords;
//    getline(cin, dictWords);
//    for (auto it1 = dictWords.begin(); it1 != dictWords.end(); it1++)
//    {
//        *it1 = toupper(*it1);
//    }
//
//    stringstream ss;
//    map<string, int> dictMap;
//    ss << dictWords;
//    string temp;
//    int ind = 0;
//    while(ss >> temp){
//        auto ret = dictMap.insert(make_pair(temp, ind));
//        if(ret.first->second == 0){
//            dictMap[temp] = ind;
//        }
//        ind++;
//    }
//
//    ss << stringLine;
//    vector<string> res;
//    while(ss >> temp){
//        if(temp[0] == '\"'){
//            res.push_back(temp);
//        }
//    }
//
//
//
//}
