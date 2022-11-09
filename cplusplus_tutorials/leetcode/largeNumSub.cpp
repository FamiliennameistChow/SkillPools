/**************************************************************************
 * largeNumSub.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.10.18
 * 
 * @Description: 大数相减
 *  高精度的减法运算，两个数值非常大的数相减，大数相减是通过字符串实现的，
 *  运用的思想是我们平时通过笔纸去算的思想，通过不足借位的思想进行运算，
 *  如果减数比被减数大，结果为负数，此时减数减被减数等于被减数减减数的相反数。
 *  输入：
 *  第一行是数a(string)
 *  第二行是数b(string)
 *  输出
 *  a-b的值(string)
 ***************************************************************************/
# include <iostream>
# include <string>

using namespace std;

// 这里保证a>b
string largeNumSub(string strA, string strB){
    long long lenA = strA.size()-1;
    long long lenB = strB.size()-1;

    int flag = 0; // 设置进位标志符

    string res;
    int a, b;
    // 从末尾开始计算
    while(lenA >= 0){
        a = strA[lenA] - '0';
        b = lenB < 0 ? 0 : strB[lenB] - '0';

        int t = a - flag - b;
        if(t < 0){
            flag = 1;
            t += 10;
        }else{
            flag = 0;
        }

        cout << "t: " << t << " flag: " << flag << endl;
        cout << " lenA: " << lenA << " lenB: " << lenB << endl;

        res.push_back(t+'0');

        lenA--;
        lenB--;
    }

    cout << res << endl;

    // 去除前置的0
    int validInd = -1;
    for(long long i = res.size()-1; i>=0; i--){
        if(res[i] != '0'){
            validInd = i;
            break;
        }
    }

    cout <<"valid: " << validInd << endl;

    if(validInd == -1){ //说明res全是0
        return "0";
    }

    res.resize(validInd+1);

    reverse(res.begin(), res.end());
    return res;
}

int main(){
    string a, b;
    cin >> a;
    cin >> b;
    string result = " ";

    // 判断 a b的大小
    if(a.size() < b.size()){
        swap(a, b);
        result += '-';
    }

    if(a.size() == b.size()){
        int i = 0;
        while(i < a.size()){
            if(a[i] < b[i]){
                swap(a, b);
                result += '-';
                break;
            }else if(a[i] == b[i]){
                i++;
            }else{
                break;
            }
        }
    }

    string res = largeNumSub(a, b);

    result += res;

    cout << result << endl;
}

