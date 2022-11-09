
/**************************************************************************
 * work1.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.10.12
 * 
 * @Description:
 *  
 ***************************************************************************/

#include <iostream>
#include <vector>
#include <queue>

using namespace std;

vector<int> dirs = {1, 0 ,1};
int resMin = INT_MAX;
void dfs(int m, int n, int i, int j, int sum, vector<vector<int>> map){
    if(i >= m || j >= n) return;
    if(i == m-1 && j == n-1){
        if(sum < resMin){
            resMin = sum;
        }
    }
    cout << "i , j : " << i << " " << j <<" sum: " << sum <<endl;

    for(int a=1; a<dirs.size(); a++){
        if(i+dirs[a-1] >= m || j+dirs[a] >= n) continue;
        sum += map[i+dirs[a-1]][j+dirs[a]];
        dfs(m, n, i+ dirs[a-1], j+dirs[a], sum, map);
        sum -= map[i+dirs[a-1]][j+dirs[a]];
    }

}

int findMin(vector<vector<int>> map){
    int m = map.size();
    int n = map[0].size();
    vector<vector<int>> dp(m, vector<int>(n , 0));
    
    dp[0][0] = map[0][0];
    for(int i=1; i<m;i++){
        dp[i][0] = dp[i-1][0] + map[i][0];
    }

    for(int j=1; j<n;j++){
        dp[0][j] = dp[0][j-1] + map[0][j];
    }

    for(int i=1; i<m; i++){
        for(int j=1; j<n;j++){
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + map[i][j];
        }
    }
    return dp[m-1][n-1];
}

int main(){
    vector<vector<int>> map = {{1, 3, 1},
                               {1, 5, 1},
                               {4, 2, 1}};

    resMin = findMin(map);
//    dfs(map.size(), map[0].size(), 0, 0, map[0][0], map);
    cout << resMin << endl;

}