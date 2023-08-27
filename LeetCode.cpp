#include <iostream>
#include <map>
#include <vector>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace std;

struct TreeNode {
	int val;
	struct TreeNode* left;
	struct TreeNode* right;
	TreeNode(int x) :
		val(x), left(nullptr), right(nullptr) {}
};

struct ListNode {
	int val;
	struct ListNode* next;

    ListNode(int x) : val(x), next(nullptr) {}
};

#pragma region 二叉树是否包含子结构
//class Solution {
//public:
//	bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
//		if (pRoot2 == nullptr) return false;
//		if (pRoot1 == nullptr && pRoot2 != nullptr) return false;
//		if (pRoot1 == nullptr || pRoot2 == nullptr) return true;
//
//		bool res1 = Transvel(pRoot1, pRoot2);
//		bool res2 = HasSubtree(pRoot1->left, pRoot2);
//		bool res3 = HasSubtree(pRoot1->right, pRoot2);
//
//		return res1 || res2 || res3;
//	}
//
//	bool Transvel(TreeNode* root1, TreeNode* root2) {
//		if (root1 == nullptr && root2 != nullptr) return false;
//		if (root1 == nullptr || root2 == nullptr) return true;
//
//		if (root1->val != root2->val) return false;
//		cout << root1->val << "," << root2->val << endl;
//
//
//		return Transvel(root1->left, root2->right)
//			&& Transvel(root1->right, root2->right);
//	}
//};
#pragma endregion

#pragma region 二叉树转换双向链表
//class Solution {
//public:
//	TreeNode* head = nullptr;
//	TreeNode* last = nullptr;
//	TreeNode* Convert(TreeNode* root) {
//		if (!root) return nullptr;
//
//		Convert(root->left);
//
//		if (last == nullptr) {
//			last = root;
//			head = root;
//		}
//		else {
//			root->left = last;
//			last->right = root;
//			last = root;
//		}
//
//		Convert(root->right);
//
//		return head;
//	}
//};
#pragma endregion

#pragma region 二叉树打印成多行
//class Solution {
//public:
//    queue<TreeNode*> queue;
//    vector<vector<int>> res;
//
//    vector<vector<int> > Print(TreeNode* pRoot) {
//        queue.push(pRoot);
//
//        while (!queue.empty()) {
//            vector<int> cur;
//            int size = queue.size() - 1;
//            for (int i = 0; i <= size; i++) {
//                TreeNode* tmp = queue.front();
//                queue.pop();
//
//                if (tmp != nullptr) {
//                    cur.push_back(tmp->val);
//                    queue.push(tmp->left);
//                    queue.push(tmp->right);
//                }
//            }
//            res.push_back(cur);
//            cur.clear();
//        }
//
//        return res;
//    }
//
//};
#pragma endregion

#pragma region  二叉树两个节点的最近公共祖先
//class Solution {
//public:
//
//    vector<int> path1;
//    vector<int> path2;
//    bool found;
//    int res;
//
//    int lowestCommonAncestor(TreeNode* root, int o1, int o2) {
//        // write code here
//        FindPath(root, o1, path1);
//        found = false;
//        FindPath(root, o2, path2);
//
//        cout << path1.size() << "," << path2.size() << endl;
//
//        for (int i = 0; i < path1.size() && i < path2.size(); i++) {
//            if (path1[i] == path2[i])
//                res++;
//            else break;
//        }
//
//        return path1[res - 1];
//
//    }
//
//    void FindPath(TreeNode* root, int target, vector<int>& path) {
//        if (root == nullptr || found) return;
//
//        path.push_back(root->val);
//        if (root->val == target) {
//            found = true;
//            cout << "Find :" << root->val << "," << endl;
//            return;
//        }
//
//        FindPath(root->left, target, path);
//        FindPath(root->right, target, path);
//
//        if (found) return;
//        path.pop_back();
//    }
//};
#pragma endregion

#pragma region 连续数组最大和
// class Solution {
// public:
//     int FindGreatestSumOfSubArray(vector<int> array) {
//         vector<int> dp(array.size(), 0);
//         dp[0] = array[0];
//         int maxSum = dp[0];
//
//         for (int i = 1; i < array.size(); i++) {
//             dp[i] = max(dp[i - 1] + array[i], array[i]);
//             maxSum = max(dp[i], maxSum);
//         }
//
//         return maxSum;
//     }
// };
#pragma endregion

#pragma region 连续子数组最大和，返回数组序列
// class Solution {
// public:
//     /**
//      * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
//      *
//      *
//      * @param array int整型vector
//      * @return int整型vector
//      */
//     vector<int> FindGreatestSumOfSubArray(vector<int>& array) {
//         // write code here
//         vector<int> dp(array.size(),0);
//         vector<int> res;
//         dp[0] = array[0];
//         int start = 0, end = 0;
//         int l = 0, r = 0;
//         int maxSum = dp[0];
//
//         for(int i = 1; i <= array.size() - 1; i++)
//         {
//             end++;
//             dp[i] = max(dp[i - 1] + array[i], array[i]);
//
//             if(dp[i-1] +array[i] < array[i])
//             {
//                 start = end;
//             }
//             if(dp[i] >= maxSum)
//             {
//                 maxSum = dp[i];
//                 l = start;
//                 r = end;
//             }
//         }
//
//         for (int i = l; i <= r; i++)
//             res.push_back(array[i]);
//
//         return res;
//     }
// };
#pragma endregion

#pragma region 二维数组最大和路径
// class Solution {
// public:
//
//     int maxValue(vector<vector<int> >& grid) {
//         // write code here
//         vector<int> t(grid[0].size(), 0);
//         vector<vector<int>> dp(grid.size(), t);
//         dp[0][0] = grid[0][0];
//         int maxSum = dp[0][0];
//
//         for(int i = 1; i < t.size(); i++)
//         {
//             dp[0][i] = grid[0][i] + dp[0][i - 1];
//             maxSum = max(dp[0][i], maxSum);
//         }
//
//         for(int j = 1; j < grid.size(); j++)
//         {
//             dp[j][0] = grid[j][0] + dp[j - 1][0];
//             maxSum = max(dp[j][0], maxSum);
//         }
//
//         for(int j = 1; j < dp.size(); j++)
//         {
// 	        for(int i = 1; i < t.size(); i++)
// 	        {
//                 dp[j][i] = max(dp[j - 1][i] + grid[j][i], dp[j][i - 1] + grid[j][i]);
//                 maxSum = max(maxSum, dp[j][i]);
// 	        }
//         }
//
//         return maxSum;
//     }
// };
#pragma endregion

#pragma region 哈希滑动窗口求字符串最长不重复子串
// class Solution {
// public:
//     /**
//      * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
//      *
//      *
//      * @param s string字符串
//      * @return int整型
//      */
//     int lengthOfLongestSubstring(string s) {
//         // write code here
//         unordered_map<char, int> hash;
//         int maxLen = 0;
//         for(int left = 0, right = 0; right < s.size(); right++)
//         {
//             hash[s[right]]++;
//             while(hash[s[right]] > 1)
//             {
//                 hash[s[left++]]--;
//             }
//             maxLen = max(maxLen, right - left + 1);
//         }
//
//         return maxLen;
//     }
// };
#pragma endregion

#pragma region 数字串有多少种翻译方式
// class Solution {
// public:
//     /**
//      * 解码
//      * @param nums string字符串 数字串
//      * @return int整型
//      */
//     int solve(string nums)
// 	{
//         if (nums == "0") return 0;
//         if (nums == "10" || nums == "20") return 1;
//
//         vector<int> dp(nums.size(), 1);
//         dp[1] = (nums[0] == '1' || nums[0] == '2') && (nums[1] < '7' && nums[1] >'0') ? 2 : 1;
//
//         for(int i = 2; i < nums.size(); i++)
//         {
//             if (nums[i] == '0' && nums[i - 1] != '1' && nums[i - 1] != '2')
//                 return 0;
//             if ((nums[i - 1] == '1' && nums[i] != '0')
//                 || (nums[i - 1] == '2' && nums[i] > '0' && nums[i] < '7'))
//             {
//                 dp[i] = dp[i - 1] + dp[i - 2];
//             }
//             else
//                 dp[i] = dp[i - 1];
//         }
//
//         return dp[dp.size() - 1];
//     }
// };
#pragma endregion

#pragma region 西山居笔试第一题_整除几次2数组和为偶数
// void Func(vector<long long>& array,vector<int>& res)
// {
//     long long sum = 0;
//     int doubleCnt = 0;
//     int minSingle = 1000001;
//
// 	for(long long i = 0; i < array.size(); i++)
// 	{
//         if (doubleCnt == 0 && array[i] % 2 == 0 && array[i] > 1)
//             doubleCnt++;
//         if (doubleCnt == 0 && array[i] % 2 == 1 && array[i] < minSingle)
//             minSingle = array[i];
//         sum += array[i];
// 	}
//
//     if (sum % 2 == 0) res.push_back(0);
//     else
//     {
//         if (doubleCnt > 0) res.push_back(1);
//         else
//         {
//             int k = 0;
//             while (minSingle != 0)
//             {
// 	            minSingle /= 2;
//             	k++;
//             }
//             res.push_back(k);
//         }
//     }
// }

// int main()
// {
//     int n = 0;
//     cin >> n;
//
//     vector<long long> array(0);
//     vector<int> res(0);
//
//     for (int i = 0; i < n; i++)
//     {
//         long long k = 0;
//         cin >> k;
//         for (int j = 0; j < k; j++)
//         {
//             int x;
//             cin >> x;
//             array.push_back(x);
//         }
//         Func(array, res);
//         array.clear();
//     }
//
//     for (int i = 0; i < res.size(); i++) {
//         cout << res[i] << endl;
//     }
//
//     cin >> n;
//     return 0;
// }
#pragma endregion

#pragma region 西山居笔试第二题_前后交替插入解密字符串
// void HandleString(const string& str)
// {
//     string res(str.size(), 0);
//
//     int end = str.size() - 1, start = 0;
//     int n = res.size() - 1;
//     while(end > start)
//     {
//         res[n] = str[end];
//         end--;
//         n--;
//         res[n] = str[start];
//         start++;
//         n--;
//     }
//     if (end == start && n == 0) res[n] = str[end];
//
//     cout << res;
// }
//
//
// int main()
// {
//     string s;
//     cin >> s;
//     string res;
//
//     HandleString(s);
//
//     return 0;
// }
#pragma endregion

#pragma region 矩阵中路径
// class Solution {
// public:
// 	/**
// 	 * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
// 	 *
// 	 *
// 	 * @param matrix char字符型vector<vector<>>
// 	 * @param word string字符串
// 	 * @return bool布尔型
// 	 */
// 	bool hasPath(vector<vector<char> >& matrix, string word) {
// 		// write code here
// 		if (matrix.empty()) return false;
// 		vector<bool> t(matrix[0].size(), false);
// 		vector<vector<bool>> flag(matrix.size(), t);
//
// 		for(int y = 0; y < matrix.size(); y++)
// 		{
// 			for(int x = 0; x < matrix[0].size(); x++)
// 			{
// 				if (dfs(matrix, x, y, word, 0, flag))
// 					return true;
// 			}
// 		}
//
// 		return false;
// 	}
//
// 	bool dfs(const vector<vector<char>>& matrix, int x, int y, const string& word, int k, vector<vector<bool>>& flag)
// 	{
// 		if (x < 0 || x >= matrix[0].size() || y < 0 || y >= matrix.size() || matrix[y][x] != word[k] || flag[y][x]) return false;
// 		if (k == word.size() - 1) return true;
//
// 		flag[y][x] = true;
//
// 		if (dfs(matrix, x, y - 1, word, k + 1, flag)
// 			|| dfs(matrix, x, y + 1, word, k + 1, flag)
// 			|| dfs(matrix, x - 1, y, word, k + 1, flag)
// 			|| dfs(matrix, x + 1, y, word, k + 1, flag))
// 			return true;
//
// 		flag[y][x] = false;
// 		return false;
// 	}
//
// };
#pragma endregion

#pragma region 机器人移动路径
// class Solution {
// public:
// 	int movingCount(int threshold, int rows, int cols) {
// 		int res = 0;
// 		vector<bool> t(rows, false);
// 		vector<vector<bool>> flag(cols, t);
// 		dfs(threshold, rows, cols, 0, 0, res,flag);
//
// 		return res;
// 	}
//
// 	void dfs(const int& threshold, const int& rows, const int& cols, int m, int n, int& res,vector<vector<bool>>& flag)
// 	{
// 		cout << "(" << n << "," << m << ") "<< endl;
// 		if (m >= rows || m < 0 || n >= cols || n < 0 || (m % 10 + m / 10 + n % 10 + n / 10) > threshold || flag[n][m])
// 			return;
// 		
// 		res++;
// 		flag[n][m] = true;
// 		dfs(threshold, rows, cols, m + 1, n, res,flag);
// 		dfs(threshold, rows, cols, m - 1, n, res,flag);
// 		dfs(threshold, rows, cols, m, n + 1, res,flag);
// 		dfs(threshold, rows, cols, m, n - 1, res,flag);
// 	}
// };
#pragma endregion

#pragma region 最长递增子序列
class SolutionAB
{
public:
	int lengthOfLIS(vector<int>& nums) {
		int n = (int)nums.size();
		if(n==0)
			return 0;

		int res = 0;
		vector<int> dp(n,0);
		for(int i = 0; i < nums.size(); i++)
		{
			dp[i] = 1;
			for(int j = 0; j < i; j++)
			{
				if(nums[j] < nums[i])
					dp[i] = max(dp[i], dp[j]+1);
				if(dp[i] > res)
					res = dp[i];
			}
		}

		return res;
	}
};
#pragma endregion

#pragma region 1187.使数组严格递增
class SolutionAA {
private:
	
public:
	int makeArrayIncreasing(vector<int>& arr1, vector<int>& arr2) {
        sort(arr2.begin(),arr2.end());
		int n = arr1.size();
		int m = arr2.size();

		if(n == 0) return -1;
		vector<int> dp(n,0);
		for(int i = 1; i < m; i++)
		{
			if(arr1[i] > arr1[i-1])
			{
				dp[i] = dp[i-1];
			}
			else if(arr1[i] <= arr1[i-1])
			{
			}
		}
		
	}
};
#pragma endregion

#pragma region 2413.最小偶倍数
class SolutionAC {
public:
	int smallestEvenMultiple(int n) {
		if (n == 1) return 2;
		if (n % 2 == 0) return n;

		return 2 * n;

	}
};
#pragma endregion

#pragma region 1027.最长等差数列
class SolutionAD
{
public:
	int longestArithSeqLength(vector<int>& nums) {
		if(nums.size() <= 2)
			return 0;

		// vector<vector<int>> dp(nums)
	}
};
#pragma endregion

namespace NO2_无人机对象池
{
    class Pool
    {
    public:
        Pool(const int& _total)
        {
            total = _total;
            flightsState.assign(total,true);
        }

        vector<bool> flightsState;
        map<int,pair<int,int>> taskList;
    
        int total;
    
        bool Request(const int& num,const int& id)
        {
            int start = 0;
            int end = 0;
            for(int i = 0; i < total; i++)
            {
                if(flightsState[i])
                {
                    start = i;
                    break;
                }
            }
            end = start;
            for(int i = start + 1; i < total; i++)
            {
                if(flightsState[i])
                {
                    end = i;
                    if(end - start == num)
                        break;
                }
            }

            if(end - start == num)
            {
                for(int i = 0; i < num; i++)
                {
                    flightsState[i+start] = false;
                }
                taskList.emplace(id,pair<int,int>(start,end));
                return true;
            }
            else
                return false;
        }
    
        bool Return(const int& id)
        {
            int start = taskList[id].first;
            int end = taskList[id].second;

            for(int i = start; i<=end;i++)
            {
                flightsState[i] = true;
            }
            return true;
        }
    };

    struct Task
    {
    public:
        Task(int _id, int _num)
        {
            id = _id;
            num = _num;
        }
        int id;
        int num;
    };
    
    int test_main()
    {
        int total,k;
        cin>>total;
        cin>>k;
    
        int task,num;
        queue<Task> taskQue;
        Pool pool(total);
        for(int i=0; i < k; ++i)
        {
            cin>>task;
            cin>>num;

            if(task >= 0)
            {
                const bool res = pool.Request(num, task);
            
                cout<<(res ? num : 0)<<endl;
                if(!res)
                    taskQue.emplace(task,num);
            }
            else
            {
                pool.Return(num);

                if(taskQue.empty())
                    cout<<0<<endl;
                else
                {
                    auto topTask = taskQue.front();
                    if(pool.Request(topTask.num,topTask.id))
                    {
                        cout<<topTask.num<<endl;
                        taskQue.pop();
                    }
                    else
                        cout<<0<<endl;
                }
            }
        }
    
        return 0;
    }
}

namespace NO1_范围判断
{
    struct Vector3
    {
    public:
        Vector3(float x,float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        bool Hited(const Vector3& target,int range)
        {
            float dis = ((target.X-X)*(target.X-X)
                + (target.Y-Y)*(target.Y-Y)
                + (target.Z-Z)*(target.Z-Z));
            return dis < (float)(range*range);
        }
    
        float X,Y,Z;
    };

    int test_main()
    {
        int n,r;
        cin>> n;
        cin>>r;

        if(n == 0)
            return n;

        vector<Vector3> targets;
        Vector3 pos(0,0,0);
    
        for(int i = 0; i < n;i++)
        {
            float x,y,z;
            cin>>x;
            cin>>y;
            cin>>z;
            targets.emplace_back(x,y,z);
        }
        {
            float x,y,z;
            cin>>x;
            cin>>y;
            cin>>z;
            pos.X= x;
            pos.Y = y;
            pos.Z = z;
        }

        int res = 0;
        for(auto target : targets)
        {
            if (target.Hited(pos,r))
                res++;
        }
        cout<<res;

        return 0;
    }
}

namespace _1027_最长等差数列
{
    class Solution {
    public:
        int longestArithSeqLength(vector<int>& nums) {
            int res = 0;
            vector<unordered_map<int,int>> dic(nums.size(),unordered_map<int,int>());
            // dic[1].emplace(nums[0]-nums[1],2);
            for(int i = 1; i < nums.size(); i++)
            {
                for(int j = i-1; j >= 0; j--)
                {
                    int dis = nums[j] - nums[i];
                    if(dic[i].count(dis)) continue;
                    auto pair = dic[j].find(dis);
                    dic[i].emplace(dis,pair == dic[j].end() ? 2 : pair->second + 1);
                    res = max(res,dic[i].at(dis));
                }
            }
            return res;
        }
    };
}

namespace _1105_填充书架
{
    class Solution {
    public:
        int res = 0;
        int curWidth = 0;
        int curHeight = 0;
        
        int minHeightShelves(vector<vector<int>>& books, int shelfWidth) {
            int n = books.size();
            vector<int> dp(n+1,1000000);
            dp[0] = 0;
            for(int i = 0; i < n; i++)
            {
                curWidth = 0,curHeight = 0;
                for(int j = i; j >= 0; j--)
                {
                    curWidth += books[j][0];
                    if(curWidth > shelfWidth)
                        break;
                    curHeight = max(curHeight,books[j][1]);
                    dp[i+1] = min(dp[i+1],dp[j] + curHeight);
                }
            }

            return dp[n];
        }
    };
}

namespace _1163_按字典排在最后的子串
{
    
}

namespace ChillyRoomTest
{
    class SolutionAE
    {
        
    };
}

class Solution_AR
{
public:
    int dp[2] = {1,2};

    int CalculateN(int n)
    {
        if(n <= 2) return dp[n-1];
        int i = 3;

        while(i<n)
        {
            if(i%2 != 0)
            {
                int tmp  = dp[1];
                dp[1] = dp[1] * 2;
                dp[0] = tmp;
            }
            else
            {
                dp[0] = dp[1];
                dp[1]++;
            }
            i++;
        }

        return dp[1];
    }
};

class Solution_AE
{
public:

    int start,end;

    void SortStr(string& str)
    {
        int n = str.size() - 1;
        int i = 0;

        while(i<=n)
        {
            if(str[i] >= 'a')
            {
                start = i;
                i++;
                while(str[i] < 'a' && i <= n)
                {
                    i++;
                    end = i;
                }

                std::sort(str.begin()+start+1,str.begin()+end);
            }
            else
            {
                i++;
            }
            
            i++;
        }
    }
    
};

class solution_AP
{
public:
    int return_x,return_y;
    int dp[];

    
};

#pragma region 剑指05.替换空格

class Solution_0 {
public:
	string replaceSpace(string s) {
        int count = 0,len = s.size();

        for (char c : s) {
            if (c == ' ')
                count++;
        }

        s.resize(len + 2 * count);
        for (int i = len - 1, j = s.size() - 1; 
            i < j; i--, j--) {
            if (s[i] != ' ') {
                s[j] = s[i];
            }
            else {
                s[j] = '0';
                s[j - 1] = '2';
                s[j - 2] = '%';
                j -= 2;
            }
        }

        return s;
	}
};

#pragma endregion

#pragma region 剑指58.左旋转字符串

class Solution_1 {
public:
	string reverseLeftWords(string s, int n) {
		int len = s.size();
        string res = string(len, '0');

        for (int i = 0, j=n; i < len; i++,j++) {
            if (j == len) j = 0;
            
            res[i] = s[j];
        }

        return res;
	}
};

#pragma endregion

#pragma region 剑指06.反向打印链表
class Solution_2 {
public:
    vector<int> res;

	vector<int> reversePrint(ListNode* head) {
        reverse(head);
        return res;
	}

    void reverse(ListNode* node) {
        if (node == nullptr) return;

        reverse(node->next);
        res.push_back(node->val);
    }
};
#pragma endregion

#pragma region 剑指24.翻转链表

class Solution_3 {
public:
	ListNode* reverseList(ListNode* head) {
        ListNode* pre = head;
        ListNode* cur = head->next;
        head->next = nullptr;
        ListNode* next;

        while(cur != nullptr) {
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }

        return pre;
	}
};

#pragma endregion

#pragma region 剑指35.复制复杂链表
class Node_1 {
public:
	int val;
	Node_1* next;
	Node_1* random;

	Node_1(int _val) {
		val = _val;
		next = nullptr;
		random = nullptr;
	}
};

class Solution_4 {
public:
    Node_1* copyRandomList(Node_1* head) {
        for(auto p = head; p != nullptr; p = p->next->next)
        {
            Node_1* newNode = new Node_1(p->val);
            newNode->next = p->next;
            p->next = newNode;
        }
        
        for(auto p = head; p != nullptr; p = p->next->next)
        {
            p->next->random = p->random == nullptr ? nullptr : p->random->next;
        }

        Node_1* ret = head->next;
        for(auto p = head; p != nullptr; p = p->next)
        {
            auto nodeNew = p->next;
            p->next = p->next->next;
            nodeNew->next = p->next == nullptr ? nullptr : p->next->next;
        }

        return ret;
    }
};
#pragma endregion

#pragma region 剑指18.删除链表的节点

class Solution_5 {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        if(!head) return head;

        for(auto p1 = head, p2 = head->next; p2 != nullptr; (p1 = p2,p2 = p2->next))
        {
            if(p2->val == val)
            {
                p1->next = p2->next;
                return head;
            }
        }
    }
};

#pragma endregion 

#pragma region 剑指22.链表中倒数的第k个节点

class Solution_6 {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode* fast = head;
        while(k > 0)
        {
            fast = fast->next;
            k--;
        }

        while(fast)
        {
            head = head->next;
            fast = fast->next;
        }

        return head;
    }
};

#pragma endregion

#pragma region 剑指25.合并两个排序的链表

class Solution_7 {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode head{-1};

        ListNode* prev = &head;
        while(l1 != nullptr && l2 != nullptr)
        {
            if(l1->val > l2->val)
            {
                prev->next = l2;
                l2 = l2->next;
            }
            else
            {
                prev->next = l1;
                l1 = l1->next;
            }
            prev = prev->next;
        }

        prev->next = l1 == nullptr ? l2 : l1;

        return head.next;
    }
};

#pragma endregion

#pragma region 剑指52.两个链表的第一个公共节点

class Solution_8 {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(headA == nullptr || headB == nullptr) return nullptr;
        
        int n1 = 0;
        int n2 = 0;
        
        for(const auto* p = headA; p != nullptr; p = p->next)
        {
            n1++;
        }
        for(const auto* p = headB; p != nullptr; p = p->next)
        {
            n2++;
        }

        int diff = n1 - n2;
        if(diff > 0)
        {
            for(int i = diff; diff != 0; diff--)
            {
                headA=headA->next;
            }
        }
        else if(diff < 0)
        {
            for(int i = diff; diff != 0; diff++)
            {
                headB=headB->next;
            }
        }

        while(headA != headB)
        {
            if(headA == nullptr) return nullptr;
            headA = headA->next;
            headB = headB->next;
        }

        return headA;
    }
};

#pragma endregion

#pragma region 剑指21.调整数组顺序使奇数位于偶数前面
class Solution_9 {
public:
    vector<int> exchange(vector<int>& nums) {
        if(nums.size() < 2) return nums;
        
        int fast = nums.size() - 1;
        int slow = 0;

        while(slow != fast)
        {
            if(nums[slow] % 2 != 0 && slow != fast)
                slow++;

            if(nums[fast] % 2 == 0 && slow != fast)
                fast--;
            
            if(nums[slow] % 2 == 0 && nums[fast] % 2 != 0)
            {
                const int tmp = nums[slow];
                nums[slow] = nums[fast];
                nums[fast] = tmp;
                slow++;
            }
        }

        return nums;
    }
};
#pragma endregion 

#pragma region 剑指57.和为s的两个数字

class Solution_10 {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        if(n<2) return nums;
        
        int front{0},back{n-1};
        int mid;
        int sum;
        while(front != back)
        {
            sum = nums[front] + nums[back];
            if(sum == target)
            {
                return vector<int>{nums[front],nums[back]};
            }

            if(sum > target)
            {
                mid = (front + back) / 2;
                if(nums[front] + nums[mid] > target)
                    back = mid - 1;
                else if(nums[front] + nums[mid] < target)
                    --back;
                else
                    return vector<int>{nums[front],nums[mid]};
            }
            else
            {
                ++front;
            }
        }

        return vector<int>{0,0};
    }
};

#pragma endregion

#pragma region 剑指58.翻转单词顺序
class Solution_11 {
public:
    string reverseWords(string s) {
        if(s.empty()) return s;
        int n = s.length();
        if(n == 1) return s[0] == ' ' ? "" : s;
        
        string res{};
        stack<char> stack{};
        int front{s[0] == ' ' ? 1 : 0},back{s[n-1] == ' ' ? n-2 : n-1};
        
        while(back >= 0)
        {
            if(s[back] != ' ')
            {
                while(s[back] != ' ')
                {
                    stack.push(s[back]);
                    --back;
                    if(back < 0) break;
                }

                while(!stack.empty())
                {
                    res.push_back(stack.top());
                    ++front;
                    stack.pop();
                }
                res.push_back(' ');
            }

            --back;
        }

        return res.empty() ? res : res.erase(res.length()-1);
    }
};
#pragma endregion

#pragma region 剑指09.两个栈实现队列
class CQueue {
private:
    stack<int> instack{},outstack{};

    void in2out()
    {
        while(!instack.empty())
        {
            outstack.push(instack.top());
            instack.pop();
        }
    }
public:
    CQueue() {
        
    }
    
    void appendTail(int value) {
        instack.push(value);
    }
    
    int deleteHead() {
        if(outstack.empty())
        {
            if(instack.empty())
            {
                return -1;
            }
            in2out();
        }

        int ret = outstack.top();
        outstack.pop();

        return ret;
    }
};

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue* obj = new CQueue();
 * obj->appendTail(value);
 * int param_2 = obj->deleteHead();
 */
#pragma endregion

#pragma region 剑指30.包含min函数的栈
class MinStack {
private:
    stack<int> minstack{};
    stack<int> s{};
public:
    /** initialize your data structure here. */
    MinStack() {
        minstack.push(INT_MAX);
    }
    
    void push(int x) {
        minstack.push(::min(x,minstack.top()));
        s.push(x);
    }
    
    void pop() {
        s.pop();
        minstack.pop();
    }
    
    int top() {
        return s.top();
    }
    
    int min() {
        return minstack.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(x);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->min();
 */
#pragma endregion

#pragma region 剑指59.队列最大值
class MaxQueue {
private:
    queue<int> q{};
    stack<int> maxStack{};
    int prevMax;
public:
    MaxQueue() {
        maxStack.push(INT_MIN);
    }
    
    int max_value() {
        return maxStack.top();
    }
    
    void push_back(int value) {
        q.push(value);
        maxStack.push(value > maxStack.top() ? value : maxStack.top());
    }
    
    int pop_front() {
       int res = q.back();
        q.pop();

        
        return res;
    }
};

/**
 * Your MaxQueue object will be instantiated and called as such:
 * MaxQueue* obj = new MaxQueue();
 * int param_1 = obj->max_value();
 * obj->push_back(value);
 * int param_3 = obj->pop_front();
 */
#pragma endregion

#pragma region 剑指32.从上到下打印二叉树

class Solution_12 {
private:
    vector<int> res;
    
public:
    vector<int> levelOrder(TreeNode* root) {
        if(root == nullptr) return res;

        queue<TreeNode*> queue;
        queue.push(root);
        while(!queue.empty())
        {
            int size = queue.size();
            for(int i = 0; i < size; i++)
            {
                auto tmp = queue.front();
                queue.pop();
                if(tmp->left != nullptr) queue.push(tmp->left);
                if(tmp->right != nullptr) queue.push(tmp->right);
                res.push_back(tmp->val);
            }
        }

        return res;
    }
};

#pragma endregion

#pragma region 剑指53-I.排序数组中查找数字
class Solution_13
{
public:
    int search(vector<int>& nums, int target) {
        int n = nums.size();

        int left = binarySearch(nums, target, true,0,n-1);
        int right = binarySearch(nums, target, false,left,n-1);

        return right-left +1;
    }
    
    int binarySearch(vector<int>& nums, int target, bool lower,int start, int end)
    {
        int left{start},right{end},mid;

        while(right >= left)
        {
            mid = left + (right - left) / 2;
            if(nums[mid] > target) {
                right = mid - 1;
            }
            else if(nums[mid] < target){
                left = mid + 1;
            }
            else if(nums[mid] == target){
                if(lower){
                    right = mid - 1;
                }
                else{
                    left = mid + 1;
                }
            }
        }
        return lower ? left :right;
    }
};
#pragma endregion

#pragma region 剑指53-II.0~n-1中缺失的数字
class Solution_14 {
public:
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        if(n==1) return nums[0] == 0 ? 1 : 0;
        if(nums[n-1] == n-1) return n;
        if(nums[0] != 0) return 0;
        

        int left{0},right{n-1},mid;
        while(right >= left)
        {
            mid = left + (right - left) / 2;
            if(nums[mid] > mid)
            {
                right = mid - 1;
            }
            else if(nums[mid] == mid)
            {
                left = mid + 1;
            }
        }

        return nums[mid] == mid ? mid + 1 : mid;
    }
};
#pragma endregion

#pragma region 剑指04.查找二维数组
class Solution_15 {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if(matrix.empty()) return false;
        if(matrix[0].empty()) return false;
        
        int top{0},down{(int)matrix.size()-1},mid{};
        for(int i = 0; i < matrix.size(); ++i)
        {
            if(matrix[i][0] <= target && matrix[i][matrix[i].size()-1] >= target)
            {
                if(binarySearch(matrix[i],target))
                    return true;
            }
        }

        return false;
    }
    
    bool binarySearch(vector<int>& nums,const int& target)
    {
        int left{0},right{(int)nums.size()-1},mid;
        while(left<=right)
        {
            mid = left + (right - left) / 2;
            if(nums[mid] < target)
            {
                left = mid+1;
            }
            else if(nums[mid] > target)
            {
                right = mid - 1;
            }
            else
            {
                return true;
            }
        }
        return false;
    }
};
#pragma endregion

#pragma region 剑指11.旋转数组最小数值
class Solution_16 {
public:
    int minArray(vector<int>& numbers) {
        const int n = numbers.size() - 1;
        if(n==0) return numbers[0];
        if(n==1) return min(numbers[0],numbers[1]);
        if(numbers[n]>numbers[0]) return numbers[0];
        
        int l{0},r{(int)numbers.size()-1},mid;
        while(r>=l)
        {
            mid = l + (r-l)/2;
            if(numbers[mid] < numbers[r])
            {
                r = mid;
            }
            else if(numbers[mid] > numbers[r])
            {
                l = mid + 1;
            }
            else
            {
                --r;
            }
        }

        return numbers[l];
    }
};
#pragma endregion

#pragma region 剑指50.第一个只出现一次的字符
class Solution_17 {
public:
    char firstUniqChar(string s) {
        unordered_map<char,int> map{};
        for (auto c : s)
        {
            ++map[c];
        }
        for(int i = 0; i < s.size(); ++i)
        {
            if(map[s[i]] == 1)
                return s[i];
        }

        return ' ';
    }
};
#pragma endregion

#pragma region 剑指32.从上到下打印二叉树
class Solution_18 {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res{};
        if(root == nullptr) return res;

        int depth = 1;
        queue<TreeNode*> q;
        q.push(root);
        while(!q.empty())
        {
            int size = q.size();
            vector<int> line{};
            for(int i = size; i > 0; --i)
            {
                auto t = q.front();
                line.push_back(t->val);
                if(t->left != nullptr) q.push(t->left);
                if(t->right != nullptr) q.push(t->right);
                q.pop();
            }
            ++depth;
            if(depth % 2 == 1)
                reverse(line.begin(),line.end());
            res.push_back(line);
        }

        return res;
    }
};
#pragma endregion

#pragma region 剑指12.矩阵中的路径
class Solution_19 {
private:
    int x,y;
    string s;
    vector<vector<bool>> visited;
public:
    bool exist(vector<vector<char>>& board, string word) {
        x = board.size();
        y = board[0].size();
        s = word;
        if(word.size() > x*y) return false;
        
        visited = vector<vector<bool>>(x,vector<bool>(y,false));

        for(int i = 0; i < x; ++i)
        {
            for(int j = 0; j < y; ++j)
            {
                if(dfs(board,0,i,j))
                    return true;
            }
        }

        return false;
    }

    bool dfs(vector<vector<char>>& board, int index, int i, int j)
    {
        if(board[i][j] != s[index])
            return false;
        else if(index == s.length() - 1)
            return true;

        bool res = false;
        visited[i][j] = true;
        vector<pair<int,int>> directions{{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for(auto dir : directions)
        {
            int newi = i + dir.first, newj = j + dir.second;
            if(newi < 0 || newi >= x || newj < 0 || newj >= y)
                continue;

            if(!visited[newi][newj])
            {
                if(dfs(board,index+1,newi,newj))
                {
                    res = true;
                    break;
                }
            }
        }
        visited[i][j] = false;
        return res;
    }
};
#pragma endregion

#pragma region 剑指34.二叉树中和为某一值的路径
class Solution_20 {
private:
    
public:
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        vector<vector<int>> res{};
        vector<int> path{};
        dfs(res,path,target,root);

        return res;
    }

    void dfs(vector<vector<int>>& res, vector<int>& path,int target,const TreeNode* node)
    {
        if(node == nullptr) return;

        path.push_back(node->val);
        target -= node->val;
        if(target == 0 && node->left == nullptr && node->right == nullptr)
            res.push_back(path); 

        dfs(res,path,target,node->left);
        dfs(res,path,target,node->right);
        path.pop_back();
    }
};
#pragma endregion



#pragma region 剑指13.机器人的运动范围
class Solution_21 {
public:
    int movingCount(int m, int n, int k) {
        vector<vector<int>> visited(m,vector<int>(n,0));
        int res = 0;
        dfs(0,0,m,n,k,res,visited);
        return res;
    }

    void dfs(int i, int j,const int m,const int n,const int k, int& res,vector<vector<int>>& visited)
    {
        if(i < 0 || i == m || j < 0 || j == n) return;
        
        int local = countSum(i)+countSum(j);
        if(k > local || visited[i][j] == 1) return;

        visited[i][j] = 1;
        res++;
        dfs(i+1,j,m,n,k,res,visited);
        dfs(i,j+1,m,n,k,res,visited);
        dfs(i-1,j,m,n,k,res,visited);
        dfs(i,j-1,m,n,k,res,visited);
    }
    
    int countSum(int x)
    {
        int sum = 0;
        while(x > 0)
        {
            sum += x %10;
            x /= 10;
        }
        
        return sum;
    }
};
#pragma endregion

#pragma region 剑指36.二叉搜索树和双向链表
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}

    Node(int _val) {
        val = _val;
        left = NULL;
        right = NULL;
    }

    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};

#pragma endregion

#pragma region test
class Solution_22 {
public:
    Node* treeToDoublyList(Node* root) {
        
    }
};
#pragma endregion

#pragma region 雷火测试_1
class Solution_leiuhuo_1
{
private:
    int target;
    int min{1},max{1000};
public:
    Solution_leiuhuo_1(int x):target(x){}

    void newGuess(int guess)
    {
        if(guess == target)
        {
            cout<<"Congratulations! You guessed it right!"<<endl;
        }
        else if(guess < target)
        {
            if(guess < min)
            {
                cout << "Are you kidding me?"<<endl;
                return;
            }
            cout<<"It's too small, please keep guessing!"<<endl;
            min = guess;
        }
        else if(guess > target)
        {
            if(guess > max)
            {
                cout << "Are you kidding me?"<<endl;
                return;
            }
            cout<<"It's too big, please keep guessing!"<<endl;
            max = guess;
        }
    }
};

int main_1()
{
    int length;
    cin>>length;

    vector<int> inputs;
    int input;
    while(cin >> input)
    {
        inputs.emplace_back(input);
    }

    int target = inputs.back();
    Solution_leiuhuo_1 solu(target);
    for(int i : inputs)
    {
        solu.newGuess(i);
    }
    
    return 0;
}
#pragma endregion

#pragma region 雷火测试_2
class Solution_leihuo_2
{
private:
    const int a,b,total;
    bool hasA,hasB;
    int curLength{0};
    vector<int> res;
public:
    Solution_leihuo_2(int a,int b,int total):a(a),b(b),total(total){};

    bool newGroup(const int id,vector<int>& group)
    {
        int length = group.size();
        if(length + curLength > total)
            return false;
        for(int i = 0; i < length; ++i)
        {
            if(group[i] == a)
            {
                if(!hasA)
                    hasA = true;
                else
                    return false;
            }
            else if(group[i] == b)
            {
                if(!hasB)
                    hasB = true;
                else
                    return false;
            }
        }

        res.push_back(id);
        if(length + curLength == total)
        {
            printTeam();
        }
        else
        {
            curLength += length;
        }
        
        return true;
    }

    void printTeam()
    {
        for(int i = 0; i < res.size() - 1; ++i)
        {
            cout<<res[i]<<" ";
        }
        cout<<res.back()<<endl;
        hasA = false;
        hasB = false;
        curLength = 0;
        res.clear();
    }
};

int main_test()
{
    
    list<pair<int,vector<int>>> waitList;
    vector<vector<int>> matchPool;
    int n,m,a,b;
    cin>>n>>m>>a>>b;

    Solution_leihuo_2 solu(a,b,m);
    for(int x = 1; x <= n; ++x)
    {
        int length;
        cin>>length;
        vector<int> group;
        for(int i = 0; i < length; ++i)
        {
            int input;
            cin>>input;
            group.emplace_back(input);
        }
        matchPool.push_back(group);
    }

    for(int x = 1; x <= n; ++x)
    {
        auto group = matchPool[x-1];
        if(waitList.empty())
        {
            bool flag = solu.newGroup(x,group);
            if(!flag)
                waitList.push_back(make_pair(x,group));
        }
        else
        {
            waitList.push_back(make_pair(x,group));
            for(auto it = waitList.begin(); it != waitList.end();)
            {
                bool flag = solu.newGroup(it->first,it->second);
                if(flag)
                {
                    waitList.erase(it);
                    break;
                }
                ++it;
            }
        }
    }

    return 0;
}
#pragma endregion 

#pragma region 剑指Offer68.二叉搜索树最近公共先祖
class Solution_30 {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
    {
        auto low = p->val < q->val ? p : q;
        auto high = low == p ? q : p;
        
        while(true)
        {
            if(root->val < low->val)
            {
                root = root->right;
            }
            else if(root->val > high->val)
            {
                root = root->left;
            }
            else
            {
                break;
            }
        }

        return root;
    }
    
};
#pragma endregion

#pragma region 剑指68-II.二叉树的最近公共先祖
class Solution_31 {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root == nullptr) return root;
        if(root == p || root == q) return root;

        TreeNode* left = lowestCommonAncestor(root->left,p,q);
        TreeNode* right = lowestCommonAncestor(root->right,p,q);

        if(left != nullptr || right != nullptr)
            return root;

        if(left == nullptr && right == nullptr)
            return nullptr;

        return left == nullptr ? right : left;
    }
};
#pragma endregion

#pragma region 剑指37.序列化二叉树
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string res{};
        frontOrderSerialize(root,res);

        return res;
    }

    void frontOrderSerialize(TreeNode* node,string& res)
    {
        if(node == nullptr)
        {
            res.push_back('#');
            return;
        }
        
        res.push_back(node->val);
        frontOrderSerialize(node->left,res);
        frontOrderSerialize(node->right,res);
    }
    

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        return stringToNode(data);
    }

    long long index{-1};
    TreeNode* stringToNode(string& data)
    {
        ++index;
        if(data[index] == '#')
            return nullptr;

        TreeNode* node = new TreeNode(data[index]);
        
        TreeNode* left = stringToNode(data);
        TreeNode* right = stringToNode(data);
        node->left = left;
        node->right = right;

        return node;
    }
};
#pragma endregion

#pragma region 剑指38.字符串排列
class Solution_32 {
public:
    vector<string> res;
    unordered_set<string> existMap;
    vector<string> permutation(string s) {
        string path;
        string used(s.size(),'0');
        transvel(s,path,used);

        return res;
    }

    void transvel(const string& s, string& path, string& used)
    {
        if(s.size() == path.size())
        {
            if(existMap.find(path) == existMap.end())
            {
                res.push_back(path);
                existMap.insert(path);
            }
            
            return;
        }

        for(int i = 0; i < s.size(); ++i)
        {
            if(used[i] == 1)
                continue;

            path.push_back(s[i]);
            used[i] = 1;

            transvel(s,path,used);

            path.pop_back();
            used[i] = 0;
        }
    }
};
#pragma endregion

#pragma region 剑指07.重建二叉树
class Solution_33 {
public:
    int size;
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        size = preorder.size() -1;
        return build(preorder,inorder,0,size,0,size);
    }

    TreeNode* build(vector<int>& preorder, vector<int>& inorder,int preStart,int preEnd,int inStart,int inEnd)
    {
         if(preStart > preEnd || inStart > inEnd)
            return nullptr;
        
        TreeNode* node = new TreeNode(preorder[preStart]);
        
        int inRoot = find(inorder.begin(),inorder.end(),node->val) - inorder.begin();
        
        int preStart_r = (inRoot - inStart) + preStart + 1;
        
        node->left = build(preorder,inorder,preStart+1,preStart_r-1,inStart,inRoot-1);
        node->right = build(preorder,inorder,preStart_r,preEnd,inRoot+1,inEnd);

        return node;
    }
    
};
#pragma endregion

#pragma region 剑指16.数值的整数次方
class Solution_34 {
public:
    double myPow(double x, int n) {
        double res{1};

        if(x == 0) return x;
        if(n == 0) return 1;

        long long N = n;
        
        if(N < 0)
        {
            N *= -1;
            x = 1 / x;
        }
        
        while(N != 1)
        {
            if(N % 2 == 1)
                res *= x;
            x*=x;
            N /= 2;
        }
        res*=x;

        return res;
    }
};
#pragma endregion

#pragma region 剑指33.二叉搜索树后序遍历序列
class Solution_35 {
public:
    bool verifyPostorder(vector<int>& postorder) {
        return transverl(postorder,0,postorder.size()-1);
    }
    bool transverl(vector<int>& postorder, int startIndex, int endIndex)
    {
        if(startIndex >= endIndex)
            return true;
        
        int root = postorder[endIndex];
        int leftEnd = startIndex-1;
        for(int i = endIndex; i >= startIndex; --i)
        {
            if(postorder[i] < root)
            {
                leftEnd = i;
                break;
            }
        }
        for(int i = leftEnd-1; i >= startIndex; --i)
        {
            if(postorder[i] > root)
                return false;
        }

        bool left = transverl(postorder,startIndex,leftEnd);
        bool right = transverl(postorder,leftEnd+1,endIndex-1);

        return left && right;
    }
};
#pragma endregion

#pragma region 剑指17.打印从1到最大的n位数
class Solution_36 {
public:
    vector<int> printNumbers(int n) {
        if(n == 0) return vector<int>{0};
        
        int target{1};
        int x{10};
        while(n != 1)
        {
            x*=x;
            if(n % 2 == 1)
                target *= 10;
            n /= 2;
        }
        target *= x;

        vector<int> res(target);
        for(int i = 0; i < target; ++i)
        {
            res[i] = i;
        }
        return res;
    }
};
#pragma endregion

#pragma region 剑指10.I斐波那契数列
class Solution_37 {
public:
    int fib(int n) {
        if(n < 2) return n;
        
        int a{0}, b{1};
        for(int i = 2; i <= n; ++i)
        {
            int tmp = a;
            a = b;
            b = (tmp + b) % 1000000007;
        }

        return b;
    }
};
#pragma endregion

#pragma region 剑指63.股票最大利润
class Solution_38 {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.size() == 0) return 0;
        
        int profit{0}, minPrice{INT_MAX};
        for(auto price : prices)
        {
            minPrice = min(price,minPrice);
            profit = max(profit,price - minPrice);
        }

        return profit;
    }
};
#pragma endregion

#pragma region 剑指42.连续子数组最大和
class Solution_39 {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        if(n < 1) return 0;

        int max{-101};
        int curSum{-101};
        vector<int> dp(n);
        for(auto num : nums)
        {
            curSum = std::max(num,curSum+num);
            max = std::max(curSum,max);
        }

        return max;
    }
};
#pragma endregion

#pragma region 剑指47.礼物最大价值
class Solution_40 {
public:
    int maxValue(vector<vector<int>>& grid) {
        int m{(int)grid.size()};
        int n{(int)grid[0].size()};
        int res{0};
        
        vector<vector<int>> dp(grid);
        for(int j = 1; j < m; ++j)
        {
            dp[j][0] = dp[j-1][0] + grid[j][0];
        }
        for(int i = 1; i < n; ++i)
        {
            dp[0][i] = dp[0][i-1] + grid[0][i];
        }
        
        for(int j = 1; j < m; ++j)
        {
             for(int i = 1; i < n; ++i)
            {
                dp[j][i] = max(dp[j-1][i]+grid[j][i],dp[j][i-1]+grid[j][i]);
                res = max(dp[j][i],res);
            }
        }
        return res;
    }
};
#pragma endregion

#pragma region 剑指46.数值翻译成字符串
class Solution_41 {
public:
    int translateNum(int num) {
        
        string s = to_string(num);
        int n = s.size();
        vector<int> dp(n);
        dp[0] = 0;
        for(int i = 1; i < n; ++i)
        {
            string subs = s.substr(i-1,2);
            if(subs <= "25" && subs >= "10" )
                dp[i] = dp[i-1] + 1;
            else
                dp[i] = dp[i-1];
        }

        return dp[n-1]+1;
    }
};
#pragma endregion

#pragma region 剑指48.最长不重复子字符串
class Solution_42 {
public:
    int lengthOfLongestSubstring(string s) {

        int n = s.size();
        int max{0};
        int leftIndex{0};
        unordered_set<char> sub{s[0]};
        for(int i = 1; i < n; ++i)
        {
            if(!sub.count(s[i]))
            {
                sub.insert(s[i]);
                max = std::max(max,(int)sub.size());
            }
            else
            {
                bool flag = true;
                while(flag)
                {
                    sub.erase(s[leftIndex]);
                    if(s[leftIndex] == s[i])
                        flag = false;
                    ++leftIndex;
                }
                sub.insert(s[i]);
                max = std::max(max,(int)sub.size());
            }
        }

        return max;
    }
};
#pragma endregion

#pragma region 剑指49.丑数
class Solution_43 {
public:
    int nthUglyNumber(int n) {
        int p2{1},p3{1},p5{1};
        int u2{1},u3{1},u5{1};
        int p{0};
        int size = n+1;
        vector<int> ugly(n+1);

        for(int p = 1; p < n; ++p)
        {
            int min = std::min(u2,std::min(u3,u5));
            ugly[p] = min;
            cout<<min<<endl;
            if(min == u2)
            {
                u2 = ugly[p2] *2;
                ++p2;
            }
            if(min == u3)
            {
                u3 = ugly[p3]*3;
                ++p3;
            }
            if(min == u5)
            {
                u5 = ugly[p5]*5;
                ++p5;
            }
        }

        return ugly[n];
    }
};
#pragma endregion

#pragma region 剑指60.n个骰子点数
class Solution_44 {
public:
    vector<double> dicesProbability(int n) {
        vector<double> dp(6,1.0/6.0);
        for(int i = 2; i <= n; ++i)
        {
            vector<double> tmp(5*i + 1,0);
            for(int j = 0; j < dp.size(); ++j)
            {
                for(int k = 0; k < 6; ++k)
                {
                    tmp[j+k] += dp[j] / 6.0;
                }
            }
            dp = tmp;
        }
        return dp;
    }
};
#pragma endregion

#pragma region 棋子阵营对调
class Solution_45
{
public:
    int getMaxPower(vector<int>& powers, string& camps)
    {
        int size = powers.size();
        int max = INT_MIN;
        vector<int> dp_f(size,0);
        vector<int> dp_b(size,0);

        dp_f[0] = camps[0] == 'A' ? -powers[0] : powers[0];
        max = std::max(max,dp_f[0]);
        for(int i = 1; i < size; ++i)
        {
            int t = camps[i] == 'A' ? -powers[i] : powers[i];
            dp_f[i] = dp_f[i-1] + t;
            max = std::max(max,dp_f[i]);
        }

        dp_b[size-1] = camps[size-1] == 'A' ? -powers[size-1] : powers[size-1];
        max = std::max(max,dp_b[size-1]);
        for(int i = size - 2; i >= 0; --i)
        {
            int t = camps[i] == 'A' ? -powers[i] : powers[i];
            dp_b[i] = dp_b[i+1] + t;
            max = std::max(max,dp_b[i]);
        }

        return max;
    }
};
#pragma endregion


class Solution_46
{
public:
    void memmove(int* dest, int* src, int n)
    {
        auto p1{dest},p2{src};
        for(int i = 0; i < n; ++i)
        {
            *(dest+i) = *(src+i);
        }
    }
    
};


int main_test_4()
{
    int n;
    cin>>n;

    int* array = new int[n];
    for(int i = 0; i < n; ++i)
    {
        int input;
        cin>>input;
        array[i] = input;
    }
    int dest,src,size;
    cin>>dest;
    cin>>src;
    cin>>size;

    Solution_46 solu{};
    solu.memmove(array+dest,array+src,size);

    for(int i = 0; i < n-1; ++i)
    {
        cout<<*(array+i)<<" ";
    }
    cout<<array[n-1];

    return 0;
}


class Solution
{
public:
    int getArea(vector<pair<int,int>>& points)
    {
        int xstart{-1},xend{50001},ystart{-1},yend{50001};
        int prevx,prevyStart,prevyEnd;
        int n = points.size();
        for(int i = 0; i < n-1; i+=2)
        {
            if(points[i].first != points[i+1].first)
                return 0;

            if(i != 0)
            {
                
            }
        }
        
        return 0;
    }
};

int main()
{
    vector<pair<int,int>> points{};
    int x,y;
    while(cin>>x>>y)
    {
        points.emplace_back(x,y);
    }

    sort(points.begin(),points.end());
    Solution solu{};
    cout<<solu.getArea(points);
    
    return 0;
}




