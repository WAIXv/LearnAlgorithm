#include <iostream>
#include <map>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace std;

struct TreeNode {
	char c;
	struct TreeNode* left;
	struct TreeNode* right;
	TreeNode(int x) :
		c(x), left(nullptr), right(nullptr) {}
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
class Node {
public:
	int val;
	Node* next;
	Node* random;

	Node(int _val) {
		val = _val;
		next = nullptr;
		random = nullptr;
	}
};

class Solution {
public:
	Node* copyRandomList(Node* head) {
		Node* ret = new Node(head->val);
        

        return ;
	}
};
#pragma endregion

int main()
{
    Solution_3 solu = Solution_3();
    ListNode l1(1);
	ListNode l2(2);
	ListNode l3(3);
	ListNode l4(4);
	ListNode l5(5);

    l1.next = &l2;
    l2.next = &l3;
	l3.next = &l4;
	l4.next = &l5;



    solu.reverseList(&l1);
    return 0;
}