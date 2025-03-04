from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn
import json
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def test_reward_code_contests():
    model_response = """
```python
import sys
from itertools import permutations
def main():
    N, M, R = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    A, B, C = [], [], []
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        C.append(c)
    INF = float('inf')
    dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        dist[i][i] = 0
    for i in range(M):
        a, b, c = A[i], B[i], C[i]
        dist[a][b] = c
        dist[b][a] = c
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    min_dist = INF
    for perm in permutations(r):
        total = 0
        for i in range(R-1):
            total += dist[perm[i]][perm[i+1]]
        if total < min_dist:
            min_dist = total
    print(min_dist)
if __name__ == "__main__":
    main()
    ```
    """
    metadata = {
        "input": [
            # Test case 1: Simple path with 3 cities
            "4 3 3\n1 2 3\n1 2 3\n2 3 2\n3 4 4\n",
            # Test case 2: Complete graph with 5 cities
            "5 10 4\n1 2 3 4\n1 2 5\n1 3 5\n1 4 5\n1 5 5\n2 3 5\n2 4 5\n2 5 5\n3 4 5\n3 5 5\n4 5 5\n"
            # Test case 3: Larger graph with 7 cities
            "7 21 4\n1 3 5 7\n1 2 4\n1 3 8\n1 4 1\n1 5 7\n1 6 3\n1 7 9\n2 3 5\n2 4 2\n2 5 6\n2 6 8\n2 7 4\n3 4 7\n3 5 9\n3 6 1\n3 7 6\n4 5 3\n4 6 5\n4 7 8\n5 6 2\n5 7 4\n6 7 7\n"
        ],
        "output": [
            "5\n",  
            "15\n",
            "11\n"
        ]
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="code_contests")
    output = reward(input)
    assert output.is_correct == True
    return output


def test_reward_codeforces():
    model_response = """
```python
import sys
from itertools import permutations
def main():
    n,m=map(int, input().split())
    a=sum(list(map(int, input().split())))
    if a+(n-1)*10<=m:
        print((m-a)//5)
    else:
        print(-1)
if __name__ == "__main__":
    main()
    ```
    """
    metadata = [
            # Basic case
            {"input": "3 30\n2 2 1", "output": "5"},
            # Impossible case
            {"input": "3 20\n2 1 1", "output": "-1"},
            # Exact fit case
            {"input": "4 45\n5 5 5 5", "output": "-1"},
            # Large numbers
            {"input": "5 100\n10 10 10 10 10", "output": "10"},
            # Single task
            {"input": "1 20\n5", "output": "3"},
            # Maximum possible breaks
            {"input": "2 100\n1 1", "output": "19"},
            # Edge case - just barely possible
            {"input": "3 35\n5 5 5", "output": "4"}
        ]
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="codeforces")
    output = reward(input)
    assert output.is_correct == True
    return output


def test_reward_swebench():
    reward = RewardCodeFn(RewardConfig)
    tests = {
        "instance_id": "astropy__astropy-12907",
    }
    metadata = tests
    model_response = """\
diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -242,7 +242,7 @@ def _cstack(left, right):
        cright = _coord_matrix(right, 'right', noutp)
    else:
        cright = np.zeros((noutp, right.shape[1]))
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right

    return np.hstack([cleft, cright])
    """
    input = RewardInput(
        problem="""
Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
    [False,  True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True,  True, False, False],
    [ True,  True, False, False],
    [False, False,  True, False],
    [False, False, False,  True]])
```

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True,  True, False, False],
    [ True,  True, False, False],
    [False, False,  True,  True],
    [False, False,  True,  True]])
```
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?
""",
        problem_type=RewardType.CODE,
        model_response=model_response,
        metadata=metadata,
        data_source="swebench",
    )
    output = reward(input)
    assert output.is_correct == True
    return output

def test_reward_taco():
    model_response = """
```python
import sys
from itertools import permutations
def main():
    N, M, R = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    A, B, C = [], [], []
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        C.append(c)
    INF = float('inf')
    dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        dist[i][i] = 0
    for i in range(M):
        a, b, c = A[i], B[i], C[i]
        dist[a][b] = c
        dist[b][a] = c
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    min_dist = INF
    for perm in permutations(r):
        total = 0
        for i in range(R-1):
            total += dist[perm[i]][perm[i+1]]
        if total < min_dist:
            min_dist = total
    print(min_dist)
if __name__ == "__main__":
    main()
    ```
    """
    metadata = {
        "inputs": [
            # Test case 1: Simple path with 3 cities
            "4 3 3\n1 2 3\n1 2 3\n2 3 2\n3 4 4\n",
            # Test case 2: Complete graph with 5 cities
            "5 10 4\n1 2 3 4\n1 2 5\n1 3 5\n1 4 5\n1 5 5\n2 3 5\n2 4 5\n2 5 5\n3 4 5\n3 5 5\n4 5 5\n"
            # Test case 3: Larger graph with 7 cities
            "7 21 4\n1 3 5 7\n1 2 4\n1 3 8\n1 4 1\n1 5 7\n1 6 3\n1 7 9\n2 3 5\n2 4 2\n2 5 6\n2 6 8\n2 7 4\n3 4 7\n3 5 9\n3 6 1\n3 7 6\n4 5 3\n4 6 5\n4 7 8\n5 6 2\n5 7 4\n6 7 7\n"
        ],
        "outputs": [
            "5\n",  
            "15\n",
            "11\n"
        ]
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="taco")
    output = reward(input)
    assert output.is_correct == True
    return output


def test_reward_livecodebench():
    model_response = """
Yes of course!
```python
import json

def main():
    n = input()
    phone_numbers = input().strip().split()
    seen = set()
    duplicates = set()
    for number in phone_numbers:
        if number in seen:
            duplicates.add(number)
        else:
            seen.add(number)
    
    print(len(duplicates)+1)
if __name__ == "__main__":
    main()
```
""" 
    public_test_case = [
        {
            'input': '3\n12345 530391 12345\n',
            'output': '2\n',
            'testtype': 'stdin'
        }
    ]
    metadata = public_test_case
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="livecodebench")
    output = reward(input)
    assert output.is_correct == True
    return output

def test_reward_livecodebench_leetcode():
    model_response = """
Yes of course!
```python
class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        count = 0
        for hour in hours:
            if hour >= target:
                count += 1
        return count
```
""" 
    public_test_case = [
        {
            "input": "[5, 3, 10, 8, 2]\n5",
            "output": "3",
            "testtype": "functional"
        }
    ]
    metadata = public_test_case
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="livecodebench")
    output = reward(input)
    assert output.is_correct == True
    return output

def test_reward_leetcode():
    model_response = """
Here is my response
```python
class Solution:\n    def minOperations(self, nums: List[int], k: int) -> int:\n        is_added = [False] * k\n        count = 0\n        n = len(nums)\n        for i in range(n - 1, -1, -1):\n            if nums[i] > k or is_added[nums[i] - 1]:\n                continue\n            is_added[nums[i] - 1] = True\n            count += 1\n            if count == k:\n                return n - i\n
```
"""
    tests = {
        "functional": "def check(candidate):\n    assert candidate(nums = [3,1,5,4,2], k = 2) == 4\n    assert candidate(nums = [3,1,5,4,2], k = 5) == 5\n    assert candidate(nums = [3,2,5,3,1], k = 3) == 4\n\n\ncheck(Solution().minOperations)"
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="leetcode")
    output = reward(input)
    assert output.is_correct == True
    return output

def test_reward_leetcode_bad():
    model_response = """
Here is my bad response, it is not in markdown oops
class Solution:\n    def minOperations(self, nums: List[int], k: int) -> int:\n        is_added = [False] * k\n        count = 0\n        n = len(nums)\n        for i in range(n - 1, -1, -1):\n            if nums[i] > k or is_added[nums[i] - 1]:\n                continue\n            is_added[nums[i] - 1] = True\n            count += 1\n            if count == k:\n                return n - i\n
"""
    tests = {
        "functional": "def check(candidate):\n    assert candidate(nums = [3,1,5,4,2], k = 2) == 4\n    assert candidate(nums = [3,1,5,4,2], k = 5) == 5\n    assert candidate(nums = [3,2,5,3,1], k = 3) == 4\n\n\ncheck(Solution().minOperations)"
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="leetcode")
    output = reward(input)
    assert output.is_correct == False
    return output 

def test_reward_leetcode_good_1():

    model_response = """
To solve this problem, we need to identify the smallest subarray that, when sorted, will result in the entire array being sorted. Here's a step-by-step approach to achieve this:\n\n1. **Identify the Sorted and Unsorted Parts**: Compare the original array with its sorted version to identify the first and last positions where the elements differ. These positions mark the boundaries of the unsorted subarray.\n\n2. **Edge Case**: If the array is already sorted, the length of the subarray to be sorted is 0.\n\nHere's the complete implementation:\n\n```python\nfrom typing import List\n\n
class Solution:\n    def findUnsortedSubarray(self, nums: List[int]) -> int:\n        # Create a sorted version of the array\n        sorted_nums = sorted(nums)\n        \n        # Initialize the start and end indices\n        start, end = -1, -2\n        \n        # Find the first and last positions where the elements differ\n        for i in range(len(nums)):\n            if nums[i] != sorted_nums[i]:\n                if start == -1:\n                    start = i\n                end = i\n        \n        # The length of the subarray to be sorted\n        return end - start + 1\n```\n\n### Explanation:\n\n- **Sorting**: We first sort the array and store it in `sorted_nums`.\n- **Finding Boundaries**: We iterate through the original array `nums` and compare it with `sorted_nums`. The first mismatch gives us the `start` of the unsorted subarray, and the last mismatch gives us the `end`.\n- **Calculating Length**: The length of the subarray to be sorted is `end - start + 1`. If the array is already sorted, `start` will remain `-1` and `end` will remain `-2`, resulting in a length of 0.\n\nThis approach has a time complexity of O(n log n) due to the sorting step, and a space complexity of O(n) for storing the sorted array.", "split": "train"}}

"""
    tests = {
        "functional": "def check(candidate):\n    assert candidate(nums = [2,6,4,8,10,9,15]) == 5\n    assert candidate(nums = [1,2,3,4]) == 0\n    assert candidate(nums = [1]) == 0\n\n\ncheck(Solution().findUnsortedSubarray)"
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="leetcode")
    output = reward(input)
    assert output.is_correct == True
    return output

def test_reward_leetcode_good_2():

    model_response = """
To solve this problem with a time complexity of \\(O(\\log(m+n))\\), we can use a binary search approach. The idea is to partition the two arrays such that the left half contains the smaller elements and the right half contains the larger elements. The median will then be the maximum of the left half if the total number of elements is odd, or the average of the maximum of the left half and the minimum of the right half if the total number of elements is even.\n\nHere's the complete code:\n\n```python\nfrom typing import List\n\n
class Solution:\n    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:\n        # Ensure nums1 is the smaller array\n        if len(nums1) > len(nums2):\n            nums1, nums2 = nums2, nums1\n        \n        x, y = len(nums1), len(nums2)\n        low, high = 0, x\n        \n        while low <= high:\n            partitionX = (low + high) // 2\n            partitionY = (x + y + 1) // 2 - partitionX\n            \n            # If partitionX is 0 it means nothing is there on left side. Use -inf for maxLeftX\n            # If partitionX is length of input then there is nothing on right side. Use +inf for minRightX\n            maxLeftX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]\n            minRightX = float('inf') if partitionX == x else nums1[partitionX]\n            \n            maxLeftY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]\n            minRightY = float('inf') if partitionY == y else nums2[partitionY]\n            \n            if maxLeftX <= minRightY and maxLeftY <= minRightX:\n                # We have partitioned array at correct place\n                if (x + y) % 2 == 0:\n                    return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2\n                else:\n                    return max(maxLeftX, maxLeftY)\n            elif maxLeftX > minRightY:\n                # We are too far on right side for partitionX. Go on left side.\n                high = partitionX - 1\n            else:\n                # We are too far on left side for partitionX. Go on right side.\n                low = partitionX + 1\n```\n\n### Explanation:\n1. **Ensure `nums1` is the smaller array**: This helps in minimizing the binary search range.\n2. **Binary Search**: We perform a binary search on the smaller array (`nums1`).\n3. **Partitioning**: For each partition of `nums1`, we calculate the corresponding partition of `nums2` such that the total number of elements on the left side is equal to or one more than the right side.\n4. **Check Conditions**: We check if the maximum element on the left side of `nums1` is less than or equal to the minimum element on the right side of `nums2` and vice versa.\n5. **Calculate Median**: If the partitions are correct, we calculate the median based on whether the total number of elements is odd or even.\n\nThis approach ensures that we find the median in \\(O(\\log(\\min(m, n)))\\) time complexity.", "split": "train"}}
"""
    tests = {
        "functional": "def check(candidate):\n    assert candidate(nums1 = [1,3], nums2 = [2]) == 2.00000\n    assert candidate(nums1 = [1,2], nums2 = [3,4]) == 2.50000\n\n\ncheck(Solution().findMedianSortedArrays)"
    }

    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="leetcode")
    output = reward(input)
    assert output.is_correct == True
    return output

def _process_case_leetcode(i, entry):
    model_response = f"""
```python
{entry["completion"]}
```
"""
    tests = entry["tests"]
    reward = RewardCodeFn(RewardConfig)
    input_obj = RewardInput(
        problem="",
        problem_type=RewardType.CODE,
        model_response=model_response,
        metadata=tests,
        data_source="leetcode"
    )
    output = reward(input_obj)
    failed = None
    if not output.is_correct:
        failed = {
            "index": i,
            "model_response": model_response,
            "tests": tests,
        }
    return i, output, failed

def test_reward_leetcode_batch(count=10, all=False):
    data_path = os.path.expanduser("~/rllm2/rllm/data/train/code/leetcode.json")
    failure_log_path = os.path.expanduser("~/rllm2/failed_tests.json")
    with open(data_path, "r") as f:
        data = json.load(f)
    failed_cases = []
    if all is True:
        count = len(data)

    results = {}
    failed_cases = []

    # Create a process pool and submit tasks
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_process_case_leetcode, i, data[i]): i for i in range(count)}
        for future in tqdm(as_completed(futures), total=count):
            i, output, failed = future.result()
            results[i] = output
            if failed is not None:
                failed_cases.append(failed)

    # Save failed cases to a JSON file if any
    if failed_cases:
        with open(failure_log_path, "w") as f:
            json.dump(failed_cases, f, indent=4)

    # Return the output corresponding to the last processed index
    return results[count - 1]

def test_lcb_again():
    model_response = """
<think>
Okay, I need to solve this problem where I have to find the number of days in a given year Y based on the Gregorian calendar rules. Let's think about how to approach this.

First, the problem statement says that the number of days depends on whether Y is a leap year or not. So I need to determine if Y is a leap year according to the rules given.

Let's recap the rules:

1. If Y is not a multiple of 4, then it's a common year with 365 days.
2. If Y is a multiple of 4 but not a multiple of 100, then it's a leap year with 366 days.
3. If Y is a multiple of 100 but not a multiple of 400, then it's a common year with 365 days.
4. If Y is a multiple of 400, then it's a leap year with 366 days.

So the steps I need to take are:

- Check if Y is divisible by 400. If yes, then it's a leap year (366 days).
- Else, check if it's divisible by 100. If yes, then it's not a leap year (365 days).
- Else, check if it's divisible by 4. If yes, then it's a leap year (366 days).
- Else, it's a common year (365 days).

Wait, maybe I should structure this as a series of conditions.

Another way: Let's compute whether the year is a leap year based on these conditions.

Let me think about the logic:

A year is a leap year if:
- It is divisible by 400, OR
- It is divisible by 4 but not by 100.

So the condition is: (Y % 400 == 0) or (Y % 4 == 0 and Y % 100 != 0).

So that's the condition we can use to determine if the year is a leap year.

If this condition is true, then the number of days is 366. Else, it's 365.

So the plan is:

1. Read Y as an integer.
2. Check if Y is a leap year using the condition above.
3. If it's a leap year, print 366. Else, print 365.

Let's test this logic with the sample inputs.

Sample Input 1: 2023
Check conditions: 2023 % 400 is 2023-5*400=2023-2000=23, so no. Then check if divisible by 100: 2023 is not. So since it's not divisible by 400 and not by 100, but wait wait no: wait the condition is (divisible by 400) OR (divisible by 4 and not by 100). Wait for 2023: it's not divisible by 4, so the first part is false. The second part is Y%4 ==0? 2023 mod 4 is 2023/4 is 505*4=2020, remainder 3. So no. So condition is false. So it's not a leap year. So output is 365. Which matches sample output 1.

Sample Input 2: 1992
Check: 1992 is divisible by 4. Let's see if it's not divisible by 100. Yes. So condition is true. So it's a leap year, output 366. Which matches sample 2.

Sample Input 3: 1800
Check: 1800 is divisible by 100. So check if it's divisible by 400. No. So condition is false. So not a leap year. So output 365. Which matches sample 3.

Sample Input 4: 1600
Check: 1600 is divisible by 400. So condition is true. Output 366. Which matches sample 4.

Great, so the logic seems correct.

Now, in code, how to implement this.

Read Y from input. Then:

is_leap = (Y % 400 == 0) or (Y % 4 == 0 and Y % 100 != 0)

Then, if is_leap, print 366, else print 365.

Wait, wait. Wait, let me think again.

Wait, the condition is: leap year if (Y is divisible by 400) OR (divisible by 4 and not by 100). So the code should be:

if (Y % 400 == 0) or (Y % 4 == 0 and Y % 100 != 0):
    days = 366
else:
    days = 365

Yes.

So now, code structure:

Read Y as integer.

Compute the condition.

Print 366 or 365 accordingly.

So the code will be something like:

y = int(input())
if (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0):
    print(366)
else:
    print(365)

Let me test this with sample inputs.

Sample Input 1: 2023. 2023%400 is 2023-5*400=2023-2000=23. So first condition is false. Then check if 2023%4 ==0? 2023/4 is 505.75, remainder 3. So condition is false. So print 365.

Sample Input 2:1992. 1992%400 is 392, so first condition false. But 1992%4 is 0. 1992%100 is 92, not zero. So condition is true. Print 366.

Sample Input3:1800. 1800%400= 1800-4*400= 1800-1600=200. Not zero. Then check if 1800%4 is zero: yes. Then check if 1800%100 is zero? Yes. So condition is (1800%400 !=0) and (1800%4==0 and 1800%100 !=0) → since 1800%100 is zero, the second part is false. So overall condition is false. So print 365.

Sample Input4:1600. 1600%400=0 → condition is true. So print 366.

All samples pass.

Thus, the code should work.
</think>

To solve this problem, we need to determine the number of days in a given year Y based on the rules of the Gregorian calendar. The number of days in a year can be either 365 or 366 depending on whether the year is a leap year or not.

### Approach
The problem can be broken down into checking if the year Y is a leap year using the following rules:
1. If Y is a multiple of 400, it is a leap year.
2. If Y is a multiple of 100 but not a multiple of 400, it is not a leap year.
3. If Y is a multiple of 4 but not a multiple of 100, it is a leap year.
4. Otherwise, it is a common year.

We can simplify this into a single condition to check if Y is a leap year:
- A year is a leap year if it is divisible by 400 or if it is divisible by 4 but not by 100.

Using this condition, we can determine if the year is a leap year and thus calculate the number of days.

### Solution Code
```python
y = int(input())
if (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0):
    print(366)
else:
    print(365)
```

### Explanation
1. **Reading Input:** The year Y is read as an integer from the input.
2. **Leap Year Check:** We check if Y is a leap year using the condition `(y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)`.
3. **Determine Days:** If Y is a leap year, print 366. Otherwise, print 365.

This approach efficiently determines the number of days in the year by leveraging a single conditional check, ensuring optimal performance and correctness.

    """
    tests = [{"input": "2023\n", "output": "365\n", "testtype": "stdin"}, {"input": "1992\n", "output": "366\n", "testtype": "stdin"}, {"input": "1800\n", "output": "365\n", "testtype": "stdin"}, {"input": "1600\n", "output": "366\n", "testtype": "stdin"}]
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="livecodebench")
    output = reward(input)
    assert output.is_correct == True
    return output


if __name__ == "__main__":
    # print(test_lcb_again())
    # print(test_reward_leetcode())
    # print(test_reward_leetcode_bad())
    # print(test_reward_leetcode_good_1())
    # print(test_reward_leetcode_good_2())
    # # print(test_reward_leetcode())
    # # print(test_reward_leetcode_bad())
    print(test_reward_leetcode_batch(all=True))
    # print(test_reward_livecodebench())
    # print(test_reward_taco())
    # print(test_reward_codeforces())
    # print(test_reward_code_contests())
