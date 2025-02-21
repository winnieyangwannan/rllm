from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn

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

if __name__ == "__main__":
    print(test_reward_livecodebench_leetcode())
    print(test_reward_livecodebench())
    # print(test_reward_taco())
    # print(test_reward_codeforces())
    # print(test_reward_code_contests())
