from rllm.rewards.code_reward import RewardCodeFn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardType

if __name__ == "__main__":
    #test the codecontest
    model_response = """
    import sys
    from itertools import permutations
    def main():
        # Read input
        N, M, R = map(int, sys.stdin.readline().split())
        r = list(map(int, sys.stdin.readline().split()))
        A, B, C = [], [], []
        for _ in range(M):
            a, b, c = map(int, sys.stdin.readline().split())
            A.append(a)
            B.append(b)
            C.append(c)

        # Initialize distance matrix
        INF = float('inf')
        dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
        for i in range(1, N+1):
            dist[i][i] = 0

        # Set initial distances
        for i in range(M):
            a, b, c = A[i], B[i], C[i]
            dist[a][b] = c
            dist[b][a] = c

        # Floyd-Warshall algorithm
        for k in range(1, N+1):
            for i in range(1, N+1):
                for j in range(1, N+1):
                    if dist[i][k] != INF and dist[k][j] != INF:
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        # Generate all permutations of R towns
        min_dist = INF
        for perm in permutations(r):
            total = 0
            for i in range(R-1):
                total += dist[perm[i]][perm[i+1]]
            if total < min_dist:
                min_dist = total

        # Output the minimum distance
        print(min_dist)

    if __name__ == "__main__":
        main()
    """

    public_tests= {"input": ["3\n4 5\n6 3\n10 2\n"], "output": ["5\n3 4\n4 4 1 2\n"]}
    metadata = {
        "public_tests": public_tests,
        "dataset_flag": "code_contests",
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata)
    output = reward(input)
    print(f"codetest output:{output}")