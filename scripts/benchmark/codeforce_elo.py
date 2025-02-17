import pandas as pd
import numpy as np
import requests
import time

def expected_score(Ra, Rb):
    """Compute the expected score of Ra against Rb"""
    return 1 / (1 + 10 ** ((Rb - Ra) / 400))

def get_contest_ratings(contest_id):
    """Fetch contest standings and ratings from Codeforces API"""
    API_URL = "https://codeforces.com/api/contest.standings?contestId={}&from=1&count=100"
    try:
        response = requests.get(API_URL.format(contest_id))
        data = response.json()
        if data["status"] == "OK":
            return [
                (user["party"]["members"][0]["handle"], user["party"]["oldRating"])
                for user in data["result"]["rows"]
                if "oldRating" in user["party"]
            ]
    except Exception as e:
        print(f"Error fetching data for contest {contest_id}: {e}")
    return []

def calculate_llm_rating(dataset_path, solved_problems, initial_rating=1500, K=32):
    """
    Calculate LLM's rating progression using real Codeforces contest data and solved problems.
    
    Returns:
        final_rating (float): The final Elo rating of the LLM.
        rating_df (pd.DataFrame): A DataFrame showing the rating progression.
    """
    df = pd.read_csv(dataset_path)
    llm_rating = initial_rating
    rating_history = []
    
    for contest_id, contest_problems in df.groupby("contest_id"):
        contest_ratings = get_contest_ratings(contest_id)
        if not contest_ratings:
            continue  # Skip if no ratings available
        
        human_ratings = [r for _, r in contest_ratings]
        if len(human_ratings) == 0:
            continue  # Skip if no valid ratings
        
        # Determine which problems were solved by the LLM in this contest
        solved_in_contest = contest_problems[
            contest_problems["problem_name"].isin(solved_problems)
        ]
        
        # Compute rank: assuming that more solved problems implies a better rank.
        # Here we use a simple formula: rank = total problems - problems solved + 1.
        rank = len(contest_problems) - len(solved_in_contest) + 1
        total_participants = len(human_ratings)
        
        # Calculate expected score and actual score based on rank
        E = np.mean([expected_score(llm_rating, R) for R in human_ratings])
        S = 1 - (rank - 1) / total_participants  
        rating_change = K * (S - E)
        llm_rating += rating_change
        
        rating_history.append((contest_id, llm_rating, E, S, rating_change))
        
        time.sleep(1)  # Avoid API rate limits
    
    rating_df = pd.DataFrame(
        rating_history, 
        columns=["contest_id", "llm_rating", "expected_score", "actual_score", "rating_change"]
    )
    rating_df.to_csv("llm_rating_progression.csv", index=False)
    
    # Print the rating progression
    print("LLM Rating Progression:")
    print(rating_df)
    
    return llm_rating, rating_df

# Example usage:
# final_rating, progression = calculate_llm_rating("contests.csv", solved_problems=["Problem A", "Problem B"])
# print("Final Elo Rating:", final_rating)