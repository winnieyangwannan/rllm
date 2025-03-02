#!/usr/bin/env python3
import os
import requests
import bisect
import json
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from codeforces.sorted_ratings import SORTED_RATINGS

def get_percentile(rating: float, sorted_ratings: List[float]) -> float:
    """Calculate the percentile of a given rating."""
    idx = bisect.bisect_left(sorted_ratings, float(rating))
    return round(idx / len(sorted_ratings) * 100, 1)

def percentile_to_codeforces_rating(percentile: float) -> int:
    """Convert a percentile (0-100) into an approximate Codeforces Elo rating."""
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100.")
    
    normalized_percentile = percentile / 100.0
    base_rating = 800  # Starting rating for Newbies
    max_rating_increase = 2200  # Range increase from base
    k = 2  # Curve shaping factor
    
    rating = base_rating + max_rating_increase * (normalized_percentile ** k)
    return round(rating)

def calc_elo_rating(contest_id: int, problem_status: Dict[str, List[bool]], sorted_ratings: List[float]) -> Optional[Tuple[int, float]]:
    """Calculate the Elo rating for a given contest based on problem status."""
    try:
        # Fetch contest data from Codeforces API
        standings_response = requests.get(f"https://codeforces.com/api/contest.standings?contestId={contest_id}&showUnofficial=false", timeout=10)
        standings = standings_response.json()
        
        rating_changes_response = requests.get(f"https://codeforces.com/api/contest.ratingChanges?contestId={contest_id}", timeout=10)
        rating_changes = rating_changes_response.json()
        
        # Process and validate data
        handle_set: Set[str] = set()
        try:
            handle_set_standings = set(
                standings["result"]["rows"][i]["party"]["members"][0]["handle"] 
                for i in range(len(standings["result"]["rows"]))
            )
            
            handle_set_ratings = set(
                rating_changes["result"][i]["handle"] 
                for i in range(len(rating_changes["result"]))
            )
            
            handle_set = handle_set_standings.intersection(handle_set_ratings)
            
            standings["result"]["rows"] = [
                row for row in standings["result"]["rows"] 
                if row["party"]["members"][0]["handle"] in handle_set
            ]
            
            rating_changes["result"] = [
                change for change in rating_changes["result"] 
                if change["handle"] in handle_set
            ]
            
            assert len(standings["result"]["rows"]) == len(rating_changes["result"]) and len(standings["result"]["rows"]) > 200
        except Exception:
            return None
        
        # Validate results
        if ("result" not in standings or 
            "result" not in rating_changes or 
            len(standings["result"]["rows"]) != len(rating_changes["result"]) or 
            len(standings["result"]["rows"]) <= 200):
            return None
        
        # Find maximum rating
        max_rating = max(change["oldRating"] for change in rating_changes["result"])
        
        # Calculate score and penalty
        score = 0
        penalty = 0
        
        for problem in standings["result"]["problems"]:
            prob = f"{problem['contestId']}{problem['index']}"
            if prob in problem_status:
                for ith, status in enumerate(problem_status[prob]):
                    if status is True:
                        if "points" in problem:
                            score += max(0, problem["points"] - 50 * ith)
                        else:
                            score += 1
                            penalty += ith * 10
                        break
        
        # Calculate rank
        n = len(standings["result"]["rows"])
        
        rank = n
        for i in range(n):
            if (standings["result"]["rows"][i]["points"] < score or 
                (standings["result"]["rows"][i]["points"] == score and 
                 standings["result"]["rows"][i]["penalty"] > penalty)):
                rank = i
                break
        
        # Binary search for rating
        l, r = 0, max_rating + 100
        while r - l > 1:
            mid = (l + r) // 2
            new_seed = 1
            for i in range(n):
                new_seed += 1 / (1 + 10 ** ((mid - rating_changes["result"][i]["oldRating"]) / 400))
            if new_seed < rank:
                r = mid
            else:
                l = mid
        
        percentile = get_percentile(l, sorted_ratings)
        return l, percentile
    
    except Exception:
        return None

def format_grouped_contest_data(submissions: List[List[bool]], problem_ids: List[str]) -> List[Tuple[int, Dict[str, List[bool]]]]:
    """
    Groups problems by contest ID (including problem letters like A1) into a list of tuples.
    """
    if len(submissions) != len(problem_ids):
        raise ValueError("Length of submissions and problem_ids must be the same.")
    
    grouped_data = defaultdict(dict)
    
    for problem_id, submission in zip(problem_ids, submissions):
        # Extract contest ID using regex to capture leading digits
        match = re.match(r'(\d+)([A-Z].*)', problem_id)
        if not match:
            raise ValueError(f"Invalid problem ID format: {problem_id}")
        
        contest_id = int(match.group(1))  # Numeric part as contest ID
        problem_letter = match.group(0)   # Full problem ID (contest ID + letter part)
        
        # Group problems under their corresponding contest ID
        grouped_data[contest_id][problem_letter] = submission
    
    # Convert to the required list of tuples format
    combined_data = [(contest_id, problems) for contest_id, problems in grouped_data.items()]
    
    return combined_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate Codeforces percentile based on problem submissions')
    parser.add_argument('--results_path', required=True, help='Path to the results JSON file')
    
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, "./codeforces/metadata_cf.json")
    results_path = os.path.abspath(args.results_path)
    
    # Load required files
    try:
        # Load results
        with open(results_path, 'r') as file:
            results = json.load(file)
        
        # Load metadata
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return
    
    # Process the data
    try:
        # Format the data
        model_results = format_grouped_contest_data(results, metadata)
        
        # Calculate Elo ratings for each contest with progress bar
        contest_elos = []
        for contest_id, problems in tqdm(model_results, desc="Processing contests"):
            elo_result = calc_elo_rating(contest_id, problems, SORTED_RATINGS)
            if elo_result is not None:
                contest_elos.append((contest_id, elo_result))
        
        # Calculate average percentile
        percentiles = [elo[1][1] for elo in contest_elos if elo[1] is not None]
        
        if not percentiles:
            print("No valid percentiles calculated.")
            return
        
        avg_percentile = sum(percentiles) / len(percentiles)
        estimated_rating = percentile_to_codeforces_rating(avg_percentile)
        
        # Display results
        print("\n" + "="*50)
        print("CODEFORCES PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Average Percentile: {avg_percentile:.1f}%")
        print(f"Estimated Codeforces Rating: {estimated_rating}")
        print(f"Contests Processed: {len(contest_elos)}")
        print("="*50)
        
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()