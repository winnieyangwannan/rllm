import asyncio
import os
import random
import sys

from datasets import load_dataset
from dotenv import load_dotenv
from mcp_client import MCPClient

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.search_reward import rllm_reward_fn_search

load_dotenv()


def load_search_r1_data(n=1, train_size=3000, test_size=100):
    if DatasetRegistry.dataset_exists("search_r1_combined", "test"):
        test_dataset = DatasetRegistry.load_dataset("search_r1_combined", "test")
        return test_dataset.get_data()

    print("Loading HotpotQA dataset...")

    hotpot_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)
    hotpot_train = hotpot_dataset["train"]
    hotpot_val = hotpot_dataset["validation"]

    print("Loading Natural Questions dataset...")

    nq_dataset = load_dataset("sentence-transformers/natural-questions", "pair")
    nq_train = nq_dataset["train"]

    hotpot_train_subset = hotpot_train.select(range(min(train_size // 2, len(hotpot_train))))
    hotpot_val_subset = hotpot_val.select(range(min(test_size // 2, len(hotpot_val))))
    nq_subset = nq_train.select(range(min(train_size // 2, len(nq_train))))

    def process_hotpot_example(example, idx, split):
        question = example["question"]
        ground_truth = example["answer"]
        data_source = "hotpotqa"

        task = {"question": question, "ground_truth": ground_truth, "data_source": data_source}

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": f"Please answer the following question by searching for relevant information: {question}"}],
            "ability": "multi-hop-reasoning",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": split, "index": idx, "task": task, "tools": ["google_search"], "uid": f"hotpot_{example.get('id', idx)}", "question_type": example.get("type", "bridge"), "level": example.get("level", "medium")},
            "task": task,
            "uid": f"hotpot_{example.get('id', idx)}",
        }

    def process_nq_example(example, idx, split):
        question = example["query"]
        ground_truth = example["answer"][:200] + "..." if len(example["answer"]) > 200 else example["answer"]
        data_source = "natural_questions"

        task = {"question": question, "ground_truth": ground_truth, "data_source": data_source}

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": f"Please answer the following question by searching for relevant information: {question}"}],
            "ability": "fact-retrieval",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": split, "index": idx, "task": task, "tools": ["google_search"], "uid": f"nq_{idx}", "question_type": "factual", "level": "easy"},
            "task": task,
            "uid": f"nq_{idx}",
        }

    print("Processing HotpotQA training data...")
    hotpot_train_processed = [process_hotpot_example(example, idx, "train") for idx, example in enumerate(hotpot_train_subset)]

    print("Processing HotpotQA validation data...")
    hotpot_val_processed = [process_hotpot_example(example, idx, "test") for idx, example in enumerate(hotpot_val_subset)]

    print("Processing Natural Questions data...")
    nq_processed = [process_nq_example(example, idx, "train") for idx, example in enumerate(nq_subset)]

    train_processed = hotpot_train_processed + nq_processed

    remaining_nq_size = min(test_size - len(hotpot_val_processed), len(nq_train) - len(nq_subset))
    if remaining_nq_size > 0:
        nq_test_subset = nq_train.select(range(len(nq_subset), len(nq_subset) + remaining_nq_size))
        nq_test_processed = [process_nq_example(example, idx + len(nq_subset), "test") for idx, example in enumerate(nq_test_subset)]
        test_processed = hotpot_val_processed + nq_test_processed
    else:
        test_processed = hotpot_val_processed

    print(f"Combined dataset: {len(train_processed)} train examples, {len(test_processed)} test examples")

    DatasetRegistry.register_dataset("search_r1_combined", train_processed, "train")
    test_dataset = DatasetRegistry.register_dataset("search_r1_combined", test_processed, "test")

    return test_dataset.get_data()


class TavilyMCPEvaluation:
    def __init__(self, test_size=50):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print("Error: TAVILY_API_KEY environment variable not set")
            sys.exit(1)

        self.client = MCPClient(model_base_url="http://localhost:30000/v1", model_name="Qwen/Qwen3-4B")

        self.test_size = test_size
        self.results = []

    async def process_query_with_retry(self, question: str, max_retries: int = 3) -> str:
        """Process query with exponential backoff retry logic for server errors."""
        for attempt in range(max_retries):
            try:
                response = await self.client.process_query(question, max_rounds=3)
                # If response starts with "Error:", it means there was a server error
                if not response.startswith("Error:"):
                    return response
                else:
                    print(f"Server error on attempt {attempt + 1}: {response}")
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = (2**attempt) + random.uniform(0, 1)
                        print(f"Retrying in {delay:.1f} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        return response  # Return the error response on final attempt
            except Exception as e:
                print(f"Exception on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = (2**attempt) + random.uniform(0, 1)
                    print(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    return f"Error: Failed after {max_retries} attempts - {str(e)}"

        return "Error: Failed after all retry attempts"

    async def start(self):
        try:
            print("\n📡 Connecting to Tavily MCP server...")
            await self.client.connect_to_server(server_command="npx", server_args=["-y", "tavily-mcp@0.1.3"], env={"TAVILY_API_KEY": self.api_key})

            print("\n✅ Connected! Available tools:")
            for tool in self.client.mcp_tools:
                print(f"  • {tool.name}: {tool.description}")

            await self.run_evaluation()

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback

            traceback.print_exc()
        finally:
            await self.client.cleanup()

    async def run_evaluation(self):
        print(f"\n📊 Loading evaluation datasets (test_size={self.test_size})...")

        # Load the same datasets as run_search_agent.py
        tasks = load_search_r1_data(n=1, test_size=self.test_size)

        print(f"📋 Evaluating on {len(tasks)} examples...")
        print("=" * 60)

        correct = 0
        total = 0

        for i, task in enumerate(tasks):
            try:
                question = task["task"]["question"]
                ground_truth = task["task"]["ground_truth"]
                data_source = task["task"]["data_source"]

                print(f"\n[{i + 1}/{len(tasks)}] {data_source.upper()}")
                print(f"Question: {question}")
                print(f"Ground Truth: {ground_truth}")
                print("-" * 40)

                # Add instruction for boxed answer format
                formatted_question = f"""Please answer the following question by searching for relevant information: {question}

IMPORTANT: After you find the answer, please format your final answer using \\boxed{{your_answer}} so it can be easily extracted. For example:
- For a person's name: \\boxed{{John Smith}}
- For a date: \\boxed{{March 15, 1995}}
- For a number: \\boxed{{42}}
- For a short phrase: \\boxed{{Department of Defense}}

Question: {question}"""

                response = await self.process_query_with_retry(formatted_question, max_retries=3)

                print(f"Model Response: {response}")

                if response.startswith("Error:"):
                    print("SKIPPED (Server Error)")
                    reward = 0.0
                    is_correct = False
                else:
                    try:
                        reward_output = rllm_reward_fn_search(data_source, response, ground_truth)
                        reward = reward_output.reward if hasattr(reward_output, "reward") else 0.0
                        is_correct = reward_output.is_correct if hasattr(reward_output, "is_correct") else False

                        if hasattr(reward_output, "metadata"):
                            metadata = reward_output.metadata
                            print("\n🔍 EVALUATION DEBUG:")
                            print(f"  Original Response: '{response[:100]}...'")
                            print(f"  Extracted Answer: '{metadata.get('extracted_answer', 'N/A')}'")
                            print(f"  Ground Truth: '{ground_truth[:100]}...'")
                            print(f"  Evaluation Method: {metadata.get('evaluation_method', 'N/A')}")
                            print(f"  F1 Score: {metadata.get('f1_score', 0.0):.3f}")
                            print(f"  Exact Match: {metadata.get('exact_match', False)}")
                            print(f"  Threshold: {metadata.get('f1_threshold', 0.3)}")

                    except Exception as e:
                        print(f"Evaluation error: {str(e)}")
                        reward = 0.0
                        is_correct = False

                if is_correct:
                    correct += 1
                    print("✅ CORRECT")
                elif response.startswith("Error:"):
                    print("⚠️  SKIPPED")
                else:
                    print("❌ INCORRECT")

                total += 1

                # Store results
                self.results.append({"question": question, "ground_truth": ground_truth, "response": response, "reward": reward, "correct": is_correct, "data_source": data_source, "uid": task["uid"], "skipped": response.startswith("Error:")})

                print(f"Reward: {reward:.3f}")
                print(f"Running Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
                print("-" * 40)

            except Exception as e:
                print(f"Error processing example {i + 1}: {str(e)}")
                total += 1
                # Store failed result
                self.results.append(
                    {
                        "question": task["task"]["question"] if "task" in task else "Unknown",
                        "ground_truth": task["task"]["ground_truth"] if "task" in task else "Unknown",
                        "response": "Error during processing",
                        "reward": 0.0,
                        "correct": False,
                        "data_source": task["task"]["data_source"] if "task" in task else "Unknown",
                        "uid": task.get("uid", f"error_{i}"),
                        "skipped": True,
                    }
                )

        await self.print_final_results(correct, total)

    async def print_final_results(self, correct, total):
        print("\n" + "=" * 60)
        print("🎯 FINAL EVALUATION RESULTS")
        print("=" * 60)

        # Count skipped examples
        skipped_count = sum(1 for r in self.results if r.get("skipped", False))
        valid_count = total - skipped_count

        accuracy = correct / valid_count if valid_count > 0 else 0
        print(f"Overall Accuracy: {correct}/{valid_count} ({100 * accuracy:.2f}%)")

        if skipped_count > 0:
            print(f"Skipped (Server Errors): {skipped_count}/{total} ({100 * skipped_count / total:.1f}%)")

        # Break down by dataset
        hotpot_results = [r for r in self.results if r["data_source"] == "hotpotqa" and not r.get("skipped", False)]
        nq_results = [r for r in self.results if r["data_source"] == "natural_questions" and not r.get("skipped", False)]

        if hotpot_results:
            hotpot_correct = sum(1 for r in hotpot_results if r["correct"])
            hotpot_total = len(hotpot_results)
            hotpot_acc = hotpot_correct / hotpot_total if hotpot_total > 0 else 0
            print(f"HotpotQA Accuracy: {hotpot_correct}/{hotpot_total} ({100 * hotpot_acc:.2f}%)")

        if nq_results:
            nq_correct = sum(1 for r in nq_results if r["correct"])
            nq_total = len(nq_results)
            nq_acc = nq_correct / nq_total if nq_total > 0 else 0
            print(f"Natural Questions Accuracy: {nq_correct}/{nq_total} ({100 * nq_acc:.2f}%)")

        # Average reward (excluding skipped)
        valid_results = [r for r in self.results if not r.get("skipped", False)]
        avg_reward = sum(r["reward"] for r in valid_results) / len(valid_results) if valid_results else 0
        print(f"Average Reward: {avg_reward:.3f}")

        print("=" * 60)


async def main():
    # You can adjust test_size here - start small for testing
    evaluation = TavilyMCPEvaluation(test_size=20)
    await evaluation.start()


if __name__ == "__main__":
    print("Starting Tavily MCP Evaluation...")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    asyncio.run(main())
