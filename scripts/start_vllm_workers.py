import time
import signal
import sys
from rllm.rollout.distributed import DistributedVLLM

def signal_handler(sig, frame):
    print("\nReceived shutdown signal. Shutting down workers...")
    engine.shutdown(persist=True)
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize DistributedVLLM
    engine = DistributedVLLM(
        num_workers=2,
        tensor_parallel_size=2,
        model="Qwen/QwQ-32B-Preview"
    )

    print("VLLM workers initialized and running...")
    print("Press Ctrl+C to shutdown workers gracefully")
    
    try:
        # Keep the script running
        while True:
            time.sleep(60)  # Sleep for 60 seconds
            print("Workers still running... (Press Ctrl+C to shutdown)")
    except Exception as e:
        print(f"Error occurred: {e}")
        engine.shutdown(persist=True)