import gradio as gr
import torch
from fire import Fire
import json
from typing import List, Dict, Any
import re 

def main(trajectory_file: str = "./trajectories/sample_trajectories/search_trajectories.pt", server_port: int = 9782):
    trajs_data = torch.load(trajectory_file, weights_only=False)
    all_trajs = list(filter(lambda x: hasattr(x, 'steps') and len(x.steps) > 0, trajs_data))
    
    def filter_trajectories_by_reward(filter_option: str):
        """Filter trajectories based on reward"""
        if filter_option == "All Trajectories":
            return all_trajs
        elif filter_option == "Zero Reward (Failed)":
            return [t for t in all_trajs if float(t.reward) == 0.0]
        elif filter_option == "Nonzero Reward (Partial/Full Success)":
            return [t for t in all_trajs if float(t.reward) > 0.0]
        elif filter_option == "Perfect Score (Reward = 1)":
            return [t for t in all_trajs if float(t.reward) == 1.0]
        else:
            return all_trajs

    def extract_thinking_and_response(model_response: str) -> tuple[str, str]:
        """Extract thinking and final response from model output"""
        if not model_response:
            return "", ""
        
        # Look for <think> tags
        think_match = re.search(r'<think>(.*?)</think>', model_response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # Get everything after </think>
            response = model_response.split('</think>', 1)[-1].strip()
        else:
            thinking = ""
            response = model_response.strip()
        
        return thinking, response

    def format_tool_call_detailed(tool_call: Dict) -> str:
        """Format tool calls with detailed information"""
        if isinstance(tool_call, str):
            return tool_call
        
        function = tool_call.get('function', {})
        name = function.get('name', 'unknown')
        args = function.get('arguments', {})
        tool_id = tool_call.get('id', 'no-id')
        
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except:
                args = {"raw_arguments": args}
        
        if not isinstance(args, dict):
            args = {"raw_arguments": str(args)}
        
        # Special formatting for different tool types
        if name == "local_search":
            query = args.get('query', 'No query')
            return f"🔍 **Search Query:** `{query}`\n*Tool ID: {tool_id}*"
        elif name == "finish":
            response = args.get('response', 'No response')
            return f"✅ **Finish Action:**\n```\n{response}\n```\n*Tool ID: {tool_id}*"
        else:
            return f"🛠️ **Tool:** `{name}`\n**Arguments:**\n```json\n{json.dumps(args, indent=2)}\n```\n*Tool ID: {tool_id}*"

    def format_tool_outputs(tool_outputs: Dict) -> str:
        """Format tool execution results"""
        if not tool_outputs:
            return "*No tool outputs*"
        
        formatted_outputs = []
        for tool_id, output in tool_outputs.items():
            # Truncate very long outputs
            display_output = str(output)
            if len(display_output) > 500:
                display_output = display_output[:500] + "... (truncated)"
            
            formatted_outputs.append(f"**Tool ID `{tool_id}`:**\n```\n{display_output}\n```")
        
        return "\n\n".join(formatted_outputs)

    def extract_boxed_answer(text: str) -> str:
        """Extract answer from \\boxed{} format"""
        if not text:
            return ""
        
        match = re.search(r'\\boxed\{([^}]*)\}', text)
        if match:
            return match.group(1)
        return ""

    def get_trajectory_metadata(trajectory) -> Dict:
        """Extract metadata from trajectory"""
        if not trajectory.steps:
            return {}
        
        first_obs = trajectory.steps[0].observation
        if isinstance(first_obs, dict):
            return {
                'data_source': first_obs.get('data_source', 'Unknown'),
                'question_type': first_obs.get('question_type', 'Unknown'),
                'level': first_obs.get('level', 'Unknown'),
                'uid': first_obs.get('uid', 'Unknown'),
                'split': first_obs.get('split', 'Unknown'),
                'ground_truth': first_obs.get('ground_truth', 'Unknown')
            }
        return {}

    def advance_step_or_trajectory(current_traj_idx_val, current_step_idx_val, direction, level, filtered_trajs):
        current_traj_idx_val = int(current_traj_idx_val)
        current_step_idx_val = int(current_step_idx_val)
        num_filtered_trajectories = len(filtered_trajs)

        if num_filtered_trajectories == 0:
            return 0, 0

        if level == 'step':
            num_steps_current_traj = len(filtered_trajs[current_traj_idx_val].steps)
            if direction == 'next':
                next_step_idx = current_step_idx_val + 1
                next_traj_idx = current_traj_idx_val
                if next_step_idx >= num_steps_current_traj:
                    next_step_idx = 0
            else:  # prev
                next_step_idx = current_step_idx_val - 1
                next_traj_idx = current_traj_idx_val
                if next_step_idx < 0:
                    next_step_idx = num_steps_current_traj - 1 if num_steps_current_traj > 0 else 0
        else:  # trajectory
            if direction == 'next':
                next_traj_idx = current_traj_idx_val + 1
                if next_traj_idx >= num_filtered_trajectories:
                    next_traj_idx = 0
            else:  # prev
                next_traj_idx = current_traj_idx_val - 1
                if next_traj_idx < 0:
                    next_traj_idx = num_filtered_trajectories - 1 if num_filtered_trajectories > 0 else 0
            next_step_idx = 0  # Reset to first step when changing trajectories
        
        return next_traj_idx, next_step_idx

    def update_step_view(traj_idx: int, step_idx: int, filter_option: str): 
        traj_idx = int(traj_idx)
        step_idx = int(step_idx)
        
        # Get filtered trajectories
        filtered_trajs = filter_trajectories_by_reward(filter_option)
        num_filtered_trajectories = len(filtered_trajs)

        # Default empty values
        empty_content = "*No data available*"
        
        if num_filtered_trajectories == 0:
            return (
                f"No trajectories match filter: {filter_option}", empty_content, empty_content, empty_content,
                empty_content, empty_content, empty_content, empty_content, empty_content
            )

        if not (0 <= traj_idx < num_filtered_trajectories):
            return (
                "Invalid Trajectory Index", empty_content, empty_content, empty_content,
                empty_content, empty_content, empty_content, empty_content, empty_content
            )

        trajectory = filtered_trajs[traj_idx]
        num_steps = len(trajectory.steps)
        
        step_idx = max(0, min(step_idx, num_steps - 1)) if num_steps > 0 else 0
        
        # Position and basic info
        position_text = f"Trajectory {traj_idx + 1}/{num_filtered_trajectories} | Step {step_idx + 1}/{num_steps}"
        if filter_option != "All Trajectories":
            position_text += f" | Filter: {filter_option} ({num_filtered_trajectories}/{len(all_trajs)} total)"
        
        # Trajectory metadata
        metadata = get_trajectory_metadata(trajectory)
        metadata_text = f"**Data Source:** {metadata.get('data_source', 'N/A')}\n"
        metadata_text += f"**Question Type:** {metadata.get('question_type', 'N/A')}\n"
        metadata_text += f"**Difficulty:** {metadata.get('level', 'N/A')}\n"
        metadata_text += f"**Split:** {metadata.get('split', 'N/A')}\n"
        metadata_text += f"**UID:** `{metadata.get('uid', 'N/A')}`"
        
        # Trajectory performance
        perf_text = f"**Overall Reward:** {trajectory.reward:.3f}\n"
        perf_text += f"**Total Steps:** {num_steps}\n"
        perf_text += f"**Completed:** {'✅ Yes' if (num_steps > 0 and trajectory.steps[-1].done) else '❌ No'}"
        
        # Question and ground truth
        question = trajectory.steps[0].observation.get('question', 'No question found') if num_steps > 0 else 'No question'
        ground_truth = metadata.get('ground_truth', 'Unknown')
        
        question_text = f"**Question:**\n{question}\n\n**Ground Truth Answer:** `{ground_truth}`"
        
        if num_steps == 0:
            return (
                position_text, metadata_text, perf_text, question_text,
                empty_content, empty_content, empty_content, empty_content, empty_content
            )
        
        step = trajectory.steps[step_idx]
        
        # Extract thinking and response
        thinking, response = extract_thinking_and_response(step.model_response)
        thinking_text = thinking if thinking else "*No thinking recorded*"
        response_text = response if response else "*No response recorded*"
        
        # Step performance
        step_perf_text = f"**Step Reward:** {step.reward}\n"
        step_perf_text += f"**MC Return:** {step.mc_return:.3f}\n"
        step_perf_text += f"**Done:** {'✅ Yes' if step.done else '❌ No'}\n"
        step_perf_text += f"**Step Number:** {step.step}"
        
        # Actions taken
        actions_text = empty_content
        if step.action:
            if isinstance(step.action, list):
                actions_text = "\n\n".join([format_tool_call_detailed(tc) for tc in step.action])
            else:
                actions_text = format_tool_call_detailed(step.action)
        
        # Tool outputs/results
        outputs_text = empty_content
        if step.next_observation and isinstance(step.next_observation, dict):
            tool_outputs = step.next_observation.get('tool_outputs', {})
            if tool_outputs:
                outputs_text = format_tool_outputs(tool_outputs)
        
        # Final answer analysis (for current step)
        predicted_answer = ""
        has_finish_action = False
        
        if step.action:
            actions_to_check = step.action if isinstance(step.action, list) else [step.action]
            for action in actions_to_check:
                if isinstance(action, dict) and action.get('function', {}).get('name') == 'finish':
                    has_finish_action = True
                    try:
                        args_str = action['function']['arguments']
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        finish_response = args.get('response', '')
                        predicted_answer = extract_boxed_answer(finish_response)
                        break
                    except:
                        pass
        
        if has_finish_action:
            if predicted_answer:
                is_correct = predicted_answer.lower().strip() == ground_truth.lower().strip()
                final_answer_text = f"**🎯 Final Answer Provided:**\n"
                final_answer_text += f"**Predicted:** `{predicted_answer}`\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                final_answer_text += f"**Correct:** {'✅ Yes' if is_correct else '❌ No'}"
            else:
                final_answer_text = f"**⚠️ Finish action found but no boxed answer:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`"
        else:
            # Check if this is the last step without a finish action
            if step_idx == num_steps - 1:
                final_answer_text = f"**❌ No finish action in final step:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                final_answer_text += f"**Status:** Trajectory ended without explicit final answer"
            else:
                final_answer_text = f"**⏳ No final answer yet:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                final_answer_text += f"**Status:** Step {step_idx + 1}/{num_steps} - No finish action"

        return (
            position_text, metadata_text, perf_text, question_text,
            thinking_text, response_text, step_perf_text, actions_text, outputs_text, final_answer_text
        )

    # Custom CSS for better styling
    custom_css = """
    .trajectory-container { margin-bottom: 20px !important; }
    
    /* Light mode colored boxes with proper contrast */
    .metadata-box { 
        background-color: #f8f9fa !important; 
        color: #2d3748 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #dee2e6 !important; 
    }
    .performance-box { 
        background-color: #e8f5e8 !important; 
        color: #2d5016 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #c3e6c3 !important; 
    }
    .thinking-box { 
        background-color: #fff3cd !important; 
        color: #856404 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #ffeaa7 !important; 
    }
    .actions-box { 
        background-color: #e2e3f5 !important; 
        color: #3c366b !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #b8bdf0 !important; 
    }
    .outputs-box { 
        background-color: #f0f8ff !important; 
        color: #1e3a8a !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #b3d9ff !important; 
    }
    
    /* Dark mode overrides */
    .dark .metadata-box { 
        background-color: #2d3748 !important; 
        color: #e2e8f0 !important; 
        border-color: #4a5568 !important; 
    }
    .dark .performance-box { 
        background-color: #2f4f2f !important; 
        color: #c6f6d5 !important; 
        border-color: #48bb78 !important; 
    }
    .dark .thinking-box { 
        background-color: #7c6f47 !important; 
        color: #fef5e7 !important; 
        border-color: #d69e2e !important; 
    }
    .dark .actions-box { 
        background-color: #3c366b !important; 
        color: #e9e7fd !important; 
        border-color: #805ad5 !important; 
    }
    .dark .outputs-box { 
        background-color: #1e3a8a !important; 
        color: #dbeafe !important; 
        border-color: #3b82f6 !important; 
    }
    
    /* Text inputs and general styling */
    .gr-textbox textarea { 
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important; 
        color: #2d3748 !important;
    }
    .dark .gr-textbox textarea { 
        color: #e2e8f0 !important; 
    }
    
    .step-display-box textarea { 
        text-align: center !important; 
        font-weight: bold !important; 
        font-size: 1.2em !important; 
        background-color: #ebf8ff !important; 
        color: #2b6cb0 !important; 
        border: 2px solid #bee3f8 !important; 
    }
    .dark .step-display-box textarea { 
        background-color: #1e3a8a !important; 
        color: #bfdbfe !important; 
        border-color: #3b82f6 !important; 
    }
    
    /* Navigation and buttons */
    .nav-button { min-width: 120px !important; }
    .gr-button { 
        font-size: 1.1em !important; 
        padding: 0.5em 0.8em !important; 
        border-radius: 8px !important;
        background-color: #3182ce !important;
        color: white !important;
    }
    .gr-button:hover { 
        background-color: #2c5aa0 !important; 
    }
    
    /* Markdown content in colored boxes */
    .metadata-box p, .metadata-box h1, .metadata-box h2, .metadata-box h3, .metadata-box h4,
    .performance-box p, .performance-box h1, .performance-box h2, .performance-box h3, .performance-box h4,
    .actions-box p, .actions-box h1, .actions-box h2, .actions-box h3, .actions-box h4,
    .outputs-box p, .outputs-box h1, .outputs-box h2, .outputs-box h3, .outputs-box h4 {
        color: inherit !important;
    }
    
    /* Code blocks in colored boxes */
    .metadata-box code, .performance-box code, .actions-box code, .outputs-box code {
        background-color: rgba(0,0,0,0.1) !important;
        color: inherit !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    
    .dark .metadata-box code, .dark .performance-box code, .dark .actions-box code, .dark .outputs-box code {
        background-color: rgba(255,255,255,0.1) !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Search Agent Trajectory Visualizer") as interface:
        gr.Markdown("# 🔍 Search Agent Trajectory Visualizer")
        gr.Markdown("Comprehensive visualization of agent reasoning, tool usage, and search results.")
        
        current_traj_idx_state = gr.State(0)
        current_step_idx_state = gr.State(0)

        # Filter controls
        with gr.Row():
            with gr.Column(scale=1):
                filter_dropdown = gr.Dropdown(
                    choices=[
                        "All Trajectories",
                        "Zero Reward (Failed)", 
                        "Nonzero Reward (Partial/Full Success)",
                        "Perfect Score (Reward = 1)"
                    ],
                    value="All Trajectories",
                    label="🎯 Filter by Reward",
                    interactive=True
                )
            with gr.Column(scale=1):
                # Compute filter stats for display
                zero_count = len([t for t in all_trajs if float(t.reward) == 0.0])
                nonzero_count = len([t for t in all_trajs if float(t.reward) > 0.0])
                perfect_count = len([t for t in all_trajs if float(t.reward) == 1.0])
                
                filter_stats = gr.Markdown(
                    f"**Dataset Stats:**\n"
                    f"- Total: {len(all_trajs)} trajectories\n"
                    f"- Failed (0): {zero_count}\n"
                    f"- Partial/Full Success (>0): {nonzero_count}\n"
                    f"- Perfect Score (=1): {perfect_count}"
                )

        # Navigation
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    prev_traj_button = gr.Button("⬅️ Previous Trajectory", elem_classes=["nav-button"])
                    next_traj_button = gr.Button("Next Trajectory ➡️", elem_classes=["nav-button"])
                with gr.Row():
                    prev_step_button = gr.Button("⬅️ Previous Step", elem_classes=["nav-button"])
                    next_step_button = gr.Button("Next Step ➡️", elem_classes=["nav-button"])
            
            with gr.Column(scale=2):
                current_pos_display = gr.Textbox(
                    label="Current Position", 
                    interactive=False, 
                    elem_classes=["step-display-box"]
                )

        # Main content areas
        with gr.Row():
            # Left column - Trajectory info
            with gr.Column(scale=1):
                with gr.Accordion("📊 Trajectory Metadata", open=True):
                    metadata_output = gr.Markdown(elem_classes=["metadata-box"])
                
                with gr.Accordion("🎯 Performance", open=True):
                    performance_output = gr.Markdown(elem_classes=["performance-box"])
                
                with gr.Accordion("❓ Question & Answer", open=True):
                    question_output = gr.Markdown()
                    final_answer_output = gr.Markdown()
            
            # Right column - Step details
            with gr.Column(scale=2):
                with gr.Accordion("🧠 Agent Thinking", open=True):
                    thinking_output = gr.Textbox(
                        label="Internal Reasoning",
                        lines=6,
                        interactive=False,
                        elem_classes=["thinking-box"]
                    )
                
                with gr.Accordion("💬 Agent Response", open=True):
                    response_output = gr.Textbox(
                        label="Final Response to User",
                        lines=4,
                        interactive=False
                    )
                
                with gr.Accordion("📈 Step Performance", open=True):
                    step_perf_output = gr.Markdown(elem_classes=["performance-box"])
                
                with gr.Accordion("🛠️ Actions Taken", open=True):
                    actions_output = gr.Markdown(elem_classes=["actions-box"])
                
                with gr.Accordion("📋 Tool Results", open=True):
                    outputs_output = gr.Markdown(elem_classes=["outputs-box"])

        # All outputs for update function
        all_outputs = [
            current_pos_display, metadata_output, performance_output, question_output,
            thinking_output, response_output, step_perf_output, actions_output, 
            outputs_output, final_answer_output
        ]
        
        # Helper function to reset trajectory index when filter changes
        def reset_to_first_trajectory():
            return 0, 0
        
        # Event handlers for navigation with filter support
        prev_traj_button.click(
            fn=lambda t, s, f: advance_step_or_trajectory(t, s, 'prev', 'trajectory', filter_trajectories_by_reward(f)),
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=[current_traj_idx_state, current_step_idx_state]
        )
        next_traj_button.click(
            fn=lambda t, s, f: advance_step_or_trajectory(t, s, 'next', 'trajectory', filter_trajectories_by_reward(f)),
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=[current_traj_idx_state, current_step_idx_state]
        )
        prev_step_button.click(
            fn=lambda t, s, f: advance_step_or_trajectory(t, s, 'prev', 'step', filter_trajectories_by_reward(f)),
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=[current_traj_idx_state, current_step_idx_state]
        )
        next_step_button.click(
            fn=lambda t, s, f: advance_step_or_trajectory(t, s, 'next', 'step', filter_trajectories_by_reward(f)),
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=[current_traj_idx_state, current_step_idx_state]
        )

        # Reset trajectory index when filter changes
        filter_dropdown.change(
            fn=reset_to_first_trajectory,
            outputs=[current_traj_idx_state, current_step_idx_state]
        )

        # Update view when trajectory, step, or filter changes
        current_traj_idx_state.change(
            fn=update_step_view,
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=all_outputs
        )
        current_step_idx_state.change(
            fn=update_step_view,
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=all_outputs
        )
        filter_dropdown.change(
            fn=update_step_view,
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=all_outputs
        )
        
        # Initialize view on load
        interface.load(
            fn=update_step_view, 
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], 
            outputs=all_outputs
        )

    interface.launch(share=False, server_port=server_port)

if __name__ == "__main__":
    Fire(main) 