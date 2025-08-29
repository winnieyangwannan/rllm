### What we currently have and what's the goal

Currently we records the strands events and make it into trajectory (so we sort of don't interceot the strands). This is implemented at @examples/strands/run_strands_workflow.py

but here comes the problem: then the sampling strategy and training strategy won't be aligned (as the sampling is directly from strands' model call while training is from rolloutengine).

So we need to use RLLModel from @examples/strands/run_strands.py to align thath.

But currently RLLModel didn't really implement the tool connections part while the @examples/strands/run_strands_workflow.py has 'closed loop' the inferencing process. so we need to implement RLLModel but we can find a lot of inspiration from run_strands_workflow.

Down the road RolloutEngine can be changed and update as it does not support toolspec that well too. Check out @tool_parser.py as it could be helpful.

--- AI EDIT BELOW ---
Context (concise)

- Problem: sampling policy (Strands calling external model API) != training policy (RolloutEngine). Off-policy risk.
- Goal: have Strands call the same policy via an adapter model, i.e., RLLMModel(rollout_engine=OpenAIEngine first). Keep event→Episode if useful; it’s optional.
- Scope-now: focus on OpenAIEngine (inference). Defer VerlEngine until later.

High-level steps (5)

1. Unify sampling/training surface: switch Strands’ model invocation to RLLMModel(rollout_engine=OpenAIEngine)
2. Tool-call bridge (minimal):
   - RolloutEngine/OpenAIEngine: accept tools/tool_choice; return tool_calls in ModelOutput
   - RLLMModel.stream: accept tool_specs; emit ToolUse events from tool_calls
3. Keep or skip StrandsWorkflow:
   - If you still want Episodes for AWE/metrics, keep current event→Episode logic
   - If not needed, you can skip; goal is policy unification, not forcing the workflow
4. Validate fast:
   - Inference smoke: one simple tool (calculator/firecrawl) end-to-end
   - (Optional) later VERL dry-run once VerlEngine parity exists
5. Docs/PRs and alignment:
   - Update example to default OpenAIEngine path; note future VERL option
   - Small PRs; keep branch rebased on upstream v0.2

Notes for implementers

- Prefer a feature flag (RLLM_STRANDS_TOOLS) to toggle tool bridge quickly
- Reuse existing tool_parser where helpful; start with minimal JSON arguments pass-through
- Do not block on true streaming; chunked text is fine for a first cut
