<div align="center">

# rLLM

<div>
🚀 Reinforcement Learning for Language Agents🌟
</div>
</div>
<div>
<br>

<div align="center">

[![Notion](https://img.shields.io/badge/RLLM-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/agentica-project/rllm)
[![Website](https://img.shields.io/badge/Site-%23000000.svg?style=for-the-badge&logo=semanticweb&logoColor=white)](https://www.agentica-project.com) 
[![Twitter](https://img.shields.io/badge/Agentica-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/Agentica_)
[![Hugging Face Collection](https://img.shields.io/badge/Agentica-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/agentica-org)

</div>

</div>

rLLM is an open-source framework for post-training language agents via reinforcement learning. With rLLM, you can easily build their custom agents and environments, train them with reinforcement learning, and deploy them for real-world workloads. 



## Releases  📰

<strong>[2025/04/08]</strong> We release [`DeepSWE-Preview`](https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art[…]-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33?pvs=73
), a 32B software engineering agent trained with RL only that achieves 59% on SWEBench-Verified with test-time scaling,(42.2% Pass@1), topping the SWEBench leaderboard for open-weight models. 


<strong>[2025/04/08]</strong> We release [`DeepCoder-14B-Preview`](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51), a 14B coding model that achieves an impressive **60.6%** Pass@1 accuracy on LiveCodeBench (+8% improvement), matching the performance of `o3-mini-2025-01-031 (Low)` and `o1-2024-12-17`. 

<strong>[2025/02/10]</strong> We release [`DeepScaleR-1.5B-Preview`](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2), a 1.5B model that surpasses O1-Preview and achieves <strong>43.1% Pass@1</strong> on AIME. We achieve this by iteratively scaling Deepseek's GRPO algorithm from 8K→16K->24K context length for thinking. 


## Getting Started 🎯
### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd rllm

# Create virtual environment
uv venv --python 3.10

# Activate the virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows

# Install all dependencies
uv pip install -e ./verl
uv pip install -e .
```


## Acknowledgements

- Our training experiments are powered by our heavily modified fork of [verl](https://github.com/volcengine/verl), an open-source RLHF library.
- Our models are trained on top of [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [`DeepSeek-R1-Distill-Qwen-14B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B), and [`Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32b).
- Our work is done as part of  [Berkeley Sky Computing Lab](https://skycomputing.berkeley.edu/), [Berkeley AI Research](https://bair.berkeley.edu/), and a successful collaboration with Together AI.


## Citation
Citing rLLM:
```bibtex
@misc{rllm2025,
  title={rLLM: A Framework for Post-Training Language Agents},
  author={Sijun Tan and Michael Luo and Colin Cai and Tarun Venkat and Kyle Montgomery and Aaron Hao and Tianhao Wu and Arnav Balyan and Manan Roongta and Chenguang Wang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  year={2025},
  howpublished={\url{https://pretty-radio-b75.notion.site/rLLM-A-Framework-for-Post-Training-Language-Agents-21b81902c146819db63cd98a54ba5f31}},
  note={Notion Blog}
  year={2025}
}
```

Citing DeepSWE:
```bibtex
@misc{deepswe2025,
  title={DeepSWE: Training a State-of-the-Art Coding Agent from Scratch by Scaling RL},
  author={Michael Luo and Naman Jain and Jaskirat Singh and Sijun Tan and Ameen Patel and Qingyang Wu and Alpay Ariyak and Colin Cai and Tarun Venkat and Shang Zhu and Ben Athiwaratkun and Manan Roongta and Ce Zhang and Li Erran Li and Raluca Ada Popa and Koushik Sen and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepSWE-Training-a-Fully-Open-sourced-State-of-the-Art-Coding-Agent-by-Scaling-RL-22281902c1468193aabbe9a8c59bbe33}},
  note={Notion Blog},
  year={2025}
}
```

Citing DeepCoder and DeepScaleR:
```bibtex
@misc{deepcoder2025,
  title={DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level},
  author={Michael Luo and Sijun Tan and Roy Huang and Ameen Patel and Alpay Ariyak and Qingyang Wu and Xiaoxiang Shi and Rachel Xin and Colin Cai and Maurice Weber and Ce Zhang and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51}},
  note={Notion Blog},
  year={2025}
}
```

Citing DeepScaleR:
```bibtex
@misc{deepscaler2025,
  title={DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL},
  author={Michael Luo and Sijun Tan and Justin Wong and Xiaoxiang Shi and William Y. Tang and Manan Roongta and Colin Cai and Jeffrey Luo and Li Erran Li and Raluca Ada Popa and Ion Stoica},
  year={2025},
  howpublished={\url{https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2}},
  note={Notion Blog}
  year={2025}
}
```