"""
SubAgent 循环状态提示：代码维护，不经过 LLM，防"状态栏投毒"。

设计依据：docs/ARCHITECTURE.md §4.5
决策依据：docs/DECISIONS.md D27
⚠️ 必须单测覆盖（确定性优先）

状态：⏳ 待实现。按 planning/roadmap.md 的阶段顺序来，不要跳过前置依赖
（见 roadmap.md 第七节第 3 条：不要"整个系统一次生成"）。
"""
