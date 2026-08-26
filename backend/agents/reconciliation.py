"""
调和层：独立 LLM 调用，只收两侧结论与依据，不接收原始检索内容。

设计依据：docs/ARCHITECTURE.md §5.2 步骤 5
决策依据：docs/DECISIONS.md D14
roadmap：阶段 4.2 任务 7

状态：⏳ 待实现。按 planning/roadmap.md 的阶段顺序来，不要跳过前置依赖
（见 roadmap.md 第七节第 3 条：不要"整个系统一次生成"）。
"""
