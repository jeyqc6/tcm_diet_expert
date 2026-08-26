"""
三级查找：dish_ingredient_map → user_dish_aliases(仅已晋升) → LLM 兜底。

设计依据：docs/ARCHITECTURE.md §4.2
决策依据：docs/DECISIONS.md D27 修订一

状态：⏳ 待实现。按 planning/roadmap.md 的阶段顺序来，不要跳过前置依赖
（见 roadmap.md 第七节第 3 条：不要"整个系统一次生成"）。
"""
