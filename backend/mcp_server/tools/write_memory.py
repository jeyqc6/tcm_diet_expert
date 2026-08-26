"""
category=critical → 写 user_profile（需人在环确认）；category=daily_log → 写 diet_log（需 idempotency_key）。仅中枢 agent 持有。

设计依据：docs/ARCHITECTURE.md §2.2/§2.3

状态：⏳ 待实现。按 planning/roadmap.md 的阶段顺序来，不要跳过前置依赖
（见 roadmap.md 第七节第 3 条：不要"整个系统一次生成"）。
"""
