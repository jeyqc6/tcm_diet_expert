"""
幂等 ingest：evals/conflict_rules.jsonl → conflict_rules 表，ON CONFLICT (rule_id) DO UPDATE。

设计依据：docs/ARCHITECTURE.md §1.2
参考：docs/ENGINEERING.md §4.1

状态：⏳ 待实现。按 planning/roadmap.md 的阶段顺序来，不要跳过前置依赖
（见 roadmap.md 第七节第 3 条：不要"整个系统一次生成"）。
"""
