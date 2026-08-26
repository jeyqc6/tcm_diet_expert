"""
模型调用统一 adapter：超时分层/重试+退避+jitter/熔断器/双档模型切换。

设计依据：docs/ARCHITECTURE.md §7
决策依据：docs/DECISIONS.md D19
roadmap：阶段 4.2 任务 1（"后补代价极大，必须第一个建"）

状态：⏳ 待实现。按 planning/roadmap.md 的阶段顺序来，不要跳过前置依赖
（见 roadmap.md 第七节第 3 条：不要"整个系统一次生成"）。
"""
