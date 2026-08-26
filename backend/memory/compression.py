"""
分层压缩：压缩优先级表 + 结构化归档摘要 + 两级触发时机（SubAgent内同步/中枢异步+同步兜底）。

设计依据：docs/ARCHITECTURE.md §4.4/§4.4.1
决策依据：docs/DECISIONS.md D8/D27
roadmap：阶段 7（本项目技术制高点）

状态：⏳ 待实现。按 planning/roadmap.md 的阶段顺序来，不要跳过前置依赖
（见 roadmap.md 第七节第 3 条：不要"整个系统一次生成"）。
"""
