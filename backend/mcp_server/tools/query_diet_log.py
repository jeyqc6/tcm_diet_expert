"""
按时间范围/聚合维度查询 diet_log，含相对日期解析。

设计依据：docs/ARCHITECTURE.md §2.2
注意：time_range 要接受相对表达（"昨天"/"上周"），时区基准是待决问题，见 DECISIONS.md 待决问题表
roadmap：阶段 4.2 任务 3

状态：⏳ 待实现。按 planning/roadmap.md 的阶段顺序来，不要跳过前置依赖
（见 roadmap.md 第七节第 3 条：不要"整个系统一次生成"）。
"""
