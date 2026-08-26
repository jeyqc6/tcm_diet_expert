# 冲突规则表 · 使用与扩充指南

**当前** 40 条 · 目标 ≥30(V1)/ ≥50(V1.5)· 最后更新 2026-08-01

这是全项目**唯一真正难被复制的资产**(`docs/PRD.md` §4.1),也是 M5 冲突调和正确率的分母。

---

## 一、当前构成

| 维度 | 分布 |
|---|---|
| **关系类型** | conflict 15 · complementary 8 · aligned_negative 5 · conditional_conflict 4 · aligned 3 · partial_conflict 2 · nutrition_internal 2 · tcm_internal 1 |
| **分线** | 补剂交互 10 · 体重管理 8 · 寒热 7 · 生熟烹饪 5 · 补血补益 5 · 痰湿乳制品 3 · 民间说法 2 |
| **来源状态** | ✅ verified 18 · ⚠️ needs_source 22 |
| **置信度** | high 22 · medium 15 · low 3 |

> ⚠️ **22 条 `needs_source` 是当前最大的欠账。** 按 `planning/worknotes.md` R1,没有外部来源的条目只算推断,不能作为 eval 的 ground truth。E2 测试子集**只能从 18 条 verified 里选**,或者先把来源补齐。

---

## 二、字段说明

| 字段 | 说明 |
|---|---|
| `rule_id` | 分线首字母 + 序号。W=体重 T=寒热 K=生熟烹饪 B=补血补益 S=补剂交互 D=痰湿乳制品 X=民间说法 |
| `tcm_position` / `tcm_source` | 中医立场与出处 |
| `nutrition_position` / `nutrition_source` | 营养学立场与出处 |
| `relation` | 见下节 |
| `resolution` | 调和后的可执行建议 |
| `resolution_rationale` | **为什么这样调和**——这一栏是调和层学习的对象,不能省 |
| `confidence` | high / medium / low |
| `evidence_level` | 双边实证 / 中医传统+营养实证 / 传统经验 / 证据不足 等 |
| `applicable_constitutions` | 适用体质,空数组表示不限 |
| `applicable_goals` | 适用目标标签,空数组表示不限 |
| `source_status` | `verified`(有一次来源)/ `needs_source`(待补) |

---

## 三、⭐ 八种关系类型 —— 不要只收 conflict

这是本表最容易做错的地方。**如果规则表里只有冲突,调和层会学会强行找分歧**,遇到两边本来就同意的情况反而制造矛盾。

| 类型 | 含义 | 例 |
|---|---|---|
| `conflict` | 两边给出方向相反的建议 | K01 熟食降生冷 vs 维C损失 |
| `partial_conflict` | 分歧在程度或形式,不在方向 | W03 高蛋白 vs 肥甘厚味 |
| `conditional_conflict` | 只在特定人群/条件下冲突 | D02 海带化痰 vs 甲状腺人群碘摄入 |
| `complementary` | 不冲突,但各自补充了对方缺失的维度 | B04 肝脏补血(中医)+ 维A上限(营养) |
| `aligned` | 两边一致支持 | W07 减少油脂 |
| `aligned_negative` | **两边一致否定** | W05 冰水提代谢 · K03 大骨汤补钙 |
| `tcm_internal` | 中医体系内部的交互 | S06 西洋参性凉叠加阳虚体质 |
| `nutrition_internal` | 营养学内部的交互 | S07 铁抑制锌吸收 |

**为什么要收后两类**:单侧内部规则不是"中西冲突",但如果不收,对应的 SubAgent 推理会漏项——中医 SubAgent 想不到补剂本身的寒热属性会叠加体质,营养 SubAgent 想不到矿物质拮抗。

---

## 四、几条值得单独说的规则

**B01 红枣补血** —— 表里最重要的一条。冲突的本质不是"谁对谁错",而是**同名概念指向不同对象**:中医的"补血"是气血津液的整体状态,不等同于西医纠正缺铁性贫血。系统必须能识别概念不对等,而不是简单站队。B03(阿胶)、B02(红糖)同构。

**S04 维生素K与华法林** —— 正确答案是"**一致性优于回避**",不是"别吃绿叶菜"。而中医"饮食有节"与营养学"摄入稳定"在此完全同构,是难得的双边互证。很多食物禁忌的正确答案都是"别忽多忽少"而非"别吃"。

**S02 vs S01 方向相反** —— 鱼油**增强**出血倾向,西洋参**减弱**华法林抗凝。由此推导出 S10:同时服用两者的用户,两种相反干扰叠加使药效更难预测。**这展示了规则条目之间可以组合产生新规则**,调和层应当能识别这类组合效应。

**S03 大蒜 vs S02 人参:证据分级** —— 同为"与抗凝药有交互",人参在四类临床决策资源中被一致识别,大蒜生姜证据强度弱得多。规则表必须体现这个差别,不能一律写成"有交互"——那会制造不必要的焦虑,也不专业。

**D01 牛奶生湿助痰** —— 唯一一条 `confidence: low` 且中医侧依据本身存疑的规则。保留它是因为它示范了一件事:**不应把所有中医表述都当作等强度的立场**。`evidence_level` 字段存在的意义就在这里。

**X02 发物** —— 最常被问到又最模糊的中医概念。处理方式是拆分:与已知过敏原重合的子集按过敏原规则严格执行,其余部分说明属于传统经验、各家说法不一。**拆分而非一刀切,是本项目调和能力的典型展示。**

---

## 五、扩充到 50 条:建议的下一批方向

| 方向 | 预计条数 | 说明 |
|---|---|---|
| **补齐 22 条 needs_source** | — | 🔴 **优先级高于新增**。规则数量不是目标,可用的规则数量才是 |
| 时令/六淫线 | 6-8 | 倒春寒、秋燥、长夏湿困 × 体质,与 Open-Meteo 数据直接挂钩 |
| 血糖/代谢线 | 5-6 | 蜂蜜润燥 vs 精制糖、果汁 vs 全果、粥养胃 vs 升糖快 |
| 咖啡因/提神线 | 3-4 | 咖啡提神 vs 阴虚心悸、浓茶 vs 铁吸收(已有 S08) |
| 特禀质/过敏线 | 4-5 | ⚠️ 高风险,每条都要 verified,不接受 needs_source |
| 老年/孕期线 | 4-5 | 人群画像 × 中医体质的交叉 |

---

## 六、写新规则时的检查清单

- [ ] `tcm_source` 和 `nutrition_source` 都是**一次来源**(教材页码 / 文献标识 / 官方文件),不是百科或社交平台
- [ ] `relation` 选对了——**不要把互补和一致误标成冲突**
- [ ] `resolution` 是**可执行的**,不是"注意平衡"这类空话
- [ ] `resolution_rationale` 说清了**为什么这样调和**,而不是重复 resolution
- [ ] `evidence_level` 诚实反映两侧证据强度,包括"中医依据存疑"这种情况
- [ ] 涉及 `te_bing`(特禀质)的规则,`source_status` 必须是 `verified`
- [ ] 不出现任何数值化的体重/热量目标(ED 防护,见 `docs/PRD.md` §10)

---

## 七、校验

```bash
python3 -c "
import json,collections
rows=[json.loads(l) for l in open('evals/conflict_rules.jsonl') if l.strip()]
print('总数',len(rows))
print(collections.Counter(r['relation'] for r in rows))
print(collections.Counter(r['source_status'] for r in rows))
keys=set().union(*[set(r) for r in rows])
print('字段缺失:',[r['rule_id'] for r in rows if set(r)!=keys] or '无')
"
```
