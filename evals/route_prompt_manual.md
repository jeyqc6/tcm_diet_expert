# 路由分类 prompt 手动测试集

用来测 `_BRANCH_GUIDE` / `_TURN_LLM_SYSTEM`（`backend/agents/routing.py`），不是测整条建议质量。
整条 agent、按功能走查见 `evals/manual_qa.md`。和冻结 eval 的区别见 `docs/EVALUATION.md` §6。

## 怎么测

前端发一条，看 API 日志里 `stage: router` 那一行：

| 字段 | 含义 |
|---|---|
| `rule_matched: true` | 正则命中，**没打到分类 prompt** |
| `reason: llm_turn:…` | 正则没命中，**这一条才在测 prompt** |
| `branch` / `branches` | 期望值见下表 |

看完一条再发下一条。不要和上一轮追问缠在一起（新 `session_id` 或等 `done` 后再发）。

---

## A. 真正打到 prompt 的（优先测）

下面每条都故意避开现有正则。期望 `rule_matched: false` 且 `reason` 以 `llm_turn:` 开头。

### other vs full_recommend（上次翻车的那一类）

| ID | 复制这句话 | 期望 branch | 不该落到 |
|---|---|---|---|
| W01 | 今天天气怎么样 | other | full_recommend |
| W02 | 外面在下雨吗 | other | full_recommend |
| W03 | 上海现在多少度 | other | full_recommend |
| W04 | 帮我查一下明天天气 | other | full_recommend |
| W05 | 下雨天吃什么 | **full_recommend** | other |
| W06 | 今天热，有什么清淡的吃 | **full_recommend** | other |
| W07 | 闷热潮湿的日子吃什么清爽一点 | **full_recommend** | other |
| W08 | 最近降温了，饮食上要注意什么 | **full_recommend** | other |
| W09 | 出差这几天三餐怎么安排 | **full_recommend** | other |
| W10 | 飞机落地肚子有点胀，接下来几天怎么吃 | **full_recommend** | other |

W01–W04 是纯问天气；W05–W10 是「天气/情境只是约束，核心在问吃什么」。

### other 其它正例

| ID | 复制这句话 | 期望 branch |
|---|---|---|
| O01 | 1+1等于几 | other |
| O02 | 比特币现在什么行情 | other |
| O03 | 红烧肉怎么做 | other |
| O04 | 番茄炒蛋要不要先炒蛋 | other |
| O05 | 刀工怎么练才不会切到手 | other |

O03–O05 是食物相关但检索覆盖不了的做法/技巧，期望 other，回复里应带「未经知识库验证」或礼貌拒答。

### 六个正式分支（措辞绕开正则）

| ID | 复制这句话 | 期望 branch | domain_hint |
|---|---|---|---|
| F01 | 山药是温性还是平性 | fact_query | tcm |
| F02 | 鸡蛋黄胆固醇高吗 | fact_query | nutrition |
| F03 | 同事点了寿司我跟不跟 | candidate_eval | — |
| F04 | 麻婆豆腐和清蒸鱼，站在我这边帮我拿个主意 | candidate_eval | — |
| F05 | 脾虚的人日常饮食方向是什么 | single_domain | tcm |
| F06 | 叶酸缺乏应该怎么补 | single_domain | nutrition |
| F07 | 从西医营养角度看控糖早餐怎么搭 | single_domain | nutrition |
| F08 | 我没什么想法，随便给点今天的饭 | full_recommend | — |
| F09 | 中午那碗面帮我存进日志 | log_write | — |
| F10 | 刚解决了一份炒河粉，入档 | log_write | — |
| F11 | 我这周末都吃了些啥来着 | log_review | — |
| F12 | 翻翻我最近几天都吃过什么 | log_review | — |

### 隐式多任务（没有「另外/顺便」）

期望 `task_count: 2`，`branches` 含下面两个（顺序以模型切分为准，两个都在即可）。M01 没有连接词（隐式多意图）；M02 有「顺便」，但两段正则都没命中，同样会打分类 LLM。

| ID | 复制这句话 | 期望 branches |
|---|---|---|
| M01 | 山药寒热属性怎么样我中午也吃了这个 | fact_query + log_write |
| M02 | 同事点了寿司我跟不跟顺便翻翻我最近几天都吃过什么 | candidate_eval + log_review |

---

## B. 正则就会拦下（用来确认没回退，不必当 prompt 测）

这些期望 `rule_matched: true`，`reason` 不是 `llm_turn:`。测过 A 再抽查即可。

| 复制这句话 | 期望 branch |
|---|---|
| 你好 | other |
| 谢谢 | other |
| 今天该吃什么 | full_recommend |
| 今天的天气适合吃什么 | full_recommend |
| 这种天气适合吃什么 | full_recommend |
| 帮我记录一下中午吃了麻婆豆腐 | log_write |
| 我昨天晚上吃了什么 | log_review |
| 红枣是什么性味 | fact_query |
| 楼下有黄焖鸡、米线，我选哪一个吃 | candidate_eval |
| 气虚质春季该吃什么 | single_domain |
| 缺铁性贫血怎么补 | single_domain |
| 红烧肉怎么做，另外今天该吃什么 | 多任务：other + full_recommend（连接词切分，不打分类 LLM） |

---

## 记结果

复制：

```
ID    got_branch         rule_matched    pass?
W01
W05
W07
O03
F03
F08
M01
```

失败时把日志里整行 `stage: router` 留下（`branch` / `reason` / `task_count` / `branches`）。
