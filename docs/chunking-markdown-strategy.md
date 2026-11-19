# Chunking 策略对比：结构感知 Markdown（初版）

> run: `md-strategy-20261119`；同一 Markdown 语料、同一 embedding/rerank 模型，只改 chunk 配置。
> 本报告是**按标题硬切的初版**：短章节会产生大量小 chunk。加入短章节合并后的当前实现见 [`chunking-markdown-strategy-v2.md`](chunking-markdown-strategy-v2.md)。

## 实验配置

| 策略 | strategy | 父块 | 子块 | 粒度 |
|---|---|---:|---:|---|
| `recursive-overlap` | `recursive` | 1000/200 | 400/0 | child |
| `recursive-no-parent-overlap` | `recursive` | 1000/0 | 400/0 | child |
| `markdown-heading` | `markdown-heading` | 1000/0 | 400/0 | child |

`recursive-no-parent-overlap` 与 `markdown-heading` 使用完全相同的 1000/0 + 400/0，唯一变量是分块边界策略；`recursive-overlap` 保留当前 `coarse-no-overlap` profile 的父块 overlap，用于回答“结构感知是否能超过用户今天能选的策略”。

`markdown-heading` 的每个 chunk 会前置自己的 heading path（例如 `# 作品概况 > 基本信息`），用于保留章节上下文；这不调用额外模型，属于策略本身的一部分。

## 总体指标

| 策略 | n | Recall@k | MRR | Precision@k | nDCG@k | P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| `recursive-overlap` | 50 | 0.935 | 0.804 | 0.357 | 0.831 | 1131 / 1476 ms |
| `recursive-no-parent-overlap` | 50 | 0.935 | 0.759 | 0.326 | 0.792 | 1118 / 1727 ms |
| `markdown-heading` | 50 | 0.935 | 0.828 | 0.317 | 0.840 | 1213 / 2026 ms |

本次索引规模（PostgreSQL 统计）：

| 策略 | 父块数 | 子块数 |
|---|---:|---:|
| `recursive-overlap` | 62 | 190 |
| `recursive-no-parent-overlap` | 58 | 168 |
| `markdown-heading` | 196 | 243 |

## 结构完整性（直接扫描 Milvus 子块）

| 策略 | 子块数 | 中位长度 | P90 长度 | 含表格子块 | 表头保留率 | 列数一致率 |
|---|---:|---:|---:|---:|---:|---:|
| `recursive-overlap` | 191 | 312 | 389 | 68 | 0.794 | 0.794 |
| `recursive-no-parent-overlap` | 168 | 307 | 388 | 66 | 0.788 | 0.788 |
| `markdown-heading` | 243 | 192 | 376 | 69 | 1.000 | 1.000 |

表头保留率/列数一致率只统计含 `|` 行的子块；结构感知策略应接近 1.0，递归切分把表格从中间切断时会明显下降。

> Milvus 实际子块数可能比 PostgreSQL `chunk_hashes` 多 1–2 条：相同文本子块在 PG 去重、在 Milvus 各存一条向量；以实际扫描为准。

## 按题型 Recall@k

| 题型 | n | `recursive-overlap` | `recursive-no-parent-overlap` | `markdown-heading` |
|---|---:|---:|---:|---:|
| `alias` | 2 | 1.000 | 1.000 | 1.000 |
| `multi_hop` | 7 | 0.857 | 0.857 | 0.857 |
| `numeric` | 4 | 1.000 | 1.000 | 1.000 |
| `single_hop` | 31 | 0.935 | 0.935 | 0.935 |
| `unanswerable` | 4 | - | - | - |
| `version_conflict` | 2 | 1.000 | 1.000 | 1.000 |

## 相对基线（配对 bootstrap 95% CI）

| 策略 | ΔRecall | ΔMRR | ΔPrecision | ΔnDCG |
|---|---:|---:|---:|---:|
| `recursive-no-parent-overlap` | +0.000 [-0.065, +0.065] | -0.045 [-0.121, +0.033] | -0.030 [-0.057, -0.004] | -0.039 [-0.096, +0.016] |
| `markdown-heading` | +0.000 [-0.065, +0.065] | +0.024 [-0.033, +0.080] | -0.039 [-0.074, -0.009] | +0.009 [-0.038, +0.056] |

## 失败/翻转样本

只列出相对基线发生 hit/miss 翻转的问题；`best_overlap` 越低，说明检索器越没有覆盖 gold evidence。

### `recursive-no-parent-overlap`（2 条）

- `v-q031` 基线命中 → 本策略未命中：第 3 话和第 4 话分别主要围绕哪位女性配角的家庭/故乡展开？
  - 基线 top files：05-tv-episodes.md、03-characters-main.md、01-overview.md
  - 本策略 top files：05-tv-episodes.md、03-characters-main.md、01-overview.md
  - best_overlap：0.186
  - gold evidence：露库莉雅带薇尔莉特去钟楼，讲了自己的家庭：父母在西部战线被突破时死去，哥哥斯宾塞应征入伍，虽然活着回来，却因幸存者内疚而酗酒、消沉。她一边照顾哥哥，一边想告诉他“你回来就好”，却说不出口。
- `v-q032` 基线未命中 → 本策略命中：TV 第 8-9 话主要补充了哪些背景？
  - 基线 top files：01-overview.md、09-timeline.md、05-tv-episodes.md
  - 本策略 top files：09-timeline.md、05-tv-episodes.md、02-worldbuilding.md
  - best_overlap：0.618
  - gold evidence：名、教她读写、送她胸针。  最后，基尔伯特的部队进攻敌方总部，付出惨重代价后攻下。就在基尔伯特发出信号时，敌军开枪击中了他。  作用：集中补充薇尔莉特的身世、名字与胸针的由来；把“基尔伯特之死”从传闻推到眼前。    第 9 话：薇尔莉特·伊芙加登  回忆继续：薇尔莉特试图救重伤…

### `markdown-heading`（2 条）

- `v-q031` 基线命中 → 本策略未命中：第 3 话和第 4 话分别主要围绕哪位女性配角的家庭/故乡展开？
  - 基线 top files：05-tv-episodes.md、03-characters-main.md、01-overview.md
  - 本策略 top files：05-tv-episodes.md、03-characters-main.md、01-overview.md
  - best_overlap：0.174
  - gold evidence：露库莉雅带薇尔莉特去钟楼，讲了自己的家庭：父母在西部战线被突破时死去，哥哥斯宾塞应征入伍，虽然活着回来，却因幸存者内疚而酗酒、消沉。她一边照顾哥哥，一边想告诉他“你回来就好”，却说不出口。
- `v-q032` 基线未命中 → 本策略命中：TV 第 8-9 话主要补充了哪些背景？
  - 基线 top files：01-overview.md、09-timeline.md、05-tv-episodes.md
  - 本策略 top files：09-timeline.md、05-tv-episodes.md、00-version-notes.md
  - best_overlap：0.612
  - gold evidence：名、教她读写、送她胸针。  最后，基尔伯特的部队进攻敌方总部，付出惨重代价后攻下。就在基尔伯特发出信号时，敌军开枪击中了他。  作用：集中补充薇尔莉特的身世、名字与胸针的由来；把“基尔伯特之死”从传闻推到眼前。    第 9 话：薇尔莉特·伊芙加登  回忆继续：薇尔莉特试图救重伤…

### 基线未命中（3 条）

- `v-q011` 克劳迪亚·霍金斯经营的公司叫什么？；best_overlap=0.102；top files：03-characters-main.md、02-worldbuilding.md、01-overview.md；gold evidence：中奥斯卡剧本里的“魔法少女与精灵”是戏中戏，不是现实设定。  地理与政治  故事的核心区域是大陆南部的莱顿沙夫特里希（Leidenschaftlich），首都莱顿（Leiden）。C·H 邮政公司就位于莱顿。这个国家在故事开始前刚结束一场长达四年的大陆战争。  国家/地区  说明…
- `v-q026` 2020 年剧场版中，黛西·马格诺利亚是谁的孙女？；best_overlap=0.303；top files：04-characters-supporting.md、01-overview.md、11-faq.md；gold evidence：。  •  性格：温柔、克制、坚强，面对死亡有恐惧也有不舍。  •  关系：安（女儿）、薇尔莉特（代笔人）。  •  关键词：50 年、生日信、母爱、遗书。    安·马格诺利亚（Anne Magnolia）  项目  内容  中文名  安·马格诺利亚  其他译名  安妮·玛格诺…
- `v-q032` TV 第 8-9 话主要补充了哪些背景？；best_overlap=0.125；top files：01-overview.md、09-timeline.md、05-tv-episodes.md；gold evidence：名、教她读写、送她胸针。  最后，基尔伯特的部队进攻敌方总部，付出惨重代价后攻下。就在基尔伯特发出信号时，敌军开枪击中了他。  作用：集中补充薇尔莉特的身世、名字与胸针的由来；把“基尔伯特之死”从传闻推到眼前。    第 9 话：薇尔莉特·伊芙加登  回忆继续：薇尔莉特试图救重伤…

## 解读

- **Recall** 衡量“至少命中一条 gold evidence”，对 chunk 边界最敏感。
- **MRR / nDCG** 衡量相关证据是否排在前面；相同 Recall 下，它们能区分排序质量。
- **Precision** 衡量前 k 条里有多少是证据相关，能暴露“大 chunk 把噪声一起带进来”。
- 如果 overlap 策略提升 Recall 但提高索引大小/延迟，应报告这个 trade-off，而不是只报最高分。

## 结论

- 结构切分确实生效：表格子块的表头保留率和列数一致率明显高于递归策略（见上表）。
- 但 Recall@5 没有提升；MRR/nDCG 的正向变化 95% CI 跨 0，Precision 的下降 CI 不跨 0；主要代价是子块更小、更多，P95 延迟更高。
- 因此**不替换默认 `fine-overlap`**。`markdown-heading` 适合章节/表格结构本身就是检索信号的 Markdown 知识库；下一步应增加“短章节合并”或按文档调 child size，再验证 Precision 和延迟。

> 单次 50 题评测仍有抽样波动；本报告的 95% CI 是配对 bootstrap，解释时应同时看区间和效应量。
