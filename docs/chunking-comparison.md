# Chunking 策略对比

> run: `real-20261116`；同一语料、同一 embedding/rerank 模型，只改 chunk 配置。
> `gold_evidence` 为自动生成标签，本次已修正别名题 `v-q047` 的误匹配；正式提交前仍建议人工抽检 10–15 条。
> 本报告是 **DOCX 尺寸对比**；结构感知 Markdown 策略对比见 [`chunking-markdown-strategy.md`](chunking-markdown-strategy.md)。

## 实验配置

| 策略 | 父块 | 子块 | 粒度 |
|---|---:|---:|---|
| `child-200-o0` | 1000/200 | 200/0 | child |
| `child-200-o50` | 1000/200 | 200/50 | child |
| `child-400-o0` | 1000/200 | 400/0 | child |

## 总体指标

| 策略 | n | Recall@k | MRR | Precision@k | nDCG@k | P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| `child-200-o0` | 50 | 0.848 | 0.670 | 0.300 | 0.700 | 2870 / 5777 ms |
| `child-200-o50` | 50 | 0.913 | 0.715 | 0.330 | 0.749 | 3146 / 5577 ms |
| `child-400-o0` | 50 | 0.957 | 0.765 | 0.357 | 0.811 | 2683 / 3625 ms |

本次索引规模（PostgreSQL 统计）：

| 策略 | 父块数 | 子块数 |
|---|---:|---:|
| `child-200-o0` | 58 | 321 |
| `child-200-o50` | 58 | 352 |
| `child-400-o0` | 58 | 164 |

## 相对基线（配对 bootstrap 95% CI）

| 策略 | ΔRecall | ΔMRR | ΔPrecision | ΔnDCG |
|---|---:|---:|---:|---:|
| `child-200-o50` | +0.065 [-0.022, +0.153] | +0.045 [-0.052, +0.142] | +0.030 [-0.013, +0.070] | +0.049 [-0.027, +0.133] |
| `child-400-o0` | +0.109 [+0.022, +0.217] | +0.095 [-0.009, +0.203] | +0.057 [+0.017, +0.100] | +0.111 [+0.028, +0.196] |

## 失败/翻转样本

只列出相对基线发生 hit/miss 翻转的问题；`best_overlap` 越低，说明检索器越没有覆盖 gold evidence。

### `child-200-o50`（5 条）

- `v-q002` 基线未命中 → 本策略命中：TV 动画《紫罗兰永恒花园》的导演是谁？
  - 基线 top files：07-production.docx、00-version-notes.docx
  - 本策略 top files：07-production.docx、00-version-notes.docx、06-films.docx
  - best_overlap：0.627
  - gold evidence：let Evergarden  原作  晓佳奈（作者）、高濑亚贵子（插画）  原作出版社  京都动画 / KA Esuma 文库  TV 动画制作  京都动画  TV 动画导演  石立太一  系列构成  吉田玲子  角色设计  高濑亚贵子  音乐  Evan Call  TV 播…
- `v-q015` 基线未命中 → 本策略命中：2020 年剧场版《紫罗兰永恒花园 剧场版》的导演是谁？
  - 基线 top files：00-version-notes.docx、07-production.docx、06-films.docx
  - 本策略 top files：06-films.docx、00-version-notes.docx、07-production.docx
  - best_overlap：0.758
  - gold evidence：场版  日文名  劇場版 ヴァイオレット・エヴァーガーデン  英文名  Violet Evergarden: The Movie  上映  2020-09-18（日本）  导演  石立太一  剧本  吉田玲子  角色设计  高濑亚贵子  动画制作  京都动画  发行  松竹  音…
- `v-q032` 基线命中 → 本策略未命中：TV 第 8-9 话主要补充了哪些背景？
  - 基线 top files：09-timeline.docx、05-tv-episodes.docx、11-faq.docx
  - 本策略 top files：01-overview.docx、09-timeline.docx、05-tv-episodes.docx
  - best_overlap：0.145
  - gold evidence：名、教她读写、送她胸针。  最后，基尔伯特的部队进攻敌方总部，付出惨重代价后攻下。就在基尔伯特发出信号时，敌军开枪击中了他。  作用：集中补充薇尔莉特的身世、名字与胸针的由来；把“基尔伯特之死”从传闻推到眼前。    第 9 话：薇尔莉特·伊芙加登  回忆继续：薇尔莉特试图救重伤…
- `v-q046` 基线未命中 → 本策略命中：第 1 话中，薇尔莉特为什么想加入 C·H 邮政的代笔部门？
  - 基线 top files：09-timeline.docx、05-tv-episodes.docx、03-characters-main.docx
  - 本策略 top files：09-timeline.docx、03-characters-main.docx、05-tv-episodes.docx
  - best_overlap：0.742
  - gold evidence：薇尔莉特在邮局看到人偶替一位不识字的男人写信，男人终于把难以启齿的爱意表达出来。这一幕让她想起基尔伯特最后的“我爱你”。她向霍金斯提出：我想做自动手记人偶，想弄明白那句话是什么意思。霍金斯被她的“自主选择”打动，接受了这个没有受过正规训练的少女。
- `v-q047` 基线未命中 → 本策略命中：“维尔莉特·埃弗加登”指的是哪个角色？
  - 基线 top files：01-overview.docx、03-characters-main.docx
  - 本策略 top files：03-characters-main.docx、01-overview.docx、00-version-notes.docx
  - best_overlap：0.962
  - gold evidence：爱你”》。  中文译名对照  同一角色在不同字幕、小说译本和粉丝讨论中会有不同写法。检索时建议做别名扩展。  常见主译名  其他写法/别名  日文  英文  薇尔莉特·伊芙加登  维尔莉特·埃弗加登、薇尔莉特·艾佛加登  ヴァイオレット・エヴァーガーデン  Violet Ever…

### `child-400-o0`（5 条）

- `v-q002` 基线未命中 → 本策略命中：TV 动画《紫罗兰永恒花园》的导演是谁？
  - 基线 top files：07-production.docx、00-version-notes.docx
  - 本策略 top files：07-production.docx、00-version-notes.docx
  - best_overlap：1.000
  - gold evidence：let Evergarden  原作  晓佳奈（作者）、高濑亚贵子（插画）  原作出版社  京都动画 / KA Esuma 文库  TV 动画制作  京都动画  TV 动画导演  石立太一  系列构成  吉田玲子  角色设计  高濑亚贵子  音乐  Evan Call  TV 播…
- `v-q004` 基线未命中 → 本策略命中：TV 动画《紫罗兰永恒花园》的首播和最终话播出日期分别是？
  - 基线 top files：07-production.docx、00-version-notes.docx、09-timeline.docx
  - 本策略 top files：00-version-notes.docx、09-timeline.docx、07-production.docx
  - best_overlap：0.974
  - gold evidence：KA Esuma 文库  TV 动画制作  京都动画  TV 动画导演  石立太一  系列构成  吉田玲子  角色设计  高濑亚贵子  音乐  Evan Call  TV 播出  2018-01-11 至 2018-04-05，全 13 话  OVA  第 14 话，2018-0…
- `v-q015` 基线未命中 → 本策略命中：2020 年剧场版《紫罗兰永恒花园 剧场版》的导演是谁？
  - 基线 top files：00-version-notes.docx、07-production.docx、06-films.docx
  - 本策略 top files：00-version-notes.docx、07-production.docx、06-films.docx
  - best_overlap：1.000
  - gold evidence：场版  日文名  劇場版 ヴァイオレット・エヴァーガーデン  英文名  Violet Evergarden: The Movie  上映  2020-09-18（日本）  导演  石立太一  剧本  吉田玲子  角色设计  高濑亚贵子  动画制作  京都动画  发行  松竹  音…
- `v-q046` 基线未命中 → 本策略命中：第 1 话中，薇尔莉特为什么想加入 C·H 邮政的代笔部门？
  - 基线 top files：09-timeline.docx、05-tv-episodes.docx、03-characters-main.docx
  - 本策略 top files：05-tv-episodes.docx、01-overview.docx、09-timeline.docx
  - best_overlap：1.000
  - gold evidence：薇尔莉特在邮局看到人偶替一位不识字的男人写信，男人终于把难以启齿的爱意表达出来。这一幕让她想起基尔伯特最后的“我爱你”。她向霍金斯提出：我想做自动手记人偶，想弄明白那句话是什么意思。霍金斯被她的“自主选择”打动，接受了这个没有受过正规训练的少女。
- `v-q047` 基线未命中 → 本策略命中：“维尔莉特·埃弗加登”指的是哪个角色？
  - 基线 top files：01-overview.docx、03-characters-main.docx
  - 本策略 top files：03-characters-main.docx、01-overview.docx、00-version-notes.docx
  - best_overlap：0.962
  - gold evidence：爱你”》。  中文译名对照  同一角色在不同字幕、小说译本和粉丝讨论中会有不同写法。检索时建议做别名扩展。  常见主译名  其他写法/别名  日文  英文  薇尔莉特·伊芙加登  维尔莉特·埃弗加登、薇尔莉特·艾佛加登  ヴァイオレット・エヴァーガーデン  Violet Ever…

### 基线未命中（7 条）

- `v-q002` TV 动画《紫罗兰永恒花园》的导演是谁？；best_overlap=0.405；top files：07-production.docx、00-version-notes.docx；gold evidence：let Evergarden  原作  晓佳奈（作者）、高濑亚贵子（插画）  原作出版社  京都动画 / KA Esuma 文库  TV 动画制作  京都动画  TV 动画导演  石立太一  系列构成  吉田玲子  角色设计  高濑亚贵子  音乐  Evan Call  TV 播…
- `v-q004` TV 动画《紫罗兰永恒花园》的首播和最终话播出日期分别是？；best_overlap=0.000；top files：07-production.docx、00-version-notes.docx、09-timeline.docx；gold evidence：KA Esuma 文库  TV 动画制作  京都动画  TV 动画导演  石立太一  系列构成  吉田玲子  角色设计  高濑亚贵子  音乐  Evan Call  TV 播出  2018-01-11 至 2018-04-05，全 13 话  OVA  第 14 话，2018-0…
- `v-q011` 克劳迪亚·霍金斯经营的公司叫什么？；best_overlap=0.079；top files：01-overview.docx、02-worldbuilding.docx、11-faq.docx；gold evidence：中奥斯卡剧本里的“魔法少女与精灵”是戏中戏，不是现实设定。  地理与政治  故事的核心区域是大陆南部的莱顿沙夫特里希（Leidenschaftlich），首都莱顿（Leiden）。C·H 邮政公司就位于莱顿。这个国家在故事开始前刚结束一场长达四年的大陆战争。  国家/地区  说明…
- `v-q015` 2020 年剧场版《紫罗兰永恒花园 剧场版》的导演是谁？；best_overlap=0.405；top files：00-version-notes.docx、07-production.docx、06-films.docx；gold evidence：场版  日文名  劇場版 ヴァイオレット・エヴァーガーデン  英文名  Violet Evergarden: The Movie  上映  2020-09-18（日本）  导演  石立太一  剧本  吉田玲子  角色设计  高濑亚贵子  动画制作  京都动画  发行  松竹  音…
- `v-q026` 2020 年剧场版中，黛西·马格诺利亚是谁的孙女？；best_overlap=0.139；top files：04-characters-supporting.docx、06-films.docx、11-faq.docx；gold evidence：。  •  性格：温柔、克制、坚强，面对死亡有恐惧也有不舍。  •  关系：安（女儿）、薇尔莉特（代笔人）。  •  关键词：50 年、生日信、母爱、遗书。    安·马格诺利亚（Anne Magnolia）  项目  内容  中文名  安·马格诺利亚  其他译名  安妮·玛格诺…
- `v-q046` 第 1 话中，薇尔莉特为什么想加入 C·H 邮政的代笔部门？；best_overlap=0.434；top files：09-timeline.docx、05-tv-episodes.docx、03-characters-main.docx；gold evidence：薇尔莉特在邮局看到人偶替一位不识字的男人写信，男人终于把难以启齿的爱意表达出来。这一幕让她想起基尔伯特最后的“我爱你”。她向霍金斯提出：我想做自动手记人偶，想弄明白那句话是什么意思。霍金斯被她的“自主选择”打动，接受了这个没有受过正规训练的少女。
- `v-q047` “维尔莉特·埃弗加登”指的是哪个角色？；best_overlap=0.000；top files：01-overview.docx、03-characters-main.docx；gold evidence：爱你”》。  中文译名对照  同一角色在不同字幕、小说译本和粉丝讨论中会有不同写法。检索时建议做别名扩展。  常见主译名  其他写法/别名  日文  英文  薇尔莉特·伊芙加登  维尔莉特·埃弗加登、薇尔莉特·艾佛加登  ヴァイオレット・エヴァーガーデン  Violet Ever…

## 本次结论

- **child 400/0 是本次最佳配置**：Recall@5 从 0.848 提升到 0.957（+0.109，95% CI +0.022–+0.217），Precision@5 和 nDCG@5 也显著提升；MRR 的区间跨 0，只能说明方向为正，不能断言稳定提升。
- **child 200/50 的绝对指标更高，但统计上不显著**：Recall +0.065、nDCG +0.049 的 95% CI 都包含 0；overlap 在这个小评测集上不足以证明稳定收益。
- **更大 chunk 同时更快**：400 字子块只有 164 个，少于 200/0 的 321 个和 200/50 的 352 个，P50/P95 也最低；代价是单块更可能混入无关上下文。
- **别名题是明显翻转点**：`v-q047` 在 200/0 下未命中，在 200/50 和 400/0 下命中，说明 chunk 边界会直接影响“译名对照表”这类表格证据。
- **不可回答样本的 clean rate 为 0%**：4 条不可回答样本在 child 检索下都至少返回了一个 chunk。当前 child 检索没有拒答阈值，这项不能用来评价 chunking，应结合答案评测的拒答指标看。

## 解读

- **Recall** 衡量“至少命中一条 gold evidence”，对 chunk 边界最敏感。
- **MRR / nDCG** 衡量相关证据是否排在前面；相同 Recall 下，它们能区分排序质量。
- **Precision** 衡量前 k 条里有多少是证据相关，能暴露“大 chunk 把噪声一起带进来”。
- 如果 overlap 策略提升 Recall 但提高索引大小/延迟，应报告这个 trade-off，而不是只报最高分。

> 单次 50 题评测仍有抽样波动；本报告的 95% CI 是配对 bootstrap，解释时应同时看区间和效应量。
