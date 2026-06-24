# DiSan 宣传稿件（Press Kit）

> 本文档为 DiSan（Disentangled Sanitization）项目的中文宣传素材包，包含：
> 1. 一句话卖点 / 电梯演讲
> 2. 技术博文版（长文）
> 3. 社交媒体短帖（Twitter/X、知乎、即刻、朋友圈）
> 4. 技术社区发帖模板（GitHub Discussions、Reddit、Hacker News）
> 5. 新闻通稿（媒体投稿版）

---

## 一、核心信息卡（Key Messaging）

| 维度 | 内容 |
|------|------|
| **项目名称** | DiSan（Disentangled Sanitization）|
| **一句话** | 让跨组织 Agent 共享文本时，既能保留业务语义，又不泄露机构身份。 |
| **核心创新** | 首个将「解耦表示学习」用于跨 Agent 文本脱敏的联邦框架，分离「任务语义」与「源标识风格」。 |
| **关键数字** | PII 暴露 ↓20×、TF-IDF 归因 ↓73.2%、BERT 探测 ↓70.6%、RAG 忠实度保持 83% |
| **所属机构** | 上海人工智能实验室（Shanghai AI Lab）、上海交通大学 |
| **集成产品** | Intern-Shannon 多智能体协作框架内置隐私模块 |
| **开源地址** | `https://github.com/RezinChow/DiSan` |
| **论文地址** | `https://arxiv.org/abs/2606.15335` |
| **项目页面** | `https://rezinchow.github.io/DiSan/` |

---

## 二、电梯演讲（30 秒版）

> 跨组织的 AI Agent 在协作时，经常需要互相交换文档片段来回答问题。但直接传原文会泄露隐私——不只是姓名、账号这些显式信息，更重要的是每家机构的**文风指纹**：特有的格式惯例、词汇偏好、句法模式。
>
> 现有的标识符掩码（把名字替换成 [ORG]）远远不够，我们的实验显示，即使掩掉 19.2% 的 token，攻击者依然能用 TF-IDF 以 67% 的准确率识别来源。
>
> DiSan 的做法是：**在表示层面就把「任务内容」和「机构风格」彻底分开**。我们用双路编码器把文本解耦成两个子空间，只把任务内容传给对方，风格信息留在本地。这样既能保证下游任务拿到有用的信息，又让攻击者无从下手。
>
> 在真实 Enron 邮件和合成金融 RAG 基准上，DiSan 把作者归因成功率从 82.5% 降到了 22.1%，同时保持了 83% 的答案质量。

---

## 三、技术深度博文（长文版，适合知乎/公众号/技术博客）

### 标题建议
- 《DiSan：当 Agent 开始跨公司协作，如何防止它"出卖"东家？》
- 《别再把名字改成 [ORG] 就以为脱敏了——分布式 Agent 隐私的真正解法》
- 《从 82.5% 到 22.1%：我们如何把文本作者归因攻击打趴下》

### 正文框架

#### 引子：一个反直觉的发现
（用 Table 1 的例子开场）

假设你是风控部门分析师，需要向数据管理 Agent 查询 ApexChip Q3 的财务风险敞口。对方返回了一段脱敏后的文本：

> "Per [ORG]'s Counterparty Risk Bulletin (Ref: [ID]), [ORG] carried $6.1M exposure as of Q3 close, was downgraded to BB+, and was flagged for portfolio review."

名字和编号都被打码了，看起来安全？

**错。** 攻击者仍能从这句话里读出太多东西：
- "Counterparty Risk Bulletin" 是某家银行的**专有报告模板**
- "Ref: [ID]" 保留了**内部引用体系**
- "downgraded to BB+" 的**措辞习惯**暴露了来源

我们的实验证明，即使使用业界最强的 GLiNER 检测器，掩码掉 19.2% 的 token，攻击者依然能用 TF-IDF 达到 **67.2%** 的作者归因准确率。隐式分布签名（distributional signatures）才是跨机构文本共享中真正的大象。

#### 核心思路：Role-Style 解耦

DiSan 的核心洞察是：**文本里同时包含「说了什么」（Role）和「谁说的」（Style）**。只要我们在表示层面把这两者彻底分离，就能只传内容、不传身份。

具体做法：
1. **双流编码器**：基于 LongT5 构建两个独立子网络，分别输出 Role 嵌入和 Style 嵌入
2. **正交约束**：通过正交性损失（orthogonality loss）强制两个子空间互不泄露
3. **联邦原型对齐**：各 Agent 本地计算 Role 原型，只上传聚合向量，无需集中原始文本
4. **对抗训练**：用梯度反转层（GRL）训练一个"客户端判别器"，迫使 Role 嵌入无法被归因

推理时，**只有 Role 流参与解码**，Style 信息被彻底丢弃。

#### 实验结果：不只是数字

**Enron 真实邮件（7 位作者，各 500 封）：**

| 方法 | TF-IDF 归因 F1 | 降幅 |
|------|----------------|------|
| 原始文本 | 82.5% | — |
| GLiNER 掩码（19.2% token） | 67.2% | 18.6% |
| JAMDEC（NeurIPS 2024） | 81.6% | 1.2% |
| StyleRemix Fixed | 73.7% | 11.5% |
| **DiSan** | **22.1%** | **73.2%** |

从 82.5% 到 22.1%，接近随机基线（14.3%）。这意味着，**经过 DiSan 处理后，攻击者几乎无法从文本中分辨作者。**

**RAG 效用（合成金融 7-Agent 分布式基准）：**

DiSan 将答案级 PII 暴露从 40% 压到 2%，同时保持了 83% 的答案忠实度，语义相似度 0.82，与原始共享的差距仅 0.03。

#### 已落地：Intern-Shannon 的内置组件

DiSan 不是停留在论文里的概念。它已被集成进上海 AI Lab 的 **Intern-Shannon** 多智能体协作框架，作为内置的文本脱敏模块。在内部协作、跨部门对齐、跨机构联盟三种场景中，Agent 可在返回证据前自动调用 DiSan，实现"数据不出域，知识可共享"。

#### 开源与复现

代码、训练脚本、RAG 管道已全部开源。单条命令即可启动联邦训练：

```bash
bash fed_lightweight/train_lora_v2.sh
```

---

## 四、社交媒体短帖

### Twitter / X（英文）

```
Privacy in multi-agent collaboration isn't just about masking names.

DiSan learns to *disentangle* what is said (role) from who said it (style).
Only the role stream is shared. The style stays local.

Results on Enron: 73.2% reduction in stylometric attribution,
while keeping 83% answer faithfulness in distributed RAG.

Project page: https://rezinchow.github.io/DiSan/
Code: https://github.com/RezinChow/DiSan

Built into Intern-Shannon by Shanghai AI Lab.
```

### 知乎 / 即刻 / 朋友圈（中文）

**版本 A（硬核向）：**

> 做了个叫 DiSan 的工作，解决分布式 Agent 协作时的文本隐私问题。
>
> 核心发现：把名字改成 [ORG] 根本不够。攻击者通过文风、格式、句法习惯，依然能以 67% 准确率识别来源。
>
> 我们的解法是在表示层把「任务语义」和「机构风格」彻底解耦，只传内容不传身份。Enron 真实邮件上把作者归因从 82.5% 打到 22.1%，同时保持 83% 的下游任务效果。
>
> 已集成进 Intern-Shannon，代码开源。

**版本 B（故事向）：**

> 想象你的 AI Agent 正在帮另一家公司查数据。
>
> 你小心翼翼地隐去了所有姓名和账号，以为这就安全了。
>
> 但对方从你返回文本的「格式模板」「引用体系」「措辞习惯」里，轻而易举地推断出了你的公司身份。
>
> 这就是 DiSan 要解决的问题。我们提出的不是更好的"打码"，而是让 AI 从根本上学会：什么是可以共享的知识，什么是必须保留的隐私。

---

## 五、技术社区发帖模板

### GitHub Discussions / Reddit r/MachineLearning

**Title:** [R] DiSan: Disentangled Text Sanitization for Privacy-Preserving Multi-Agent Collaboration

**Body:**

```
Hi folks, we’d like to share DiSan, a framework for privacy-preserving text sanitization in distributed agent systems.

**Problem:** When agents exchange text across organizational boundaries, privacy leaks through not just explicit identifiers (PII) but also distributional signatures—formatting conventions, vocabulary choices, and syntactic patterns that encode the source’s identity. We show that even aggressive placeholder masking (19.2% tokens) only reduces stylometric attribution by 18.6%.

**Solution:** DiSan factorizes text into a source-invariant *role* subspace (task semantics) and a source-identifying *style* subspace. Only the role stream is shared; style remains local. Training uses federated prototype alignment + adversarial regularization, so raw text never leaves the agent.

**Key results:**
- Enron 7-way authorship: TF-IDF F1 drops from 0.825 → 0.221 (73.2% reduction)
- BERT probe: 0.691 → 0.203 (70.6% reduction)
- Distributed RAG: 20× reduction in answer-level PII exposure, 83% answer faithfulness retained

**Code:** https://github.com/RezinChow/DiSan
**Project page:** https://rezinchow.github.io/DiSan/

Happy to answer questions!
```

### Hacker News

**Title:** Show HN: DiSan – Privacy-preserving text sanitization for multi-agent collaboration

**Body:**

```
DiSan is a text sanitization framework we built at Shanghai AI Lab for cross-organizational agent collaboration.

The core idea is disentanglement: learn two separate representations from text—one that captures what the text is about (role), and one that captures who wrote it (style). At inference time you throw away the style vector and decode from role only.

We evaluate it on a distributed RAG benchmark (synthetic finance docs across 7 non-IID agents) and on the Enron email corpus. The stylometric attribution numbers are the most striking: even the best PII detector + masking baseline leaves attribution at 67% F1, while DiSan brings it down to 22% (near the 14% random baseline).

It’s already integrated as a built-in module in Intern-Shannon, our multi-agent platform. The training code, RAG pipeline, and attack evaluation scripts are all in the repo.

Would love feedback from the HN community on the threat model and the federated training setup.
```

---

## 六、新闻通稿（媒体投稿版，~800 字）

**标题：上海人工智能实验室发布 DiSan：为跨机构 AI Agent 协作打造"隐私防火墙"**

**【上海，2025 年 X 月 X 日】** 随着大模型驱动的 AI Agent 在企业和机构间广泛部署，跨组织协作中的数据隐私保护成为关键挑战。上海人工智能实验室（Shanghai AI Lab）联合上海交通大学今日发布 **DiSan**（Disentangled Sanitization）框架，通过解耦表示学习技术，在保证业务语义的同时，系统性消除文本共享中的隐私泄露风险。

传统文本脱敏主要依赖标识符检测与掩码——例如将人名、公司名替换为通用标签。然而 DiSan 团队的研究表明，这种"打码"方式远不足以应对真实攻击。**即使使用业界最强的检测器掩码 19.2% 的文本内容，攻击者仍能通过文风、格式、句法等隐式分布特征，以超过 67% 的准确率识别数据来源。**

"隐私泄露不只是显式的姓名和账号，更是文本中无处不在的机构指纹，"论文第一作者介绍道，"DiSan 的创新在于，我们不再试图'擦掉'敏感信息，而是在表示层面将'任务内容'与'机构风格'彻底分离。"

DiSan 采用基于 LongT5 的双流编码器架构，将输入文本映射至两个正交子空间：保留任务语义的 **Role 子空间** 与捕获源标识特征的 **Style 子空间**。在联邦学习框架下，各参与方仅需上传聚合后的原型向量，原始文本始终保留在本地。推理阶段，系统仅使用 Role 子空间生成脱敏文本，Style 信息则被完全隔离。

在真实世界的 Enron 邮件语料（7 位作者，各 500 封邮件）上，DiSan 将基于 TF-IDF 的作者归因攻击成功率从 82.5% 降至 22.1%，降幅达 **73.2%**；在神经网络探测攻击下，降幅亦达到 **70.6%**，接近随机猜测水平。同时，在合成金融文档构建的分布式 RAG（检索增强生成）基准测试中，DiSan 将答案级 PII 暴露降低了 **20 倍**，并保持 **83%** 的答案忠实度。

与现有方法相比，DiSan 的优势显著：近期 NeurIPS 2024 提出的 JAMDEC 方法仅能降低 1.2% 的归因准确率，StyleRemix 最佳变体降低 11.5%，且存在"反噬"风险——某变体反而使归因准确率上升 20.8%。DiSan 的降幅是最佳基线的约 6 倍，是最新对比方法的 60 倍。

据悉，DiSan 已作为内置隐私保护模块集成于上海 AI Lab 的 **Intern-Shannon** 多智能体协作平台，支持内部协作、跨部门对齐与跨机构联盟三类场景。相关训练代码、评估管道与 RAG 数据集构建工具已面向研究社区开源。

**关于上海人工智能实验室**
上海人工智能实验室是我国人工智能领域的新型科研机构，致力于打造具有国际影响力的人工智能创新中心。

**媒体联络：**
xxx@pjlab.org.cn

---

## 七、视觉素材建议

| 素材 | 用途 | 建议 |
|------|------|------|
| 项目 Logo | 全渠道 | 基于 "DS" 字母 + 解耦双流的视觉隐喻 |
| 架构图 | 论文/博客 | Role/Style 双路编码器 + 联邦训练流程 |
| 对比表格 | 社媒/海报 | 原始 vs 掩码 vs DiSan 的三栏对比 |
| 效果图 | 演示 | 同一输入经过不同方法处理的输出对比 |
| 数字海报 | Twitter/即刻 | 20× / 73.2% / 83% 三个核心数字的极简设计 |

---

## 八、FAQ（快速应答）

**Q: DiSan 和差分隐私（DP）有什么区别？**
A: DP 是在梯度或输出上加噪声，提供数学保证但通常伴随 30-50% 的效用损失。DiSan 是在表示层面做结构化分离，实验中效用损失极小（<5%），更适合 RAG 等语义 fidelity 要求高的场景。

**Q: 是否需要集中原始数据训练？**
A: 不需要。DiSan 使用联邦原型对齐，各 Agent 本地训练，仅交换聚合后的原型向量。

**Q: 是否支持中文？**
A: 当前开源版本基于英文 LongT5 验证，架构本身与语言无关。迁移到中文仅需替换底座模型为 mT5/中文 LongT5。

**Q: 和 Intern-Shannon 的关系？**
A: DiSan 是 Intern-Shannon 的内置隐私模块之一，可在 Agent 返回证据前自动调用。

---

*本稿件由项目作者基于论文与开源代码整理，欢迎转发与改编，请注明出处。*
