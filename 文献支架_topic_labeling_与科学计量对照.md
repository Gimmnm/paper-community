# 文献支架（topic_labeling / 可解释性 / 科学计量对照）推荐清单（含可点击链接与 PDF）

> 用途：给当前 `topic_labeling` 后处理模块、结构解释与科学计量对照提供文献依据。  
> 说明：优先给出**官方页面/论文页**与**可直接点击的 PDF 链接**；部分出版社 PDF 可能需要机构权限。

---

## A. 直接对应本项目当前工作：Topic / Cluster 自动命名（Labeling）

### 1) Mei, Shen, Zhai (KDD 2007)
**Automatic Labeling of Multinomial Topic Models**  
- 作用：经典起点；把“topic labeling”明确作为独立问题来研究（不是主题建模本体）
- 论文页（ACM DOI）：<https://dl.acm.org/doi/10.1145/1281192.1281246>
- 公开 PDF（作者页面）：<https://timan.cs.illinois.edu/czhai/pub/kdd07-label.pdf>

### 2) Lau, Grieser, Newman, Baldwin (ACL 2011)
**Automatic Labelling of Topic Models**  
- 作用：候选生成 + 排序（与你当前“候选 + 排序”的工程逻辑高度相关）
- 论文页（ACL Anthology）：<https://aclanthology.org/P11-1154/>
- PDF：<https://aclanthology.org/P11-1154.pdf>

### 3) Aletras, Stevenson (ACL 2014)
**Labelling Topics using Unsupervised Graph-based Methods**  
- 作用：无监督图方法（PageRank/图排序）做标签选择，适合支撑“非规则”路线
- 论文页（ACL Anthology）：<https://aclanthology.org/P14-2103/>
- PDF：<https://aclanthology.org/P14-2103.pdf>

### 4) Bhatia, Lau, Baldwin (COLING 2016)
**Automatic Labelling of Topics with Neural Embeddings**  
- 作用：embedding 路线；可作为你后续升级方向（摆脱手工规则）
- 论文页（ACL Anthology）：<https://aclanthology.org/C16-1091/>
- PDF：<https://aclanthology.org/C16-1091.pdf>
- arXiv（可下载 PDF）：<https://arxiv.org/abs/1612.05340>

### 5) Alokaili, Aletras, Stevenson (SIGIR 2020)
**Automatic Generation of Topic Labels**  
- 作用：生成式标签（比“从候选中选”更进一步），可放在“未来工作”
- 论文页（ACM DOI）：<https://dl.acm.org/doi/10.1145/3397271.3401185>
- Publisher PDF（可能需权限）：<https://dl.acm.org/doi/pdf/10.1145/3397271.3401185>
- arXiv（可下载 PDF）：<https://arxiv.org/abs/2006.00127>
- 作者机构公开 PDF（White Rose）：<https://eprints.whiterose.ac.uk/id/eprint/161293/1/sigir20_areej.pdf>

---

## B. 主题解释性 / 评估（用于回答“置信度不是概率”以及后续评价）

### 6) Mimno et al. (EMNLP 2011)
**Optimizing Semantic Coherence in Topic Models**  
- 作用：主题可解释性评估（semantic coherence）的经典文献
- 论文页（ACL Anthology）：<https://aclanthology.org/D11-1024/>
- PDF：<https://aclanthology.org/D11-1024.pdf>
- 作者页面 PDF：<https://mimno.infosci.cornell.edu/papers/mimno-semantic-emnlp.pdf>

### 7) Röder, Both, Hinneburg (WSDM 2015)
**Exploring the Space of Topic Coherence Measures**  
- 作用：系统梳理 coherence 指标，适合你在答辩中说明“可引入更系统评价”
- 论文页（ACM DOI）：<https://dl.acm.org/doi/10.1145/2684822.2685324>
- Publisher PDF（可能需权限）：<https://dl.acm.org/doi/pdf/10.1145/2684822.2685324>
- 公开 PDF（AKSW）：<https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf>

---

## C. 科学计量图谱里的“解释层”先例（证明 topic_labeling 不是脱离文献的随意工程）

### 8) Chen, Ibekwe-SanJuan, Hou (JASIST 2010 / arXiv preprint)
**The Structure and Dynamics of Co-Citation Clusters: A Multiple-Perspective Co-Citation Analysis**  
- 作用：非常关键；把网络可视化、聚类、**自动 cluster labeling**、摘要整合到同一解释框架
- arXiv 页面：<https://arxiv.org/abs/1002.1985>
- arXiv PDF：<https://arxiv.org/pdf/1002.1985>
- 出版信息索引（如需正式版检索）：<https://yonsei.elsevierpure.com/en/publications/the-structure-and-dynamics-of-cocitation-clusters-a-multiple-pers/>

### 9) Chen (JASIST 2006)
**CiteSpace II: Detecting and Visualizing Emerging Trends and Transient Patterns in Scientific Literature**  
- 作用：科学计量可视化与知识图谱解释的经典方法背景
- 预印本 PDF（作者页面）：<https://cluster.cis.drexel.edu/~cchen/citespace/doc/JASIST_CiteSpace_preprint.pdf>
- 文献索引页（ideas.repec）：<https://ideas.repec.org/a/bla/jamist/v57y2006i3p359-377.html>

### 10) VOSviewer（Van Eck & Waltman 路线）
**VOSviewer: Visualizing scientific landscapes**（工具与方法生态入口）  
- 作用：说明“citation / co-citation / co-authorship / text mining + 可视化解释”在科学计量里非常常见
- 官方网站：<https://www.vosviewer.com/>
- 官方手册 PDF（1.6.19）：<https://www.vosviewer.com/documentation/Manual_VOSviewer_1.6.19.pdf>
- 在线示例（含 co-citation/citation 可视化案例）：<https://app.vosviewer.com/docs/examples/>

---

## D. 回应“为什么 topic modeling 与 citation/co-citation 结构不一致”的关键比较文献

### 11) Leydesdorff, Nerghes (2017; arXiv 2015)
**Co-word Maps and Topic Modeling: A Comparison Using Small and Medium-Sized Corpora**  
- 作用：直接支持“co-word / 网络映射 与 topic modeling 结果可能显著不同”的论点
- arXiv 页面：<https://arxiv.org/abs/1511.03020>
- arXiv PDF：<https://arxiv.org/pdf/1511.03020>
- 作者公开 PDF（镜像）：<https://www.adinanerghes.com/resources/Articles/Co-word_Maps_and_Topic_Modeling_A_Compar.pdf>

### 12) Xie, Waltman (Scientometrics 2025; arXiv 2023 preprint)
**A comparison of citation-based clustering and topic modeling for science mapping**  
- 作用：非常贴近你的问题；直接比较 citation-based clustering 与 topic modeling 的 science mapping 结果
- Springer 页面（正式发表）：<https://link.springer.com/article/10.1007/s11192-025-05324-z>
- arXiv 页面（预印本）：<https://arxiv.org/abs/2309.06160>
- arXiv PDF：<https://arxiv.org/pdf/2309.06160>

---

## E. 连接你参考论文方法脉络：Mixed-SCORE / Leiden（结构层基础）

### 13) Jin, Ke, Luo (Mixed-SCORE)
**Mixed Membership Estimation for Social Networks**  
- 作用：Mixed-SCORE 的核心方法文献；连接你参考论文的混合成员谱方法脉络
- arXiv 页面：<https://arxiv.org/abs/1708.07852>
- arXiv PDF：<https://arxiv.org/pdf/1708.07852>
- 作者页面 PDF（可选）：<https://www.stat.cmu.edu/~jiashun/Research/Recents/MSCORE-arXiv.pdf>

### 14) Traag, Waltman, van Eck (Scientific Reports 2019)
**From Louvain to Leiden: guaranteeing well-connected communities**  
- 作用：Leiden 社区发现算法的标准引用（你项目结构层分析的基础）
- Nature 页面：<https://www.nature.com/articles/s41598-019-41695-z>
- arXiv 页面：<https://arxiv.org/abs/1810.08473>
- arXiv PDF：<https://arxiv.org/pdf/1810.08473>

---

## F. 你可以如何在论文/答辩里使用这套文献支架（建议写法）

### F.1 三层定位（最稳妥）
1. **核心建模层**（你们已实现）
   - Leiden（社区发现）
   - Topic-SCORE（社区聚合文本主题建模）
   - 多分辨率对齐（相似度 + Hungarian）

2. **解释层（本次讨论重点）**
   - Topic / cluster labeling（Mei 2007; Lau 2011; Aletras 2014; Bhatia 2016; Alokaili 2020）
   - 主题可解释性评价（Mimno 2011; Röder 2015）

3. **科学计量对照层**
   - 共被引/知识图谱中的聚类解释与自动命名（Chen 2010; CiteSpace; VOSviewer）
   - TM 与 citation-based clustering 差异（Leydesdorff & Nerghes; Xie & Waltman）

### F.2 一个可直接复用的表述（中文）
> 本项目中的 `topic_labeling` 不属于参考文献《统计学家的共被引与共著网络》的原始建模流程，而是位于 Topic-SCORE 之后的解释层模块。该扩展并非脱离文献依据的工程拼接：在 topic modeling 与科学计量图谱分析文献中，自动标签生成（topic/cluster labeling）已被广泛作为提升可解释性的后处理步骤。因此，我们将其作为结构分析结果的辅助解释工具，而不将其作为主要结构发现证据。  

---

## G. 阅读优先级（如果时间有限）

### 必读（与你当前工作最相关）
- Mei et al. 2007（topic labeling 起点）
- Lau et al. 2011（候选+排序）
- Chen et al. 2010（科学计量里的自动 cluster labeling 先例）
- Xie & Waltman（citation clustering vs topic modeling 比较）

### 第二优先级（答辩加分）
- Mimno et al. 2011（可解释性/coherence）
- Röder et al. 2015（coherence 指标体系）
- Leiden 2019（结构层基础）
- Mixed-SCORE（参考论文方法脉络）

### 未来工作扩展
- Bhatia et al. 2016（embedding 标签）
- Alokaili et al. 2020（生成式标签）

