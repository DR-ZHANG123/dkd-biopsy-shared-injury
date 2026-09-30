# Stage 20 — SCP 小管细胞内状态成分与 KPMP 适应不良 / 修复失败小管状态

运行：`bash scripts/stages/20_repair_state_all.sh`（programs → concord → decomp → donors → bulk → clinical → genes → summary）。
参数：`config/run.yaml` 的 `repair_state` 段（新增键）。每个子目录有 `PROVENANCE.json`；顶层 `PROVENANCE.json` 汇总全部子目录。
公共代码：`scripts/lib/repair_state.py`（状态定义、配对程序、供体检验）、`repair_shapley.py`（三层 Shapley 分解）、`repair_bulk.py`（bulk 读数）。

口径：状态 = KPMP atlas v2 `SubclassLevel2`。PT：正常（PT-S1/S2/S3）、aPT（adaptive/maladaptive）、frPT（failed-repair）、dPT（degenerative）、
cycPT（cycling）；TAL 同（dTAL = dC-TAL ∪ dM-TAL ∪ dTAL，MD 不计）。「修复失败」主定义 rf = a ∪ fr（rfPT、rfTAL）。
供体为重复单位；P 为供体标签置换 / 符号翻转；snRNA 为主（皮质、有 eGFR），scRNA 为复现（解离使应激 / 损伤状态占比偏高）。
SCP = stage 19 TUB 核心（与 KPMP 完全独立定义）。所有独立队列只作检验，不参与任何定义；状态程序只用 KPMP 单细胞、不读疾病标签。

## programs/（20a，`20_repair_state_programs.py`）
| 文件 | 用途 |
|---|---|
| `programs.tsv.gz` | 数据集 × 谱系 × 状态 × 供体子集（all / REF / CKD）× 基因：供体内配对（状态 vs 同供体正常细胞）log2FC、d_z、配对 t、BH-FDR、上调供体比例、上 / 下调调用。 |
| `program_sets.tsv` | 打分用状态程序（snRNA 全部供体，FDR<0.05 且 \|lfc\|≥1，按 d_z 每方向 top-100）及 scRNA 复现标记。 |
| `program_summary.tsv` | 每个程序的上 / 下调基因数、snRNA vs scRNA lfc Spearman、调用在 scRNA 的复现率、全部供体 vs 只用 REF / 只用 CKD 的程序一致性。 |
| `fractions.tsv.gz` | 供体 × 谱系 × 状态 的细胞数与占谱系比例（含 rf 合并与 altered = 全部非正常），附病理类别、取材方式、eGFR 分箱。 |

## concord/（20b，`20_repair_state_concord.py`）
| 文件 | 用途 |
|---|---|
| `lineage_de.tsv.gz` | PT / TAL（全部状态或只正常细胞）供体 pseudobulk 的 CKD vs REF（及取材匹配）逐基因 log2FC、置换 P（面板 b 散点）。 |
| `concordance.tsv` | 上述谱系内疾病变化与各状态程序的逐基因 Spearman（全部表达基因 / SCP 核心），零分布 = 供体标签置换。 |
| `bulk_overlap.tsv` | bulk SCP g_re 与状态程序：逐基因 Spearman、SCP 上 / 下调核心 ∩ 程序上 / 下调的 Fisher OR、表达量分层匹配随机集 z。 |
| `regression.tsv` | 供体水平：PT / TAL 谱系 SCP 分数 ~ logit(修复失败占比) / 正常细胞 SCP 分数 / 两者的 R² 与置换 P。 |
| `donor_lineage_scores.tsv` | 回归所用的供体值（谱系 SCP、正常细胞 SCP、修复失败占比；散点输入）。 |

## decomp/（20c，`20_repair_state_decomp.py`）
| 文件 | 用途 |
|---|---|
| `shapley_contrast.tsv` | SCP 组织分数组间差的 Shapley 分解（CKD / DKD / HKD / AKI vs REF、取材匹配）：块（composition、repair_state_fraction、PT_TAL_normal_cell_profile、PT_TAL_repair_cell_profile、PT_TAL_other_altered_profile、other_lineage_*）与单因子（factor:*）的 Δ、占总差 / 占细胞内状态差的份额、供体 bootstrap CI、置换 P（面板 c 主表）。 |
| `shapley_donor.tsv.gz` | 每个供体的 Shapley 值（供体水平分解，可画堆叠条）。 |
| `shapley_core_genes.tsv.gz` | 每个 SCP 核心基因组织 log2CPM 的 CKD vs REF Δ 按因子 / 块分解（关键基因的变化来自修复失败细胞扩增还是正常细胞自身）。 |
| `sanity.tsv` | 本分解 full 分数与 stage 19c full 分数的一致性（Pearson）。 |

## donors/（20d，`20_repair_state_donors.py`）
| 文件 | 用途 |
|---|---|
| `donor_table.tsv` | 供体宽表：各状态占比、类别、取材、eGFR 中点、stage 19c SCP（full / state_only / comp_only）（面板 d/e 输入）。 |
| `category_tests.tsv` | 各病理类别（DKD、HKD、OTHER、CKD_unadj、AKI、DM_R、CKD；取材匹配 CKD_perc vs REF_perc）vs REF 的 logit 占比差、Hedges g、AUROC、置换 P。 |
| `correlations.tsv` | 状态占比与 SCP 分数、eGFR 的 Spearman（置换 P）；对 eGFR 另给控制 SCP 的偏 Spearman。 |

## bulk/（20e，`20_repair_state_bulk.py`）
| 文件 | 用途 |
|---|---|
| `sample_measures.tsv.gz` | 每个 bulk 样本：SCP、状态程序分数（full / noSCP = 剔除全部 SCP 核心基因 / 只上调）、BayesPrism θ 份额（logit）。 |
| `unit_contrasts.tsv` | 单元内每病种 vs 对照：Hedges g、方差、AUROC、置换 P。 |
| `meta_contrasts.tsv` | 跨单元 DL 合并（面板 f 森林图）与同号比例。 |
| `scp_relation.tsv` | 单元内各读数与 SCP 分数的 Spearman（全体 / 病人内）。 |
| `resolvability.tsv` | 状态在 bulk 中是否可分辨：θ 份额中位数、<1% 样本比例、θ 份额与程序分数的一致性。 |

## clinical/（20f，`20_repair_state_clinical.py`；独立数据，只运行一次）
| 文件 | 用途 |
|---|---|
| `tests.tsv` | GSE175759 eGFR、GSE142025 晚期 vs 早期、GSE115857 IgAN 分级、KPMP 切片 bulk eGFR / 蛋白尿、GSE166239 eGFR / 蛋白尿：疾病 vs 对照 AUROC、Spearman（置换 P）、控制 SCP 的偏 Spearman / 残差 AUROC；KPMP 切片 bulk 另给去掉与单细胞图谱重叠参与者的结果。 |
| `scores.tsv.gz` | 每个独立样本的全部读数（面板 g 散点 / 箱线）。 |

## genes/（20g，`20_repair_state_genes.py`）
| 文件 | 用途 |
|---|---|
| `driver_calls.tsv` | 每个 SCP TUB 核心基因 × 状态（aPT、frPT、aTAL、frTAL）的程序效应与驱动基因判定（snRNA 显著且 scRNA 同号复现、方向与 SCP 一致）。 |
| `key_genes.tsv` | 驱动基因 + 经典损伤 / 修复标志：KPMP 状态特异性（最高类型、修复失败 vs 正常 / 其余类型）、CKD 组织计数中来自修复失败细胞的份额、正常细胞内 CKD vs REF、组织 Δ 的 Shapley 份额、bulk meta g、独立队列效应与同号队列数（面板 h）。按修复失败可归因的组织 Δ 排序。 |
| `state_expression.tsv.gz` | 关键基因在各类型（PT / TAL 拆为 正常 / 修复失败 / 其他改变）的均值 log2CPM（面板 a 热图）。 |
| `independent_gene_effects.tsv.gz` | 关键基因在独立队列中的逐基因效应（长表）。 |

## figdata/（20h，`20_repair_state_summary.py`；主图直接读这些表）
`panel_a_*`（状态程序概况、关键基因状态表达）、`panel_b_*`（逐基因一致性散点与统计、bulk 重叠）、`panel_c_*`（Shapley 块与单因子）、
`panel_d_*`（供体占比与类别检验）、`panel_e_*`（占比与 SCP / eGFR 相关、谱系回归）、`panel_f_*`（bulk 跨队列合并、单元效应、与 SCP 关系、可分辨性）、
`panel_g_*`（独立临床）、`panel_h_key_genes.tsv`；`key_numbers.tsv` = 正文关键数字。

## 读表注意
- Shapley 的「占细胞内状态」份额 = 块 Δ / (full Δ − composition Δ)；各块相加 = 1（交互项已按 Shapley 平均分配）。
- 谱系回归中「正常细胞 SCP」的 R² 含部分–整体成分（正常细胞占谱系细胞多数），解读以 Shapley 分解为准。
- bulk θ 状态层：ERCB 芯片上 rfPT 份额中位数约 0.2–0.3%（>60% 样本 <1%），TAL 状态层被 dTAL 吸收（约 90%）——绝对份额不可解读，只作秩；程序分数与 θ 份额在 PT 上一致（Spearman 0.5–0.9），在 aTAL 上弱。
- 状态程序分数与 SCP 在 bulk 中高度共线（单元内 Spearman 0.6–0.9，剔除 SCP 基因后仍如此）；增量信息用偏相关判断。
