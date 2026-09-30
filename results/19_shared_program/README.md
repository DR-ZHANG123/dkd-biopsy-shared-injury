# Stage 19 — 泛 CKD 共享疾病–对照程序（SCP）与病种特异残差

运行：`bash scripts/stages/19_shared_program_all.sh`（core → cells → kpmp → replicate → residual → summary）。
参数：`config/run.yaml` 的 `shared_program` 段。每个子目录有自己的 `PROVENANCE.json`，顶层 `PROVENANCE.json` 汇总子目录与 figdata。
来源单元 = 带对照的 ERCB / NEPTUNE / IgAN 芯片单元（GLOM 4 个、TUB 3 个）；不读 DKD 标签，独立队列（GSE175759、GSE142025、
GSE115857、GSE162830、KPMP 切片 bulk、GSE166239）一律不作来源。核心阈值在查看任何独立数据前写入 config。

## core/（19a，`19_shared_program_core.py`）
| 文件 | 用途 |
|---|---|
| `meta_{GLOM,TUB}.tsv.gz` | 全部基因的来源单元 g、DL 随机效应 g_re / I² / HK P / 同号来源比例 / 留一稳定性与核心标记（火山图输入）。 |
| `core_genes.tsv` | SCP 核心基因表（方向、效应、异质性、逐病种同号率、是否两区室共享、是否经典即早基因）。 |
| `sources.tsv` | 每个来源单元的病人 / 对照数与共同基因数。 |
| `stability.tsv` | 来源两两 Spearman 与留一来源后核心保留率（核心稳健性）。 |
| `disease_vs_control_effects.tsv.gz` | 来源单元内逐病种 vs 对照的逐基因 g（病种一致性的原始数据）。 |
| `disease_concordance.tsv` | 每个病种 vs 对照、以及 DKD 外部检验（DKD vs 对照）与去掉该单元后 SCP 的 Spearman / 核心同号率 / SCP AUROC。 |

## cells/（19b，`19_shared_program_cells.py`）
| 文件 | 用途 |
|---|---|
| `kpmp_localisation_all_genes.tsv.gz` | KPMP 健康参考中每个基因的最高表达细胞类型、tau、最高类型份额（snRNA / scRNA）。 |
| `core_localisation.tsv` | 核心基因附带其细胞类型定位。 |
| `core_top_type_enrichment.tsv` | 上 / 下调核心按最高表达类型的富集（Fisher OR、FDR）。 |
| `marker_program_z.tsv` | KPMP 各细胞类型标志基因集上 SCP g_re 的富集 z。 |
| `composition_r2.tsv` | 每单元 SCP 分数 ~ BayesPrism 组成（CLR）的 R² / CV R²（随机基因集对照）与组成调整前后的疾病 vs 对照 AUROC。 |
| `composition_celltype_rho.tsv` | 每单元各细胞类型比例与 SCP 分数的 Spearman（全体 / 病人内）。 |
| `stage16_axis_r2_summary.tsv` | stage 16 已有的共享轴 ~ 组成解释度汇总（原样复制，便于同表）。 |

## kpmp/（19c，`19_shared_program_kpmp.py`；供体为重复单位，P 为供体标签置换）
| 文件 | 用途 |
|---|---|
| `groups.tsv` | 每个数据集的供体分组人数（REF、CKD、经皮活检子集、健康手术取材）。 |
| `gene_type_de.tsv.gz` | 核心基因在每个细胞类型内各对比的 log2FC / g / 置换 P。 |
| `set_direction.tsv` | 核心基因集在每个细胞类型内的方向一致率 vs 背景（置换 P）。 |
| `abundance.tsv` | 各细胞类型比例的 CKD vs REF（及取材匹配）效应量与置换 P。 |
| `gene_class.tsv` | 每个核心基因的来源判定（细胞内状态 / 组成 / 两者 / 反向 / 未定）。 |
| `decomp_scores.tsv.gz` | 每个供体在 full / comp_only / state_only / 逐类型反事实组织上的 SCP 分数。 |
| `decomp_tests.tsv` | 上述各场景的 CKD vs REF（含取材匹配、分病种）效应、AUROC、置换 P（组成 vs 状态分解主表）。 |
| `egfr.tsv` | snRNA 供体 SCP 分数（full / comp_only / state_only）与 eGFR 分箱中点的 Spearman（随机集百分位）。 |
| `procurement.tsv.gz` | 组织 pseudobulk 逐基因：健康手术 vs 健康经皮（取材效应）、CKD 经皮 vs 健康经皮、CKD vs REF。 |
| `core_procurement.tsv` | 核心基因的取材敏感标记与取材匹配下的方向一致性。 |

## replicate/（19d，`19_shared_program_replicate.py`；独立数据，只运行一次）
| 文件 | 用途 |
|---|---|
| `auroc.tsv` | 各独立队列 疾病 vs 健康 的 SCP AUROC（bootstrap CI、1000 个同大小随机集百分位；full / 去取材敏感 / 去即早基因；主区室与另一区室）。 |
| `clinical.tsv` | 与临床的关联：GSE175759 / GSE166239 eGFR、蛋白尿，KPMP 切片 bulk eGFR / 蛋白尿分级，GSE142025 晚期 vs 早期，GSE115857 IgAN 分级。 |
| `scores.tsv` | 每个独立样本的 SCP 分数（画散点 / 箱线图用）。 |

## residual/（19e，`19_shared_program_residual.py`）
| 文件 | 用途 |
|---|---|
| `unit_effects.tsv.gz` | 每单元 病种 vs 其余病人 的逐基因 g（SCP 调整 / 未调整）。 |
| `disease_scp_shift.tsv` | 每单元每病种相对其余病人的 SCP 分数差（病种处在共享程序的哪一端）。 |
| `residual_meta.tsv.gz` | 扣除 SCP 后病种特异效应的跨单元随机效应合并。 |
| `top_genes.tsv` | 每病种扣除 SCP 后的 top 上 / 下调基因。 |
| `signatures.tsv` | 独立检验所用病种残差签名（每方向 50 基因）。 |
| `cell_programs.tsv` | 病种残差在 KPMP 标志程序上的富集 z（关键细胞程序）。 |
| `consistency.tsv` | 与 stage 13（外部轴调整）和 stage 18（RRG-ID head）的一致性。 |
| `independent_tests.tsv` | 病种残差签名在独立数据中的病人内 AUROC、SCP 调整 AUROC、随机签名百分位与复现判定。 |

## figdata/（19f，`19_shared_program_summary.py`；主图直接读这些表）
| 文件 | 用途 |
|---|---|
| `fig1_core_volcano.tsv.gz` | 面板：SCP meta 火山图（两区室，核心 + 代表基因标注）。 |
| `fig1_core_forest.tsv` | 面板：代表基因在各来源单元的 g 与合并 g（森林图）。 |
| `fig1_disease_concordance.tsv` | 面板：各病种与 DKD（外部检验）对 SCP 的一致性热图。 |
| `fig2_core_localisation.tsv` | 面板：核心基因最高表达细胞类型富集（OR）。 |
| `fig2_marker_program_z.tsv` | 面板：细胞类型标志程序上的 SCP 富集 z。 |
| `fig2_composition_r2.tsv` | 面板：组成对 SCP 分数的 CV R² 与组成调整前后 AUROC。 |
| `fig2_kpmp_decomposition.tsv` | 面板：KPMP 组成 vs 细胞内状态分解（总体与逐细胞类型的 Δ 分数）。 |
| `fig2_kpmp_gene_class.tsv` | 面板：核心基因来源判定按细胞类型计数。 |
| `fig2_kpmp_abundance.tsv` | 面板：KPMP 细胞类型丰度变化。 |
| `fig2_kpmp_set_direction.tsv` | 补充面板：细胞类型内核心基因方向一致率。 |
| `fig3_replication_auroc.tsv` | 面板：独立复现 AUROC 森林图（含随机集百分位）。 |
| `fig3_clinical.tsv` | 面板：临床关联汇总（含 KPMP snRNA 供体 eGFR）。 |
| `fig3_kpmp_egfr_scatter.tsv` | 面板：KPMP snRNA 供体 SCP 分数 vs eGFR 散点。 |
| `fig3_scores.tsv` | 面板：独立队列样本分数分布。 |
| `fig4_residual_programs.tsv` | 面板：病种残差细胞程序热图。 |
| `fig4_residual_top_genes.tsv` | 面板：病种残差 top 基因。 |
| `fig4_residual_tests.tsv` | 面板：病种残差独立检验（复现 / 不复现 / 描述性）。 |
| `fig4_disease_scp_shift.tsv` | 面板：各病种在 SCP 上的位置。 |
| `key_numbers.tsv` | 正文关键数字（核心数、R²、KPMP 分解 AUROC、独立复现 AUROC）。 |
