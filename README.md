# kidney-injury-repair

Code, configuration and result tables for the study

> **A tubular injury–repair response is the largest shared source of variation in biopsy transcriptomes across chronic kidney diseases**

The study combines 1,462 unique kidney biopsy specimens from public microarray and RNA-sequencing series with the Kidney Precision Medicine Project (KPMP) single-cell and single-nucleus atlases. It defines the injury–repair response shared by native biopsies of different chronic kidney diseases, separates changes in cell proportions from changes within cell lineages, quantifies the contribution of adaptive and failed-repair tubular states, tests the response in independent cohorts and against clinical measures, and compares diseases after adjustment for the response.

## Data

All data are public. Accessions, platforms and roles are listed in `metadata/data_sources.tsv`; the download scripts record MD5 checksums.

- Bulk kidney biopsy transcriptomes (GEO): GSE30528, GSE30529 (both part of GSE30122), GSE32591, GSE37460, GSE47183, GSE69438, GSE93798, GSE96804, GSE99339, GSE104948, GSE104954, GSE108109, GSE108112, GSE115857, GSE133288, GSE142025, GSE175759, GSE182380, GSE200818 (GSE116626 was screened and excluded).
- Independent test cohorts: GSE162830, GSE166239 and KPMP biopsy-section bulk RNA-sequencing (open-access tier of the KPMP atlas repository).
- Single-cell references: KPMP atlas v2.0 single-nucleus and single-cell data from CZ CELLxGENE Discover; GSE131882 and GSE209781.
- Protein interactions: STRING v12.0.
- Curated published DKD gene signatures: `metadata/published_dkd_signatures.tsv`.

Raw and intermediate data are written to `data/` (not tracked).

## Environments

```bash
conda env create -f environment.yml          # Python analysis environment, named "kir"
```

BayesPrism (stage 16) runs in a separate R environment (R 4.3 with BayesPrism 2.2.3), referred to as `kir_bayesprism` in `scripts/stages/16_deconv_run.py`.
The upstream single-cell and CEL processing uses the environments in `upstream/`.

## Upstream processing

`upstream/scripts/` contains the steps that produce two inputs of the main pipeline:

- `03_rma_from_cel.R`, `03b_rma_summarize.py`, `03c_validate_rma.py`: RMA of the GSE30528 and GSE30529 CEL files (the deposited matrices are probe-centred) and validation against the deposited processing.
- `04_convert_scrna_rds.R`, `06_scrna_qc.py`, `07_cell_annotation.py`, `08_pseudobulk_de.py`: quality control, annotation and donor-level pseudobulk of GSE131882 and GSE209781.

Their outputs are read from `upstream/data_processed/` (paths in `config/run.yaml`).

## Running the pipeline

```bash
bash run_all.sh            # stages with an existing PROVENANCE.json are skipped
bash run_all.sh --force    # rerun everything
```

All parameters are in `config/run.yaml`; scripts read them and contain no hard-coded analysis settings. Every stage writes its tables to `results/<stage>/` together with a `PROVENANCE.json` (input and output checksums, seed, configuration hash, package versions). The global seed is 20260928.

| Stage | Content |
|---|---|
| 00–05 | Downloads, ingestion and scale checks, sample-level duplicate detection, diagnostic labels, within-sample ranks, prior gene graphs |
| 11, 11b | Injury–repair score estimated without the evaluated cohort and without DKD; split-control and cross-cohort tests |
| 12 | Scoring of published DKD signatures |
| 13, 14 | Cell-type enrichment; KPMP marker reference and donor-level tests |
| 15 | Disease-specific classifier: development on non-DKD tasks, frozen design (`results/15_model/FROZEN_DESIGN.json`), single evaluation on DKD |
| 16, 16b | BayesPrism deconvolution; simulation-trained composition estimates |
| 17, 17a, 17b | Independent cohorts: candidate screening, planned tests (`results/17_independent/PLAN*.json`) and gene-set replication |
| 18 | Gene- and program-level interpretation of the classifier |
| 19 | Injury–repair response genes, composition versus within-lineage change, replication and clinical association, disease-specific signals after adjustment |
| 20 | Adaptive and failed-repair tubular states: programs, Shapley decomposition, donor, bulk and clinical analyses |
| 21, 22 | Robustness and sensitivity analyses (variance share, transplant gene sets, procurement-sensitive genes, signal injection, empirical nulls, state-level decomposition, interaction terms, eGFR sensitivity, bootstrap intervals, deconvolution recovery) |

Stages 06–09 contain an exploratory graph-encoder pretraining that is not used in the reported analyses.

## Figures

`scripts/figures/` draws every main and supplementary figure from the result tables only; the exact plotted values are written to `figures/source_data/`. `scripts/figures/check_overlap.py` checks rendered text for overlaps and out-of-bounds labels.

## Licence

MIT (see `LICENSE`).
