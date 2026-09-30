#!/usr/bin/env bash
# Full pipeline in the order used for the manuscript.
# Usage: bash run_all.sh            (stages whose PROVENANCE.json exists are skipped)
#        bash run_all.sh --force
set -euo pipefail
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate dkd-shared-injury
FORCE=0; [[ "${1:-}" == "--force" ]] && FORCE=1
mkdir -p logs
run() { local tag=$1; shift
  if [[ $FORCE == 0 && -f results/$tag/PROVENANCE.json ]]; then echo "done  $tag"; return; fi
  echo ">>    $tag"; "$@" 2>&1 | tee "logs/$tag.log"; }

# Data acquisition (public GEO, STRING, KPMP / CELLxGENE)
[[ $FORCE == 0 && -s results/00_download/checksums.tsv ]] || bash scripts/stages/00_download.sh
[[ $FORCE == 0 && -s results/00_download/kpmp_checksums.tsv ]] || bash scripts/stages/00b_download_kpmp.sh
[[ $FORCE == 0 && -s results/00_download/kpmp_lmd_checksums.tsv ]] || bash scripts/stages/00c_download_kpmp_lmd.sh

# Specimens, labels, ranks and prior gene graphs
run 01_ingest        python scripts/stages/01_ingest.py
run 02_overlap       python scripts/stages/02_overlap_audit.py
run 03_labels        python scripts/stages/03_labels.py
run 04_ranks         python scripts/stages/04_gene_universe_ranks.py
run 05_graphs        python scripts/stages/05_prior_graphs.py

# Injury-repair response, published signatures and cell-type decomposition
run 11_injury        python scripts/stages/11_injury_axis.py
run 11b_shared_axis  python scripts/stages/11b_shared_axis_tests.py
run 12_signature_audit python scripts/stages/12_signature_audit.py
run 13_celltype      python scripts/stages/13_celltype_decomposition.py
run 14_kpmp          python scripts/stages/14_kpmp_validation.py

# Classifier of the adjusted diagnosis-associated signal (developed on non-DKD tasks, evaluated once on DKD)
run 15_model/dev     python scripts/stages/15_dev.py --families E1,E2,E4
python scripts/stages/15_summarize.py
[[ -f results/15_model/final_dkd/predictions.tsv ]] || python scripts/stages/15_final_dkd.py cuda:0
python scripts/stages/15_final_bootstrap.py

# Deconvolution (BayesPrism in the dkd-shared-injury-bayesprism environment) and a simulation-trained alternative
for s in prepare run plausibility eval summarize; do python scripts/stages/16_deconv_$s.py; done
python scripts/stages/16b_deconv_fast.py; python scripts/stages/16b_deconv_fast_eval.py

# Independent cohorts
python scripts/stages/17a_independent_candidates.py
[[ -f results/17_independent/predictions.tsv ]] || python scripts/stages/17_independent_test.py
[[ -f results/17b_key_genes/replication.tsv ]] || python scripts/stages/17b_key_gene_replication.py

# Interpretation, response genes, adaptive and failed-repair states
bash scripts/stages/18_interpret_all.sh cuda:0
bash scripts/stages/19_shared_program_all.sh
bash scripts/stages/20_repair_state_all.sh

# Robustness and sensitivity analyses
for s in a1_share a1_pca a2_genesets a2_transplant kpmp shapley a3_indep a4_injection a5_ratio a6_fractions a7_bulk; do
  python scripts/stages/21_robustness_$s.py; done
for s in s1_egfr s2_auroc_ci s3_deconv; do python scripts/stages/22_sensitivity_$s.py; done

# Pathways of the response; external-score gene number and duplicate-threshold sensitivity
for s in 24a_response_pathways 24b_external_score_topk 24c_duplicate_thresholds; do python scripts/stages/$s.py; done

# Figures (read result tables only; write figures/ and figures/source_data/)
for f in scripts/figures/Fig*.py scripts/figures/graphical_abstract.py; do python "$f"; done
python scripts/figures/check_overlap.py
