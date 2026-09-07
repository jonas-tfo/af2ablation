#!/usr/bin/env bash

# fetch msas and write the a3m files to targets/<target>/<subdir>/uniref.a3m
python -m afpert.prep.source_msas

# run the perturbation of query and msa for all targets in targets/ and write to runs/
# python -m afpert.scripts.cross_perturb_all \
#     --targets-dir targets/ \
#     --query-masks 5,15,50,90 \
#     --msa-masks 5,15,30,50 \
#     --depths 16,128,5120 \
#     --runs-root runs / \
#     --overwrite

#python -m afpert.scripts.cross_perturb_all \
#    --targets-dir targets/ \
#    --query-masks 0,5 \
#    --msa-masks   0,5,10,15,30 \
#    --depths      32,256,1024,5120 \
#    --num-seeds   5 \
#    --runs-root   runs/

python -m afpert.scripts.cross_perturb_all \
    --targets-dir targets/ \
    --query-masks 0,5,15,50,90 \
    --msa-masks   0,5,15,30,50 \
    --depths      32,256,1024,5120 \
    --num-seeds   5 \
    --runs-root   runs/ \
    --overwrite

# run the predictions for all perturbed inputs (so the query, mask, subsample combis) in runs/
python -m afpert.scripts.run_predictions --depths  32,256,1024,5120

# generate csv (just tm scores now)
python3 -m afpert.scripts.evaluate_all

# 28.05
# python -m afpert.scripts.evaluate_target --tm --rmsd --structure1 7DSQ --structure2 6IRS --target-dir 7DSQ_2 --method alphamask
python -m afpert.scripts.evaluate_target --tm --rmsd --structure1 6IRS_B --structure2 7DSQ_B --target-dir 7DSQ_2 --method custom

python -m afpert.scripts.visualize --tm --rmsd --target 7DSQ_2 --method alphamask



uv run python -m afpert.scripts.evaluate_dataset --rmsf --dataset oc23.csv --method afsample2
uv run python -m afpert.scripts.visualize_dataset --rmsf --rmsf-aggregate --dataset oc23.csv --method afsample2


uv run python -m afpert.scripts.evaluate_dataset --tm --dataset oc23.csv --method afsample2

# rebuild with tmalign
uv run python -m afpert.scripts.score_oc23
uv run python -m afpert.scripts.benchmark_all
uv run python -m afpert.visualize.boxplots
