#!/usr/bin/env bash

# fetch msas and write the a3m files to targets/<target>/<subdir>/uniref.a3m
python -m afpert.prep.source_msas

# run the perturbation of query and msa for all targets in targets/ and write to runs/
python -m afpert.scripts.cross_perturb_all \
    --targets-dir targets/ \
    --query-masks 5,15,50,90 \
    --msa-masks 5,15,30,50 \
    --depths 16,128,5120 \
    --runs-root runs/

# run the predictions for all perturbed inputs (so the query, mask, subsample combis) in runs/
python -m afpert.scripts.run_predictions
