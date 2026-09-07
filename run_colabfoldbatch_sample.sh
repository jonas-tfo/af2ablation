#!/bin/bash

INPUTFILE="runs/7DSQ_2/original"
OUTPUTDIR="${INPUTFILE}__"
RANDOMSEED=0

export PATH="/home/friedrich/localcolabfold/.pixi/envs/default/bin:${PATH}"

colabfold_batch \
  --num-recycle 1 \
  --num-models 1 \
  --model-type alphafold2 \
  --model-order 5 \
  --random-seed ${RANDOMSEED} \
  --max-msa 16:32 \
  --num-seeds 1 \
  ${INPUTFILE}.a3m \
  __${OUTPUTDIR}
