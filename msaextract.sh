#!/bin/bash

INPUTFILE="7DSQ_2"
OUTPUTDIR="${INPUTFILE}"
RANDOMSEED=0

export PATH="/home/friedrich/localcolabfold/.pixi/envs/default/bin:${PATH}"

colabfold_batch \
  --msa-only \
  ${INPUTFILE}.fasta \
  ${OUTPUTDIR}_32_msa 

