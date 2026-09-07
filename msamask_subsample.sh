#!/bin/bash

INPUTFILE="7DSQ_2_msas_30"
OUTPUTDIR="${INPUTFILE}"
RANDOMSEED=1337

export PATH="/home/friedrich/localcolabfold/.pixi/envs/default/bin:${PATH}"

colabfold_batch \
  --num-recycle 1 \
  --num-models 5 \
  --model-type alphafold2 \
  --model-order 1,2,3,4,5 \
  --random-seed ${RANDOMSEED} \
  --num-seeds 1 \
  --max-msa 8:16 \
  ${INPUTFILE}\
  ${OUTPUTDIR}_16

colabfold_batch \
  --num-recycle 1 \
  --num-models 5 \
  --model-type alphafold2 \
  --model-order 1,2,3,4,5 \
  --random-seed ${RANDOMSEED} \
  --num-seeds 1 \
  --max-msa 16:32 \
  ${INPUTFILE}\
  ${OUTPUTDIR}_32
  
colabfold_batch \
  --num-recycle 1 \
  --num-models 5 \
  --model-type alphafold2 \
  --model-order 1,2,3,4,5 \
  --random-seed ${RANDOMSEED} \
  --num-seeds 1 \
  --max-msa 128:256 \
  ${INPUTFILE}\
  ${OUTPUTDIR}_256

colabfold_batch \
  --num-recycle 1 \
  --use-gpu-relax \
  --num-models 5 \
  --model-type alphafold2 \
  --model-order 1,2,3,4,5 \
  --random-seed ${RANDOMSEED} \
  --num-seeds 1 \
  --max-msa 512:1024 \
  ${INPUTFILE}\
  ${OUTPUTDIR}_1024


colabfold_batch \
  --num-recycle 1 \
  --use-gpu-relax \
  --num-models 5 \
  --model-type alphafold2 \
  --model-order 1,2,3,4,5 \
  --random-seed ${RANDOMSEED} \
  --num-seeds 1 \
  ${INPUTFILE}\
  ${OUTPUTDIR}_5120