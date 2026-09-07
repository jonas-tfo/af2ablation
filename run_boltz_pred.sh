#!/bin/bash

source /home/friedrich/boltz-gb10-spark/.env/bin/activate

INPUTFILE="7DSQ_1"
OUTPUTDIR="${INPUTFILE}"
RANDOMSEED=0

boltz predict \
  --recycling_steps 1\
  --use_msa_server \
  --model boltz2 \
  --seed ${RANDOMSEED} \
  --max_msa_seqs 5120 \
  --num_subsampled_msa 1024\
  --diffusion_samples 5\
  --subsample_msa \
  ${INPUTFILE}.fasta \
  --output_format pdb \
  --out_dir boltz_${OUTPUTDIR}_5120 &

boltz predict \
  --recycling_steps 1\
  --use_msa_server \
  --model boltz2 \
  --seed ${RANDOMSEED} \
  --max_msa_seqs 512 \
  --num_subsampled_msa 256\
  --diffusion_samples 5\
  --subsample_msa \
  ${INPUTFILE}.fasta \
  --output_format pdb \
  --out_dir boltz_${OUTPUTDIR}_512 &

boltz predict \
  --recycling_steps 1\
  --use_msa_server \
  --model boltz2 \
  --seed ${RANDOMSEED} \
  --max_msa_seqs 64 \
  --num_subsampled_msa 32\
  --diffusion_samples 5\
  --subsample_msa \
  ${INPUTFILE}.fasta \
  --output_format pdb \
  --out_dir boltz_${OUTPUTDIR}_32
deactivate