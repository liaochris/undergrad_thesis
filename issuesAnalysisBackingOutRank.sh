#!/bin/bash
#SBATCH --job-name=issuesAnalysis
#SBATCH --output=Rank.out
#SBATCH --error=Rank.err
#SBATCH --account=pi-hortacsu
#SBATCH --time=05:00:00
#SBATCH --partition=bigmem
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5
#SBATCH --mem-per-cpu=12000

module load python/anaconda-2023.09
ipython issuesAnalysisBackingOutRank.py
