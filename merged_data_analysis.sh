#!/bin/bash
#SBATCH --job-name=getting_committers
#SBATCH --output=getting_committers.out
#SBATCH --error=getting_committers.err
#SBATCH --account=pi-hortacsu
#SBATCH --time=33:30:00
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=30000

module unload python
module load python/anaconda-2022.05  
ipython merged_data_analysis.py