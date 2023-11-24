#!/bin/bash
#SBATCH --job-name=aggregating_push_data
#SBATCH --output=aggregating_push_data.out
#SBATCH --error=aggregating_push_data.err
#SBATCH --account=pi-hortacsu
#SBATCH --time=03:30:00
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=30000

module unload python
module load python/anaconda-2022.05  
ipython aggregating_push_data.py