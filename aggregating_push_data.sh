#!/bin/bash
#SBATCH --job-name=aggregating_push_data
#SBATCH --output=aggregating_push_data.out
#SBATCH --error=aggregating_push_data.err
#SBATCH --account=pi-hortacsu
#SBATCH --time=03:30:00
#SBATCH --partition=caslake
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000

module unload python
module load python/anaconda-2022.05  
ipython aggregating_push_data.py