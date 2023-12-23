#!/bin/bash
module unload python
module load python/anaconda-2022.05  
screen
python match_committers_pr.py
python match_committers_push.py