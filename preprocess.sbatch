#!/bin/bash
#SBATCH --job-name fashion         # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short          # Slurm partition to use
#SBATCH --ntasks 10                # Number of tasks to run. By default, one CPU core will be allocated per task
#SBATCH --time 0-03:00             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=2000         # Memory limit for each tasks (in MB)
#SBATCH -o out/fashion_%j.out      # File to which STDOUT will be written
#SBATCH -e out/fashion_%j.err      # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ronakdm@uw.edu # Email to which notifications will be sent
 
module load Python
source ../opt/bin/activate
python featurize.py
python fit_knn.py 1 1