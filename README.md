# Dimensionality-Reduction-Techniques

Implementation and comparison of various dimensionality reduction techniques on FLIM data collected from HeLa cells treated with Doxorubicin at 4 time points (control, 20 min, 40 min, 60 min). 

## TSNE 
Given TSNE is compute-intensive, RAPIDS' implmentation of TSNE was used to take advantage of GPUs, which significantly accelerated the process. The following commands were used to run the program (will vary based on machine and configuration):
```
module load apptainer/1.2.2 rapidsai/23.10
apptainer run --nv $CONTAINERDIR/rapidsai-23.10.sif flim-tsne.py
```

