### Files

For all files the values were aggregated over 6 models (NN, LGR, RF, DT, SVM, NB) with 40 iterations each. 

- CHANGE_(attr).csv -> change percentage from the base model
- MEAN_(attr).csv -> mean values over all runs
- STD_(attr).csv -> std of values from all runs

### RQ1: WHICH LOSS (initial thoughts)

Rule out based on prediction performance on adult (compas has good performance for all):
- FL the lowest performance on all 4 metrics
- FK, L, VAE, also significantly lowest metrics for adult sex and very high variance
- F 2nd lowest performance and high var on adult race

- FP, LP, P, KL, KP, K all have similar results 

note that all well performing losses seem to use either K or P


BIAS mit performance?

KL, KP, K seem to have consistently good DI, SF, ASPD, for botgh datasets and attributes

PLAN: 
keep going with the KL, KP, K
KL - is advrserial so ideally would rule it out due to time constraints. if K does not reliably perform this will be suggested as the alternative without using Y labels!!!
KP - seems best BUT uses Y. (justify KP by the lower var on adult race aod???? ... sus)
K - seems bit more variance and less debias but will wait for enough evidence