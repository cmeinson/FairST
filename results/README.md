## OG FILES:
output of the main...py files

#### MAIN_all_2_race.csv
#### MAIN_all_2_sex.csv
- all single attr runs on Adult and Compas with EN, NN, LGR, RF, DT, SVM, NB
- 4x10 run for each config

#### MAIN_all_race_sex.csv
- 4x10 runs on Compas for multiattr with EN, NN, LGR, RF, DT, SVM, NB
- has incomplete runs for Adult but those mostly errored bc of the EN

#### MAIN_all_Adult_race_sex.csv
- 4x10 runs on Adult for multiattr with NN, LGR, RF, DT, SVM, NB


## Processed FILES:
Files with:
- EN removed
- no summation rows (rows that avg results over last 4 iterations)
- TPR, FPR, TNR, FNR, MCC added
- all multiattr runs merged into one
- VAR columns removed
- 40 runs of each config

#### RESULTS_race.csv
#### RESULTS_sex.csv
#### RESULTS_race_sex.csv

