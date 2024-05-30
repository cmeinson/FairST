## OG FILES:
output of the main...py files

1. MAIN_all_2_sex.csv 
2. MAIN_all_2_race.csv
3. MAIN_defualt_sex.csv
4. MAIN_all_race_sex.csv
5. MAIN_all_Adult_race_sex.csv

#### MAIN_all_2_sex.csv & MAIN_all_2_race.csv
- all single attr runs on Adult and Compas with EN, NN, LGR, RF, DT, SVM, NB
- 4x10 run for each config

#### MAIN_defualt_sex.csv
- all single attr runs on Default with NN, LGR, RF, DT, SVM, NB
- 4x10 run for each config

#### MAIN_all_race_sex.csv
- 4x10 runs on Compas for multiattr with EN, NN, LGR, RF, DT, SVM, NB
- has incomplete runs for Adult but those mostly errored bc of the EN

#### MAIN_all_Adult_race_sex.csv
- 4x10 runs on Adult for multiattr with NN, LGR, RF, DT, SVM, NB


## Processed FILES:
1. RESULTS_race.csv
2. RESULTS_sex.csv
3. RESULTS_race_sex.csv

Files with:
- EN removed
- no summation rows (rows that avg results over last 4 iterations)
- TP, FP, TN, FN, MCC added
- all multiattr runs merged into one
- VAR columns removed
- 40 runs of each config
- if the initialinvalid representation oof DF is present -> change it to the correct one


## "no vae" files:
experiments with just the "MASK" (just ensamble predictions without any VAE subgroup translation) model vs base model.

Output files: MAIN_only_NO_VAE_[sens attr].csv
Processed files: RESULTS_with_no_vae_[sens attr].csv
