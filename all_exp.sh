#!/bin/bash

mkdir -p logs

###############################################
# model 1 - SimpleTM, 4 datasets, 3 runs each #
###############################################


echo "(1/6) model-SimpleTM[1/4] Weather start"
bash ./scripts/long_term_forecast/Weather_script/SimpleTM.sh \
> logs/SimpleTM_Weather_wClusterGNN.txt 2>&1
echo "(1/6) model-SimpleTM [1/4] Weather done"

echo "(1/6) model-SimpleTM [2/4] ECL start"
bash ./scripts/long_term_forecast/ECL_script/SimpleTM.sh \
> logs/SimpleTM_ECL_wClusterGNN.txt 2>&1
echo "(1/6) model-SimpleTM [2/4] ECL done"

echo "(1/6) model-SimpleTM [3/4] Solar start"
bash ./scripts/long_term_forecast/Solar_script/SimpleTM.sh \
> logs/SimpleTM_Solar_wClusterGNN.txt 2>&1
echo "(1/6) model-SimpleTM [3/4] Solar done"

echo "(1/6) model-SimpleTM [4/4] Exchange start"
bash ./scripts/long_term_forecast/Exchange_script/SimpleTM.sh \
> logs/SimpleTM_Exchange_wClusterGNN.txt 2>&1
echo "(1/6) model-SimpleTM [4/4] Exchange done"


#############################################
#model 2 - TimeXer, 4 datasets, 3 runs each #
#############################################


echo "(2/6) model-TimeXer [1/4] Weather start"
bash ./scripts/long_term_forecast/Weather_script/TimeXer.sh \
> logs/TimeXer_Weather_wClusterGNN.txt 2>&1
echo "(2/6) model-TimeXer [1/4] Weather done"

echo "(2/6) model-TimeXer [2/4] ECL start"
bash ./scripts/long_term_forecast/ECL_script/TimeXer.sh \
> logs/TimeXer_ECL_wClusterGNN.txt 2>&1
echo "(2/6) model-TimeXer [2/4] ECL done"

echo "(2/6) model-TimeXer [3/4] Solar start"
bash ./scripts/long_term_forecast/Solar_script/TimeXer.sh \
> logs/TimeXer_Solar_wClusterGNN.txt 2>&1
echo "(2/6) model-TimeXer [3/4] Solar done"

echo "(2/6) model-TimeXer [4/4] Exchange start"
bash ./scripts/long_term_forecast/Exchange_script/TimeXer.sh \
> logs/TimeXer_Exchange_wClusterGNN.txt 2>&1
echo "(2/6) model-TimeXer [4/4] Exchange done"


################################################
# model 3 - TimeMixer, 4 datasets, 3 runs each #
################################################


echo "(3/6) model-TimeMixer [1/4] Weather start"
bash ./scripts/long_term_forecast/Weather_script/TimeMixer.sh \
> logs/TimeMixer_Weather_wClusterGNN.txt 2>&1
echo "(3/6) model-TimeMixer [1/4] Weather done"

echo "(3/6) model-TimeMixer [2/4] ECL start"
bash ./scripts/long_term_forecast/ECL_script/TimeMixer.sh \
> logs/TimeMixer_ECL_wClusterGNN.txt 2>&1
echo "(3/6) model-TimeMixer [2/4] ECL done"

echo "(3/6) model-TimeMixer [3/4] Solar start"
bash ./scripts/long_term_forecast/Solar_script/TimeMixer.sh \
> logs/TimeMixer_Solar_wClusterGNN.txt 2>&1
echo "(3/6) model-TimeMixer [3/4] Solar done"

echo "(3/6) model-TimeMixer [4/4] Exchange start"
bash ./scripts/long_term_forecast/Exchange_script/TimeMixer.sh \
> logs/TimeMixer_Exchange_wClusterGNN.txt 2>&1
echo "(3/6) model-TimeMixer [4/4] Exchange done"



# ###################################################
# # model 4 - iTransformer, 4 datasets, 3 runs each #
# ###################################################


echo "(4/6) model-iTransformer [1/4] Weather start"
bash ./scripts/long_term_forecast/Weather_script/iTransformer.sh \
> logs/iTransformer_Weather_wClusterGNN.txt 2>&1
echo "(4/6) model-iTransformer [1/4] Weather done"

echo "(4/6) model-iTransformer [2/4] ECL start"
bash ./scripts/long_term_forecast/ECL_script/iTransformer.sh \
> logs/iTransformer_ECL_wClusterGNN.txt 2>&1
echo "(4/6) model-iTransformer [2/4] ECL done"

echo "(4/6) model-iTransformer [3/4] Solar start"
bash ./scripts/long_term_forecast/Solar_script/iTransformer.sh \
> logs/iTransformer_Solar_wClusterGNN.txt 2>&1
echo "(4/6) model-iTransformer [3/4] Solar done"

echo "(4/6) model-iTransformer [4/4] Exchange start"
bash ./scripts/long_term_forecast/Exchange_script/iTransformer.sh \
> logs/iTransformer_Exchange_wClusterGNN.txt 2>&1
echo "(4/6) model-iTransformer [4/4] Exchange done"



# ###############################################
# # model 5 - PatchTST, 4 datasets, 3 runs each #
# ###############################################


echo "(5/6) model-PatchTST [1/4] Weather start"
bash ./scripts/long_term_forecast/Weather_script/PatchTST.sh \
> logs/PatchTST_Weather_wClusterGNN.txt 2>&1
echo "(5/6) model-PatchTST [1/4] Weather done"

echo "(5/6) model-PatchTST [2/4] ECL start"
bash ./scripts/long_term_forecast/ECL_script/PatchTST.sh \
> logs/PatchTST_ECL_wClusterGNN.txt 2>&1
echo "(5/6) model-PatchTST [2/4] ECL done"

echo "(5/6) model-PatchTST [3/4] Solar start"
bash ./scripts/long_term_forecast/Solar_script/PatchTST.sh \
> logs/PatchTST_Solar_wClusterGNN.txt 2>&1
echo "(5/6) model-PatchTST [3/4] Solar done"

echo "(5/6) model-PatchTST [4/4] Exchange start"
bash ./scripts/long_term_forecast/Exchange_script/PatchTST.sh \
> logs/PatchTST_Exchange_wClusterGNN.txt 2>&1
echo "(5/6) model-PatchTST [4/4] Exchange done"


# ##################################################
# # model 6 - Crossformer, 4 datasets, 3 runs each #
# ##################################################


echo "(6/6) model-Crossformer [1/4] Weather start"
bash ./scripts/long_term_forecast/Weather_script/Crossformer.sh \
> logs/Crossformer_Weather_wClusterGNN.txt 2>&1
echo "(6/6) model-Crossformer [1/4] Weather done"

echo "(6/6) model-Crossformer [2/4] ECL start"
bash ./scripts/long_term_forecast/ECL_script/Crossformer.sh \
> logs/Crossformer_ECL_wClusterGNN.txt 2>&1
echo "(6/6) model-Crossformer [2/4] ECL done"

echo "(6/6) model-Crossformer [3/4] Solar start"
bash ./scripts/long_term_forecast/Solar_script/Crossformer.sh \
> logs/Crossformer_Solar_wClusterGNN.txt 2>&1
echo "(6/6) model-Crossformer [3/4] Solar done"

echo "(6/6) model-Crossformer [4/4] Exchange start"
bash ./scripts/long_term_forecast/Exchange_script/Crossformer.sh \
> logs/Crossformer_Exchange_wClusterGNN.txt 2>&1
echo "(6/6) model-Crossformer [4/4] Exchange done"

echo "All experiments finished"