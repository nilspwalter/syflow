source setup.sh

pwd
for i in  syflow # rsd  bh
do
    python dimension_scaling_experiment.py --method $i --alpha 1 
done