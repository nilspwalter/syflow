source setup.sh
for i in sd-mean syflow # rsd bh sd-kl
do
    python target_dist_experiment.py --method $i --alpha 1 &
done