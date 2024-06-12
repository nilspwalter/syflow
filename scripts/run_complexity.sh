source setup.sh

for i in sd-mean syflow #sd-kl rsd  bh
do
    python rule_complexity_experiment.py --method $i --alpha 1 &
done