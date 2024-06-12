source setup.sh

for i in syflow sd-mean #sd-kl  bh rsd
do
    python real_world_experiment.py --method $i --alpha 1 &
done