#!/bin/bash
source setup.sh
for m in {0..9}
do
    START=$((10*$m))
    END=$((10*($m+1)))
    for (( seed=$START; seed<=$END; seed++ ))
    do
        python real_world_experiment.py --seed $seed &
        echo $seed
    done
    wait
done
wait