#!/bin/bash
PYTHONPATH=$PYTHONPATH:/home/bastien/Documents/work/mermoz/src
for ((i=0 ; i < 23; i++));
do
    if [[ ! $i == 12 ]]; then
        python tests/test_module.py $i -mq &
    fi
done
time wait
