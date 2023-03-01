#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/bastien/Documents/work/dabry/src
for ((i=0 ; i < 23; i++));
do
    if [[ ! $i == 12 ]]; then
        python tests/test_module.py $i -mqo &
    fi
done
time wait
python tests/test_module.py -s
