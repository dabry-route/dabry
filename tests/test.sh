#!/bin/bash
if [[ ! $DABRYPATH ]]; then
  echo "Please set DABRYPATH variable to Dabry home dir"
  exit
fi
PYTHONPATH=$PYTHONPATH:$DABRYPATH/src
N=$(echo -e "from dabry.problem import IndexedProblem\nprint(len(IndexedProblem.problems))" | python)

while getopts ":s" opt; do
  case $opt in
    s) SAVE=1
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac
done

echo "Problems : $N"
if [[ -n "$(ls -A $DABRYPATH/tests/out 2>/dev/null)" ]]; then
  rm -r $DABRYPATH/tests/out/*
fi
for ((i=0 ; i < $N; i++)); do
  if [[ $SAVE ]]; then
    python "$DABRYPATH"/tests/test_module.py $i -mqo &
  else
    python "$DABRYPATH"/tests/test_module.py $i -mq &
  fi
done
time wait
python "$DABRYPATH"/tests/test_module.py -s
