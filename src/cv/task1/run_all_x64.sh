#!/bin/bash

# declaration of all testcases
declare -a testcases=(`ls ./tests/*.json`)

#every loop is one testcase
for tc in "${testcases[@]}"
do
  echo 'Running' ${tc} '...'

  OPENCV_CPU_DISABLE=AVX2,AVX timeout -v 7m ../../build/cv/task1/cvtask1 ${tc}
  echo ${tc}' done.\n'
done
