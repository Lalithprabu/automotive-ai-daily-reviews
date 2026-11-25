#!/bin/bash
# two parallel queues, 1 thread each
cd "$(dirname "$0")"
( python3 train.py --run osv --seed 0 > results/stdout/osv_s0.txt 2>&1; python3 train.py --run osv --seed 1 > results/stdout/osv_s1.txt 2>&1; python3 train.py --run osv_ext --seed 0 > results/stdout/osv_ext_s0.txt 2>&1 ) &
( python3 train.py --run base --seed 0 > results/stdout/base_s0.txt 2>&1; python3 train.py --run base --seed 1 > results/stdout/base_s1.txt 2>&1; python3 train.py --run osv_fixed --seed 0 > results/stdout/osv_fixed_s0.txt 2>&1 ) &
wait
