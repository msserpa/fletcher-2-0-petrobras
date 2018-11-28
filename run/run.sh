#!/bin/bash

#time ../ModelagemFletcher.exe TTI 248 248 248 16 12.5 12.5 12.5 0.001 0.1 | tee log.txt
# time nvprof ../ModelagemFletcher.exe TTI 472 472 472 16 12.5 12.5 12.5 0.001 0.1 &> log.txt
time nvprof ../ModelagemFletcher.exe TTI 216 216 216 16 12.5 12.5 12.5 0.001 0.1 &> log.txt
# time nvprof ../ModelagemFletcher.exe TTI 88 88 88 16 12.5 12.5 12.5 0.001 0.1 &> log.txt
./compare.sh TTI.rsf /scr1/msserpa/TTI.rsf
