#!/bin/bash

#time ../ModelagemFletcher.exe TTI 248 248 248 16 12.5 12.5 12.5 0.001 0.1 | tee log.txt
AU=`pwd`
cd ..
./comp.sh
cd $SCRATCH
cp $AU/../ModelagemFletcher.exe .
cp $AU/compare.sh .
cp $AU/compare.exe .
echo "EXECUTE"
time nvprof -f -o trace ./ModelagemFletcher.exe TTI 472 472 472 16 12.5 12.5 12.5 0.001 0.1 &> log.txt
#time nvprof ./ModelagemFletcher.exe TTI 216 216 216 16 12.5 12.5 12.5 0.001 0.1 &> log.txt
# time nvprof ./ModelagemFletcher.exe TTI 88 88 88 16 12.5 12.5 12.5 0.001 0.1 &> log.txt
cat log.txt | grep -i samples
./compare.sh TTI.rsf ok.rsf
cp trace $AU/
cd $AU
