#!/bin/bash
# Usage: cpu track

python='/usr/bin/python3'
dir="$( cd "$( dirname "$0" )" && pwd )"
parentdir="$(dirname "$dir")"
trainFile="$parentdir/src/python/train.py"

nohup $python $trainFile &

freq='6'
pid=$!
mydate=`date "+%H:%M:%S"`
outFile="$dir/detail.txt"

echo "pid is $pid"
while ps -p $pid > /dev/null; do
    ps -p$pid -opid -opcpu -ocomm -c | grep $pid | sed "s/^/$mydate /" >> $outFile
    sleep $freq
    mydate=`date "+%H:%M:%S"`
done
