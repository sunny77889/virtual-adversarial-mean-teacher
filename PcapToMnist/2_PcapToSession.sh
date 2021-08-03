#!/bin/bash
DIR=$(readlink -f "$(dirname "$0")")  
for file in $DIR/1_Pcap/*
do
    if test -f $file
    then
    	arr=$file
    	name=$(basename $arr .pcap)
    	out="2_Session/AllLayers/"${name}"-ALL"
    	mono 0_Tool/SplitCap_2-1/SplitCap.exe -p 500 -b 50000 -r $arr -o $out
    	rdfind -deleteduplicates true -ignoreempty false $out
    fi
done