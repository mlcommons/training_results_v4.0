#!/bin/bash

topdir=$PWD
# find directories
for d in $(find . -type d | tail -n+2)
do
	cd $d
	echo "Processing $d..."
	# find any files in this dir with hostname.txt
	count=0
	for nodefile in $(find . -maxdepth 1 | egrep '.*[A-Z0-9]{7}\.txt')
	do
		echo "mv $nodefile node_${count}.txt"
		mv $nodefile node_${count}.txt
		count=$((count+1))
	done
	echo "Moved $count files."
	cd $topdir
done
