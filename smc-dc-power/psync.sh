#
# Parallel Rsync function 2020 by Pascal Suter @ DALCO AG, Switzerland
# documentation and explanation at http://wiki.psuter.ch/doku.php?id=parallel_rsync
#
# version 1: initial release in 2017
# version 2: May 2020, removed the need to escape filenames by using 
#            null delimiter + xargs to run commands such as mkdir and rsync, 
#            added ability to resume without rescanning (argument $5) and to skip 
#            already synced directories (argument $6)
#

psync() {
	# $1 = source
	# $2 = destination
	# $3 = dirdepth
	# $4 = numjobs
	# $5 = dirlist file (optional) --> will allow to resume without re-scanning the entire directory structure
        # $6 = progress log file (optional) --> will allow to skip previously synced directory when resuming with a dirlist file
	source=$1
	destination=$2
	depth=$3
	threads=$4
	dirlistfile=$5
	progressfile=$6
	
	# gets directory listing form remote or local using ssh and find
	dirlist(){
		#$1 = path, $2 = maxdepth
		path=$1
		echo "$path" | grep -P "^[^@]*@[^:]*:" > /dev/null
		if [ $? -eq 0 ]; then
			remote=`echo "$path" | awk -F : '{print $1}'`
			remotepath=${path:$((${#remote}+1))}
			ssh $remote "find $remotepath/./ -maxdepth $2 -type d | perl -pe 's|^.*?/\./|\1|'"
		else 
			find $1/./ -maxdepth $2 -type d | perl -pe 's|^.*?/\./|\1|'
		fi
	}
	
	# get a sorted list of md5sums of all files in a directory (remote via ssh or local)
	md5list(){
		#$1 = path
		path=$1
		echo "$path" | grep -P "^[^@]*@[^:]*:" > /dev/null
		if [ $? -eq 0 ]; then
			remote=`echo "$path" | awk -F : '{print $1}'`
			remotepath=${path:$((${#remote}+1))}
			ssh $remote "cd $remotepath; find -type f -print0 | xargs -0 -P $threads -n 1 md5sum | sort -k 2"
		else 
			cd $path; find -type f -print0 | xargs -0 -P $threads -n 1 md5sum | sort -k 2
		fi
	}

	# generate a list of directories to sync 
	if [ -z "$dirlistfile" ]; then
		rawfilelist=$(dirlist $source $depth)
	else 
		# dirlist filename was passed check if it exists and load dirlist from there, otherwise create it and save the dirlist to the file
		if [ -f $dirlistfile ]; then 
			rawfilelist=$(<$dirlistfile)
		else 
			rawfilelist=$(dirlist $source $depth | tee $dirlistfile)
		fi 
	fi

	# separate paths less than DIRDEPTH deep from the others, so that only the "leafs" get rsynced recursively, the rest is synced without recursion
	i=$(($depth - 1))
	parentlist=`echo "$rawfilelist" | sed -e '/^\(.*\/\)\{'$i'\}.*$/d'`
	filelist=`echo "$rawfilelist" | sed -e '/^\(.*\/\)\{'$i'\}.*$/!d'` 
	
	# create target directory: 
	path=$destination
	echo "$path" | grep -P "^[^@]*@[^:]*:" > /dev/null
	if [ $? -eq 0 ]; then
		remote=`echo "$path" | awk -F : '{print $1}'`
		remotepath=${path:$((${#remote}+1))}
		echo -n -e "$remotepath\0" | ssh $remote "xargs -0 mkdir -p"
	else 
		echo -n -e "$path\0" | xargs -0 mkdir -p
	fi
	
	#sync parents first
	echo "==========================================================================="
	echo "Sync parents"
	echo "==========================================================================="
	function PRS_syncParents(){
		source=$2
		destination=$3
		progressfile=$4
		if [ -n "$progressfile" ] && grep -q -x -F "$1" $progressfile ; then
			echo "skipping $1 because it was synced before according to $progressfile"
		else
			echo -n -e "$1\0" | xargs -0 -I PPP rsync -aHx --numeric-ids --relative -f '- PPP/*/' $source/./'PPP'/ $destination/ 2>/tmp/debug
			status=$?
			if [ -n "$progressfile" ]; then 
				echo "$1" >> "$progressfile"
			fi
			return $status
		fi
	}
	export -f PRS_syncParents
	echo "$parentlist" | tr \\n \\0 | xargs -0 -P $threads -I PPP /bin/bash -c 'PRS_syncParents "$@"' _ PPP "$source" "$destination" "$progressfile"
	status=$?
	if [ $status -gt 0 ]; then 
		cat /tmp/debug
		rm /tmp/debug
		echo "ERROR ($status): the was an error when syncing the parent directories, check messages and try again"
		return 1
	fi
	#sync leafs recursively
	echo "==========================================================================="
	echo "Sync leafs recursively"
	echo "==========================================================================="
	function PRS_syncLeafs(){
		source=$2
		destination=$3
		progressfile=$4
		if [ -n "$progressfile" ] && grep -q -x -F "$1" $progressfile ; then
			echo "skipping $1 because it was synced before according to $progressfile"
		else
			echo -n -e "$1\0" | xargs -0 -I PPP rsync -aHx --relative --numeric-ids $source/./'PPP' $destination/ 2>/tmp/debug
			status=$?
			if [ -n "$progressfile" ]; then 
				echo "$1" >> "$progressfile"
			fi
			return $status
		fi
	}
	export -f PRS_syncLeafs
	echo "$filelist" | tr \\n \\0 | xargs -0 -P $threads -I PPP /bin/bash -c 'PRS_syncLeafs "$@"' _ PPP "$source" "$destination" "$progressfile"
	status=$?
	if [ $? -gt 0 ]; then 
		cat /tmp/debug
		rm /tmp/debug
		echo "ERROR: there was an error while syncing the leaf directories recursively, check messages and try again"
		return 1
	fi
    #exit # uncomment for debugging what happenes before the final rsync

	#run a single thread rsync across the entire project directory
	#to make sure nothing is left behind. 
	echo "==========================================================================="
	echo "final sync to double check"
	echo "==========================================================================="
	#rsync -aHvx --delete --numeric-ids $source/ $destination/
	rsync -aHx --numeric-ids $source/ $destination/
	if [ $? -gt 0 ]; then 
		echo "ERROR: there was a problem during the final rsync, check message and try again"
		return 1
	fi
    
	exit # comment out if you want to really do the md5 sums, this may take very long! 

	#create an md5 sum of the md5sums of all files of the entire project directory to comapre it to the archive copy
	echo "==========================================================================="
	echo "sanity check"
	echo "==========================================================================="
	diff <( md5list $source ) <( md5list $destination )
	if [ $? -gt 0 ]; then 
		echo "ERROR: the copy seems to be different from the source. check the list of files with different md5sums above. Maybe the files where modified during the copy process?"
		return 1
	fi

	echo "SUCCESS: the entire directory $project has successfully been copied."
}
