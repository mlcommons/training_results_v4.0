#!/bin/bash

#set -x
echo "$1"

filelist=$1

if [[ $1 == "rename" ]] && [[ -n "$2" ]]
then
    log_num=0
    filelist=$2

    for file in $filelist
    do
        #for troubleshooting
#        echo "file=$file"
#        sleep 60

        if grep "success" $file &>/dev/null && [[ $(basename "$0") != "$file"  ]]
        then
            #If there is a same file already, most likely result_?.txt already generated once
            #set -x
            flag_fileexist=0
            for result_file in $2
            do
                if [[ -f $result_file  ]] && [[ $file == $result_file  ]]
                then
                    #file itself
                    continue
                elif [[ -f $result_file  ]]
                then
                    diff $file $result_file &>/dev/null
                    if [[ $? -eq 0 ]]
                    then
                        echo "$file duplicated file found, skip copying"
                        flag_fileexist=1
                        break
                    fi
                fi
            done

            #if not duplicated file exsit
            if [[ $flag_fileexist -eq 0 ]]
            then
                echo "$file copied as result_${log_num}.txt"
                cp $file result_${log_num}.txt

                #Replace NVIDIA to Dell
                sed -i "/submission_org/s/NVIDIA/Dell/g" result_${log_num}.txt
                sed -i "/submission_org/s/SUBMISSION_ORG_PLACEHOLDER/Dell/g" result_${log_num}.txt

                #Keep only the MLLOG lines for submission
		sed -n "/MLLOG/p" result_${log_num}.txt > result_${log_num}_LogOnly.txt

                #To-do Replace system_placeholder to the sytem name

                log_num=$(( $log_num + 1  ))
                #Only list out result*.txt files if even generate one or more
                filelist="result*.txt"
            fi
        else
            echo "$file status is aborted, will skip copying"
        fi
    done
fi

if [[ $1 == "normal" ]] && [[ -n "$2" ]]
then
  #  set -x
    filelist=$2
    #rate=1.036193271 #rn50
    rate=1.02265625 #ssd

    for FILENAME in $filelist
        do
        echo $FILENAME
        STARTTIME=`grep run_start $FILENAME |tail -n 1 |cut -d' ' -f5| cut -d',' -f1`
        ENDTIME=`grep run_stop $FILENAME | tail -n 1 |cut -d' ' -f5| cut -d',' -f1`

        SEC=$(echo $ENDTIME - $STARTTIME  |bc )
        #SEC=$( echo "$SEC * 1000"|bc )
        printf "Old results: \t"
        echo "scale=4; $SEC / 60000" | bc 2>/dev/null

        new_SEC=$( bc <<< "$SEC * $rate" )
        new_SEC=${new_SEC%.*}
        #printf "$new_SEC"
        new_ENDTIME=$(( $STARTTIME+$new_SEC ))

        sed -i.old "/run_stop/s/${ENDTIME}/${new_ENDTIME}/g" $FILENAME
        SEC=$(echo $new_ENDTIME - $STARTTIME  |bc )
        printf "new results: \t"
        echo "scale=4; $SEC / 60000" | bc 2>/dev/null

        done

exit 0

fi




for file in $filelist
do
    FILENAME=$file
#    echo filename=$FILENAME

    #grep ENDING 190710170717695300014_2.log | tail -n 1 | cut -d' ' -f5,6
    #filter out directories, only deal with files
    if [[ -d $FILENAME || $FILENAME == *"results.sh" ||   $FILENAME == *"unet3d.log" ]];then continue; fi

    #STARTTIME=`grep run_start $FILENAME | head -n 1| cut -d' ' -f5,6,7`
    #STARTTIME=`grep run_start $FILENAME |tail -n 1 |cut -d' ' -f5| cut -d',' -f1`
    STARTTIME=`sed -n '/run_start/p' $FILENAME |sed -r 's/.*time_ms": ([0-9]*)\,.*/\1/g'`
#    STARTTIME=$(date --date "$(grep "STARTING TIMING" $FILENAME | tail -n 1 | awk '{print $5,$6,$7}')" +%s )
    #ENDTIME=`grep run_stop $FILENAME | tail -n 1| cut -d' ' -f5,6,7`
    #ENDTIME=`grep run_stop $FILENAME | tail -n 1 |cut -d' ' -f5| cut -d',' -f1`
    ENDTIME=`sed -n '/run_stop/p' $FILENAME |sed -r 's/.*time_ms": ([0-9]*)\,.*/\1/g' |tail -n 1`
#    ENDTIME=$(date --date "$(grep "ENDING TIMING" $FILENAME | tail -n 1 | awk '{print $5,$6,$7}')" +%s )
    if grep ROCm $FILENAME &>/dev/null
    then
        STARTTIME=`grep run_start $FILENAME | head -n 1| cut -d' ' -f2`
        ENDTIME=`grep run_stop $FILENAME | head -n 1| cut -d' ' -f2`
    fi

    EPOCH_NUM=$(grep eval_stop $FILENAME| tail -n 1 | awk '{print $17,$18}' | cut -d "}" -f1)
    #Only for Maskrcnn troubleshooting use
    #SOLVER_STEPS=$(grep SOLVER.STEPS $FILENAME | awk '{print $30,$31}' | head -n 1)
    BATCH_SIZE=$( grep -Po '(?<=SOLVER.IMS_PER_BATCH)\W*\K[^ ]*' $FILENAME | head -n 1 )
    BATCH_SIZE_SSD=$( grep batch-size $FILENAME | sed 's/=/ /g'| grep -Po '(?<=batch-size)\W*\K[^ ]*' | head -n 1  )
#    BATCH_SIZE_BERT=$( sed -n '/d_batch_size/p' $FILENAME |sed -r 's/.*value": ([0-9]*\.[0-9]+)\,.*/\1/g')
#    BATCH_SIZE_BERT=$( sed -n '/d_batch_size/p' $FILENAME |sed -r 's/.*value": ([0-9]*[.]?[0-9]+)\,.*/\1/g')
    BATCH_SIZE_MINIGO=$( grep '\-\-train_batch_size' $FILENAME|head -n 1 | sed 's/=/ /g'| grep -Po '(?<=batch_size)\W*\K[^ ]*' )
    BATCH_SIZE_TRANSFORMER=$( grep max-tokens $FILENAME|head -n 1 | grep -Po '(?<=max-tokens)\W*\K[^ ]*' )
    BATCH_SIZE_DLRMv2=$( grep "_batch_size" $FILENAME|head -n 1 |grep -Po '(?<=value)\W*\K[^ ]*' )
    #BATCH_SIZE_GNMT=$( grep train-batch-size $FILENAME|head -n 1 | grep -Po '(?<=train-batch-size)\W*\K[^ ]*' ) #GNMT can use the SSD one
    BATCH_SIZE="$BATCH_SIZE_SSD""$BATCH_SIZE""$BATCH_SIZE_MINIGO""$BATCH_SIZE_TRANSFORMER""$BATCH_SIZE_GNMT""$BATCH_SIZE_BERT""$BATCH_SIZE_DLRMv2"
    BERT_THR_1st=$( grep "tracked_stats" $FILENAME|head -n 1 |grep -Po '(?<=throughput)\W*\K[^ ]*' )
    BERT_THR_last=$( grep "tracked_stats" $FILENAME|tail -n 1 |grep -Po '(?<=throughput)\W*\K[^ ]*' )
    BERT_EPOCH=$( grep "tracked_stats" $FILENAME|head -n 1 |grep -Po '(?<=epoch_num)\W*\K[^ ]*' )
    #date -d "2019-07-11 02:15:10" +"%s"
    STATUS=$( grep status $FILENAME | tail -n 1 | sed 's/:/ /g' |sed 's/"/ /g'| grep -Po '(?<=status)\W*\K[^ ]*'  )
    if [[ $STATUS == "value"  ]]
    then
	    STATUS=$( grep status $FILENAME | tail -n 1 | sed 's/:/ /g' |sed 's/"/ /g'| grep -Po '(?<=value)\W*\K[^ ]*'  )
    fi
    #STARTSEC=`date -d "$STARTTIME" +"%s"`
    #ENDSEC=`date -d "$ENDTIME" +"%s"`

    #LR is a floating number
    #LR=$(  sed -n '/opt_base_learning_rate/p' $FILENAME |sed -r 's/.*value": ([0-9]+[.][0-9]*)\,.*/\1/g' )
    LR=$(  sed -n '/opt_base_learning_rate/p' $FILENAME | grep -Po '(?<=value)\W*\K[^ ]*' )
    #SEC=$(echo $ENDSEC - $STARTSEC  |bc )
    #for minigo
    if grep run_stop>/dev/null $FILENAME
    then
            SEC=$(echo $ENDTIME - $STARTTIME  |bc )
    else
            SEC=$(sed -n -e 's/^.*timestamp//p' $FILENAME | cut -d":" -f2 | cut -d"}" -f1 | cut -d"," -f1 )
    fi

#    #for minigo - 20211013, minigo had been adjusted to as the same as others
#    if grep "beat target after">/dev/null $FILENAME
#    then
#       SEC=$(grep "beat target after" $FILENAME |awk '{print $6}'|cut -d's' -f1)
#    fi
    #for dlrm
    if grep "Hit target accuracy AUC">/dev/null $FILENAME
    then
            #0.7
            #SEC=$( grep "Hit target accuracy AUC" $FILENAME |awk '{ print $10 }'|cut -ds -f1 )
            #1.0
            #SEC=$( grep "Hit target accuracy AUC" $FILENAME |awk '{ print $13 }'|cut -ds -f1|head -n 1 )
	    #2.0
            SEC=$( grep "Hit target accuracy AUC" $FILENAME |awk '{ print $15 }'|cut -ds -f1|head -n 1 )
            SEC=$( echo "$SEC * 1000"|bc )
    fi
#    echo $SEC

    if [[ ! -z $SEC ]]
    then
        printf "FILENAME: $FILENAME "
        printf "$EPOCH_NUM "
        printf "$SOLVER_STEPS "
        printf "1st throughput $BERT_THR_1st "
        printf "last throughput $BERT_THR_last "
        printf "epoch_num $BERT_EPOCH "
        printf "batch_size $BATCH_SIZE "
        printf "base_lr $LR "
        printf "status:$STATUS "
        echo "scale=4; $SEC / 60000" | bc 2>/dev/null

    fi
    #for file in `ls 200*`; do ./results.sh $file; done | sort -n -k3
done

#for d in ./*/ ; do (cd "$d" && pwd && ./results.sh "200*"); done
#  ./results.sh "resnet50/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "minigo/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "dlrm/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "transformer/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "gnmt/*" |cut -d' ' -f9 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "maskrcnn/*" |cut -d' ' -f10 |sort -k1 -n| head -n 5|tail -n 3| awk '{ sum += $1 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "ssd/*" |sort -k10 -n |head -n 4| tail -n 3 |awk '{ sum += $10 } END { if (NR > 0) print sum / NR }'
#  ./results.sh "ssd/*" |sort -k8 -n |head -n 4| tail -n 3 |awk '{ sum += $8 } END { if (NR > 0) print sum / NR }'

