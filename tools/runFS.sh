#!/bin/bash
#
# README:
# This is use to run several pre-defined videos (benchmarks) under various optimization techniques
#
# $1: 1(high qual), 0(low qual)
# $2: GPU ID

FSPATH=/root/face_swap/

if [ $# -eq 2 ]; then
    if [ $1 -eq 1 ]; then
        QUAL="high"
    else
        QUAL="low"
    fi

    for idol in 2 3 4
    do
        for imgH in 640 320 160
        do
            python swapVideo.py -i ${FSPATH}/data/images/rotation.MOV -o ${FSPATH}/data/output/fsbench/rotation_kcf_${imgH}_${QUAL}_${idol}.avi --idol ${idol} --gpu $2 --rotate 1 --highQual $1 --imgH ${imgH}
            python swapVideo.py -i ${FSPATH}/data/images/expression.MOV -o ${FSPATH}/data/output/fsbench/expression_kcf_${imgH}_${QUAL}_${idol}.avi --idol ${idol} --gpu $2 --rotate 1 --highQual $1 --imgH ${imgH}
            python swapVideo.py -i ${FSPATH}/data/images/segmentation.MOV -o ${FSPATH}/data/output/fsbench/segmentation_kcf_${imgH}_${QUAL}_${idol}.avi --idol ${idol} --gpu $2 --rotate 1 --highQual $1 --imgH ${imgH}
        done
    done
else
    echo "Wrong number of arguments"
fi

