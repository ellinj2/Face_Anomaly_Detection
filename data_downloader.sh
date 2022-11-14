#!/bin/bash

# This script automates the downloading and structuring of 
# WiderFace and FFHQ datasets

# DATA FLAGS
# 0 for ignore, 1 for download
WIDERFACE_TRAIN=1
WIDERFACE_TEST=0
FFHQ=0

# FUNCTIONS
# Add Folder
function add_folder() {
	if [[ ! -d "$1" ]]; then
		mkdir "$1"
		echo "created $1 folder"
	fi
}

# Download wider_face
function download {
	wget "$1" -O "$2"
	if [[ "y" == "$3" ]]; then
		unzip "$2"
	fi
}

# MAIN FUNCTION

# Force file directory
FILE_LOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd $FILE_LOC

# Check/Create Data Folder
add_folder "data"
cd data

# BEGIN WIDER_FACE
# Check/Create WIDER_FACE
add_folder "wider_face"
cd wider_face

# Check/Create train and test
add_folder "train"
add_folder "test"

if [[ $WIDERFACE_TRAIN -eq 1 ]]; then
	cd train
	if [[ ! -d "0--Parade" ]]; then
		download "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip" "widerface.zip" "y"
		mv WIDER_train/images/* ./
		rm -r WIDER_train/
		rm widerface.zip
	fi
	cd ..
fi

if [[ $WIDERFACE_TEST -eq 1 ]]; then
	cd test
	if [[ ! -d "0--Parade" ]]; then
		download "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip" "widerface.zip" "y"
		mv WIDER_test/images/* ./
		rm -r WIDER_test/
		rm widerface.zip
	fi
	cd ..
fi

if [[ $WIDERFACE_TRAIN -eq 1 || $WIDERFACE_TEST -eq 1 ]]; then
	if [[ ! -e "wider_face_train_bbx_gt.txt" ]]; then
		download "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip" "wider_face_split.zip" "y"
		mv wider_face_split/* .
		rm -r wider_face_split
		rm wider_face_split.zip
	fi
fi

cd ..
# END WIDER_FACE

# BEGIN FFHQ
# END FFHQ
