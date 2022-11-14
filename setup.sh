#! /bin/bash

if [[ ! -d ".venv" ]]; then
	python3 -m venv .venv
	echo "created .venv"
else
	echo ".venv already exists"
fi

source .venv/bin/activate

pip3 install -r requirements.txt

if [[ ! -d "data" ]]; then
	mkdir data
	echo "created data folder"
fi
