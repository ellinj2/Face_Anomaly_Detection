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


if ! command -v rclone &> /dev/null
then
    sudo -v ; curl https://rclone.org/install.sh | sudo bash
		rclone config create mydrive drive config_is_local=false
fi
