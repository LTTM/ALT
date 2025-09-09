#!/bin/bash


config_file=$1


function run() {
	${command}
}


path=$(pwd)
config_path="$1"
command="python main.py"

while IFS="=" read -r arg value; do

  if [ "${arg}" != "" ]; then
    if [ "${value}" = "" ]; then
      command="${command} --${arg}"
    else
      declare "${arg}"="${value}"
      command="${command} --${arg} ${value}"
    fi
  fi

done < "$config_path"


echo $command
run

echo "Done."
