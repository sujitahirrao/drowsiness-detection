#!/bin/bash

OSTYPE=`uname -s`;

# OSTYPE=Darwin
echo 'This is' ${OSTYPE};

case ${OSTYPE} in
	Linux)
		echo "****** Installing on Linux ******";
		sudo apt install python3 python3-pip build-essential cmake libgtk-3-dev libboost-all-dev --yes;
		sudo pip3 install -r requirements.txt;;
	Darwin)
		echo "****** Installing on MacOS ******";
		# /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)";
		# brew update;
		# brew cask install xquartz;
		brew install python3 cmake gtk+3 boost;
		brew install boost-python --with-python3;
		brew install dlib;  
		pip3 install -r requirements_mac.txt;;
	*)
		echo "****** Unsupported OS: ${OSTYPE} ******";;
esac

