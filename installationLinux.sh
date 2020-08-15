#!/bin/sh
pip3 install virtualenv
virtualenv env
. env/bin/activate
cat requirements.txt | xargs -n 1 pip3 install