#!/bin/bash
MOVIENAME=$1

python _srtscraper_.py $MOVIENAME
python merger.py $MOVIENAME
python featureExtractorCont.py $MOVIENAME

