#!/bin/bash

for file in ~/FML/project/Test-Trial-1/datasets/dialogues/*
do
   filename="${file##*/}"
   MOVIENAME="${filename%-*}"
   python _srtscraper_.py $MOVIENAME
   python merger.py $MOVIENAME
   python featureExtractorCont.py $MOVIENAME  
done


