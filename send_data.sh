#!/bin/bash
array=($(find data/videos/* -type d))
password="wumuzhe88917307"
for dir in "${array[@]}"; do   # The quotes are necessary here
    sshpass -p $password scp -r $dir henrw@lit1000.eecs.umich.edu:~/Misinfo-Engagement/data/videos
done