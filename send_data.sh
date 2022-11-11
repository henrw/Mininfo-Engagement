#!/bin/bash
array=($(find data/videos/* -type d))
for dir in "${array[@]}"; do   # The quotes are necessary here
    scp -r $dir henrw@lit1000.eecs.umich.edu:~/Misinfo-Engagement/data/videos
done