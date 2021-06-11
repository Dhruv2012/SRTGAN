#!/bin/sh
while true; 
    do 
    dt=$(date '+%d%m%Y_%H%M%S');
    FILE=experiments;
    logger=tblogger;
    value="$FILE$dt".zip""
    echo $value;
    zip  $value -r ./experiments/ ../Image-SuperResolutiontb_logger
    sleep 5h; 
done
    