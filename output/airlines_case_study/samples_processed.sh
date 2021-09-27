#!/bin/bash
if [ $# -eq 0 ]
then
    for k in *
    do
        if [[ ! -d "$k" && "$k" != `basename "$0"` ]]
        then
            wc -l "$k"
        fi
    done
else
    for k in $*
    do
         wc -l "$k"
    done
fi

