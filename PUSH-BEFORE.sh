#!/bin/bash
if [ -d ../tmp]
then
   echo tmp exists
else
   mkdir ../tmp && echo tmp created
fi

if [-d checkpoints]
then
   echo moving checkpoints ...
   mv checkpoints ../tmp
fi

if [-d out]
then
   echo moving out ...
   mv out ../tmp
fi

if [-d sample]
then
   echo moving sample ...
   mv sample ../tmp
fi

mv *.m4v ../tmp
mv *.avi ../tmp

if [ -d __pycache__]
then
   rm -rf __pycache__
fi

find . -type f -name '*.bak' -delete
find . -type f -name '*.pyc' -delete
