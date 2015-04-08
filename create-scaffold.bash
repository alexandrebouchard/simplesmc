#!/bin/bash

SCAF_DIR=/Users/bouchard/Documents/courses/stat520-sp2014-15/exercises/2/
SRC_DIR=/Users/bouchard/w/simplesmc
PRJ=simplesmc
CUR=`pwd`

# backup old
cd $SCAF_DIR
zip -r ${PRJ}-scaffold ${PRJ}-scaffold > /dev/null
mv ${PRJ}-scaffold.zip "backups/`date +%s`.zip"

# back up .git
yes | rm -r git-temp-backup 2> /dev/null
cp -r ${PRJ}-scaffold/.git git-temp-backup

# remove the scaffold
yes | rm -r ${PRJ}-scaffold

# copy back
cp -r $SRC_DIR ${PRJ}-scaffold

# remove non-scaffold .git
yes | rm -r ${PRJ}-scaffold/.git

# move back our .git
mv git-temp-backup ${PRJ}-scaffold/.git

# go over files, removing flagged parts of the code
removeFlaggedCode ${PRJ}-scaffold y

# refresh tutorialj
cd ${PRJ}-scaffold
rm create-scaffold.bash
#gradle tutorialj -x test

cd $CUR