#!/bin/bash 


curdir=$(pwd)
cd ns3
if [ "$1" == "optimized" ]
then
        ./waf configure --build-profile=optimized
else
        ./waf configure --build-profile=debug #--enable-examples --enable-tests
fi
cd $curdir

