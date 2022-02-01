#!/bin/bash 


curdir=$(pwd)
cd ns3
if [ "$1" == "optimized" ]
then
	CXXFLAGS_EXTRA="-I/home/ci/dweikert/boost/boost_1_77_0/" ./waf configure --boost-includes=/home/ci/dweikert/boost/boost_1_77_0/ --build-profile=optimized
else
	CXXFLAGS_EXTRA="-I/home/ci/dweikert/boost/boost_1_77_0/" ./waf configure --boost-includes=/home/ci/dweikert/boost/boost_1_77_0/ --build-profile=debug #--enable-examples --enable-tests
fi
cd $curdir

