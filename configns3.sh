curdir=$(pwd)
cd ns3
./waf configure --build-profile=debug #--enable-examples --enable-tests
cd $curdir

#TODO: different build config support
