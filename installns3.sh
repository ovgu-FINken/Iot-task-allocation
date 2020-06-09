git checkout NS-3-integration
git submodule init
git submodule update 

curdir=$(pwd)
cd ns3
./waf clean
./waf configure --build-profile=debug --enable-examples --enable-tests
./waf 
cd $curdir


#TODO: different build config support

