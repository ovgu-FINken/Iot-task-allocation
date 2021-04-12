curdir=$(pwd)
cd ns3
CXXFLAGS="-std=c++1z" ./waf configure --build-profile=debug --enable-examples --enable-tests
./waf clean
CXXFLAGS="-std=c++1z" ./waf -v
cd $curdir

#TODO: different build config support
