
curdir=$(pwd)
cd ns3
CXXFLAGS="-std=c++1z" ./waf
cd $curdir

#TODO: different build config support
