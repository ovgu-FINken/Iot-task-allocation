curdir=$(pwd)
cd ns3
./waf configure --build-profile=debug --enable-examples --enable-tests
./waf 
cd $(curdir)

#TODO: different build config support
