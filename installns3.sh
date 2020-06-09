git checkout NS-3-integration
curdir=$(pwd)
cd ns3
./waf clean
./waf configure --build-profile=debug --enable-examples --enable-tests
./waf 
cd $(curdir)


