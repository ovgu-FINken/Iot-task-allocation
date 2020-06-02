python3 ns-3-allinone/download.py -n ns-3.30

curdir=$(pwd)
cd ns-3-allinone/ns-3.30/
./waf clean
./waf configure --build-profile=debug --enable-examples --enable-tests
./waf 
cd $(curdir)


