#git sibmodule setup
git checkout NS-3-integration
git submodule init
git submodule update 

#ns-3 utilities download
python3 download.py



curdir=$(pwd)
cd ns3
./waf clean
./waf configure --build-profile=debug --enable-examples --enable-tests
./waf -v
cd $curdir


#TODO: different build config support

