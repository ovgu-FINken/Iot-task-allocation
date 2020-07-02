
import ns.core
import ns.network
import ns.point_to_point
import ns.applications
import ns.wifi
import ns.lr_wpan
import ns.mobility
import ns.csma
import ns.internet 
import ns.sixlowpan
import ns.internet_apps
import ns.energy
import sys

if __name__ == '__main__':
    cmd = ns.core.CommandLine()
    cmd.module = "core"
    cmd.AddValue("module", "Module for dir printing")

    cmd.Parse(sys.argv)

    module = cmd.module
    
    options = {
            "core" : ns.core,
            "network" : ns.network,
            "internet" : ns.internet}
    print(dir(options[module]))
