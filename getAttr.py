
import ns.core
import ns.network
import ns.netanim
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
    cmd.module = "ns"
    cmd.contains = ""
    cmd.search = ""
    cmd.AddValue("module", "Module for dir printing")
    cmd.AddValue("contains", "Restrict results to entreis containing this")
    cmd.AddValue("search", "search for this")
    cmd.Parse(sys.argv)

    if cmd.search == "":
        module = cmd.module
        
        options = {
                "ns" : ns,
                "core" : ns.core,
                "network" : ns.network,
                "internet" : ns.internet,
                "applications" : ns.applications,
                'lrwpan' : ns.lr_wpan}
        print(f"Attributes for {module}:")
        if cmd.contains== "":
            print(dir(options[module]))
        else:
            print(f"Restriction: {cmd.contains}")
            res = dir(options[module])
            res = [x for x in res if str(cmd.contains) in str(x)]
            print(res)
    else:
        pass
