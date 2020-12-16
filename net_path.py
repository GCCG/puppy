# Class Path
import sys
from . import net_link
from . import net_ap

class NetPath:
    def __init__(self):
        self._headAP = None
        self._tailAP = None
        self._linklist = []
        self._currentIndex = 0

    def nextLink(self):
        if self._currentIndex == (self._currentIndex % len(self._linklist)):
            link = self._linklist[self._currentIndex]
            self._currentIndex = self._currentIndex + 1 
            return link
        else:
            return None

    def getHeadAP(self):
        return self._headAP

    def getTailAP(self):
        return self._tailAP
            
    def reset(self):
        self._currentIndex = 0

    def getLinkLengthList(self):
        lenghtList = []
        for link in self._linklist:
            lenghtList.append(link.getLinkLength())
        
        return lenghtList

    def getLinkNum(self):
        return len(self._linklist)

    def getPathLength(self):
        pathLen = 0
        for link in self._linklist:
            pathLen = pathLen + link.getLinkLength()
        return pathLen
        
      
    def addLink(self, link):
        if type(link) != net_link.NetLink:
            sys.exit("The type of link is %s, not NetLink.")
        if len(self._linklist) == 0:
            self._headAP = link.getHeadAP()
            self._tailAP = link.getTailAP()
            self._linklist.append(link)
        else:
            self._tailAP = link.getTailAP()
            self._linklist.append(link)
    
def _createNetPath():
    ap_1 = net_ap.NetAP('server', 4)
    ap_2 = net_ap.NetAP('switch', 0)
    ap_3 = net_ap.NetAP('server', 3)
    link_1 = net_link.NetLink(5, ap_1, ap_2, 2, 'normal')
    link_2 = net_link.NetLink(3, ap_2, ap_3, 4, 'normal')

    path = NetPath()
    path.addLink(link_1)
    path.addLink(link_2)
    return path

if __name__ == "__main__":
    path = _createNetPath()
    print("Path's length is:%d" % sum(path.getLinkLengthList()))
    for i in range(path.getLinkNum()):
        print("Link in path has length: %d" % path.nextLink().getLinkLength())
    path.reset()
    print("After reset:")
    for i in range(path.getLinkNum()):
        print("Link in path has length: %d" % path.nextLink().getLinkLength())