# Class NetLink
import  sys
import numpy as np
from . import  net_ap
from . import parameters
from . import group
from . import group_type


class NetLink:
    generatedLinkNum = 0
    def __init__(self, length, headAP, tailAP, bandwidth, linkType):
        """ 
        NetLink object has no state and only have attribute.
        Once it is created, the value of its member will keep invariant.  
        """
        if type(headAP) != net_ap.NetAP or type(tailAP) != net_ap.NetAP:
            sys.exit("headAP and tailAP should be NetAP objects.")
        self._linkID = NetLink.generatedLinkNum
        NetLink.generatedLinkNum = NetLink.generatedLinkNum + 1
        self._length = length
        self._headAP = headAP
        self._tailAP = tailAP
        self._bandwidth = bandwidth
        self._type = linkType
    
    def getID(self):
        return self._linkID

    def getLinkLength(self):
        return self._length

    def getHeadAP(self):
        return self._headAP

    def getTailAP(self):
        return self._tailAP

    def getBandwidth(self):
        return self._bandwidth

    def getLinkType(self):
        return self._type
    
    def getKey(self):
        return 'link-'+str(self._linkID)

    def __eq__(self, other):
        if type(other) == NetLink:
            return self._linkID == other.getID()
        else:
            return False

def createALink(headAP, tailAP):
    return NetLink(np.random.randint(5,10), headAP, tailAP, np.random.randint(10,20), parameters.CODE_BACKHAUL_LINK)

if __name__ == "__main__":
    gp = group.createAGroup()

    serverList = gp.getServerList()
    tmpLinkList = []
    # tmpLinkList.append(createALink(serverList[0], serverList[1], 1))
    for i in range(len(serverList)-1):
        tmpLinkList.append(createALink(serverList[i], serverList[i+1]))
    for link in tmpLinkList:
        print("Link in list has key: %s, its type is: %s, bandwidth is: %d, length is: %d." \
            % (link.getKey(), link.getLinkType(), link.getBandwidth(), link.getLinkLength()))