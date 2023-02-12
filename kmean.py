import random
import numpy as np
import matplotlib.pyplot as plt 
import math
import regex as re
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def FindClosestPoint(pointX, pointsB) -> list:
    firstIteration = True
    for element in pointsB:
        if (type(element) == list):
            element = np.array([element[0],element[1]])
        tempDist = math.dist(element, pointX)
        if(firstIteration or tempDist < smallestDistance):
            smallestDistance = tempDist
            result = element
            firstIteration = False
    return result

    
'''def plotShow(pointListList,bCenterList, plotNumber):
    colors = {  
                1:"blue",
                2:"green",
                3:"pink",
                4:"black",
                5:"red"
                    }
    color = 0
    for element in bCenterList:
        color += 1
        print("ELEMENT",type(element),element)
        if(type(element) == list):
            print("CHANGED")
            element = np.array([element[0],element[1]])
            print(element)
        x,y = element.T
        plt.scatter(x,y, color = colors[color], marker="v")
        if(color == len(colors)):
            color = 0
    color = 0
    print("LISTLIST",type(pointListList))
    for pointList in pointListList:
        color += 1
        print("POINTLIST",pointList, type(pointList))
        if(type(pointList) == list):
            print("CHANGED")
            pointList = np.array([pointList[0],pointList[1]])
        x,y = pointList.T
        plt.scatter(x,y, color = colors[color])
        if(color == len(colors)):
            color = 0
    plt.title("Plot n°" + str(plotNumber))
    plt.show()'''

def plotShow(pointListList,bCenterList, plotNumber):
    print("BCENTERLIST",bCenterList)
    colors = {  
                1:"blue",
                2:"red",
                3:"pink",
                4:"black",
                5:"green"
                    }
    color = 0
    for element in bCenterList:
        color += 1
        if(type(element) == list):
            element = np.array([element[0],element[1]])
        x,y = element.T
        plt.scatter(x,y, color = colors[color], marker="v")
        if(color == len(colors)):
            color = 0
    color = 0
    for pointList in pointListList:
        color += 1
        for element in pointList:
            if(type(element) == list):
                element = np.array([element[0],element[1]])
            x,y = element.T
            plt.scatter(x,y, color = colors[color])
            if(color == len(colors)):
                color = 0
    plt.title("Plot n°" + str(plotNumber))
    plt.show()


def assingCenters(pointList, centerList):
    dictGroup = {}
    for points in pointList:
        for point in points:
            temp = []
            result = FindClosestPoint(point,centerList)
            groupName = str(result)
            temp.append(point[0])
            temp.append(point[1])
            try :
                dictGroup[groupName].append(temp)
            except :
                dictGroup.update({groupName : [temp]})            
    return dictGroup

'''def calculateNewCenter(groups : dict):
    oldKeys = []
    newDict = {}
    for key in groups.keys():
        oldKeys.append(key)
    for group in groups.values():
        centricValue = np.array([0,0])
        for value in group:
            temp = np.array([value[0],value[1]])
            centricValue += temp
        centricValue = centricValue/len(group)
        #print(type(centricValue),centricValue)
        temp = str(centricValue[0]) + "0 ," + str(centricValue[1]) +"0" 
        newDict[temp] = groups[oldKeys[0]]
        oldKeys.pop(0)
    return newDict'''

def calculateNewCenter(groups : dict):
    oldKeys = []
    newDict = {}
    for key in groups.keys():
        oldKeys.append(key)
    for group in groups.values():
        centricValue = np.array([float(0),float(0)])
        for value in group:
            temp = np.array([float(value[0]),float(value[1])])
            centricValue += temp
        centricValue = centricValue/len(group)
        print("Centric value ",type(centricValue),centricValue)
        temp = str(centricValue[0]) + "0 ," + str(centricValue[1]) +"0" 
        newDict[temp] = groups[oldKeys[0]]
        oldKeys.pop(0)
    return newDict

def transformDictInSeveralList(pointList : dict):
    #print("point list", pointList.values())
    centersList = []
    pointListList = []
    for center in pointList.keys():
        centersList.append([float(s) for s in re.findall(r'[\d]*[.][\d]+', center)])
        #print("CenterList", [float(s) for s in re.findall(r'[\d]*[.][\d]+', center)])
    for points in pointList.values():
        list = []
        for point in points:
            #print("POINT TYPE", type(point))
            list.append(np.array([point[0],point[1]]))
        pointListList.append(list)
    #print("TRANSFORM", pointListList)
    return pointListList, centersList




value = [np.array([
    [1, 4],
    [2, 2],
    [2, 7],
    [2, 8],
    [4, 7],
    [5, 5],
    [10, 8],
    [10,3],
    [5,2]
])]
print("VALUES#1",value)


centers = 3
X_train, true_labels = make_blobs(n_samples=100, centers=centers, center_box=(5,10), random_state=20)
X_train = StandardScaler().fit_transform(X_train)
values = [X_train]
print("VALUES#2",values)


point = np.array([
    [X_train[0][0],X_train[0][1]],
    [X_train[50][0],X_train[50][1]]
])  
print("Starting Center=",point)


#a,b = transformDictInSeveralList(calculateNewCenter(assingCenters(values,point)))
#print("\n\n\n\nA =", a)
#print("B =",b)
#plotShow(pointListList=a,bCenterList=b,plotNumber=1)


a,b = transformDictInSeveralList(calculateNewCenter(assingCenters(values,point)))
for i in range(4):
    plotShow(pointListList=a,bCenterList=b,plotNumber=i)
    a,b = transformDictInSeveralList(calculateNewCenter(assingCenters(a,b)))
