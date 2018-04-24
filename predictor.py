# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:48:26 2018

@author: ZSQ
"""
import math
import time
import copy
import datetime
from BackPropagation import *
#import matplotlib.pyplot as plt
## -----------------------------------------------------------------------------
class Data_Analysis(object):
    def __init__(self, ecs_lines, input_lines, flavor_sumNum):
        self.ecs_lines = ecs_lines
        self.input_lines = input_lines
        self.flavornum = flavor_sumNum

    def loaddatatxt(self):
        lines = self.ecs_lines
        data = []
        firstline = lines[0].split()
        endline = lines[-1].split()
        # Get start and end datetime
        stardatetime = datetime.datetime.strptime(firstline[2], "%Y-%m-%d")
        enddatetime = datetime.datetime.strptime(endline[2], "%Y-%m-%d")
        subdict = {}
            
        for i in range(len(lines)):
            linedata = lines[i].split()
            currentdatetime = datetime.datetime.strptime(linedata[2], "%Y-%m-%d")
            # judge whether or not the same datetime
            if currentdatetime != stardatetime: # not same day
                data.append(subdict)
                subdict = {}
                ## repair vacant datetime in the data
                deltadatetime = currentdatetime - stardatetime
                deltadays = int(deltadatetime.days)
                if deltadays > 1:
                    # using blank dict fill deltadays
                    for i in range(deltadays - 1):
                        data.append({})
            
                stardatetime = currentdatetime


            if linedata[1] in subdict:
                subdict[linedata[1]] += 1
            else:
                subdict[linedata[1]] = 1
        data.append(subdict)  
        return data, stardatetime, enddatetime
    
    
    def loadinput(self):
        lines = self.input_lines
        inputdict = self.input2req(lines)
        return inputdict
    
    @staticmethod
    def input2req(inputlines):
        reqflavorids = []
        reqflavers = []
        for i in range(3, 3 + int(inputlines[2])):
            reqflavorids.append(int(inputlines[i][6:8]))
            line = inputlines[i].split()
            reqflavers.append([int(line[1]), int(line[2])//1024])
            
        starttime = inputlines[3+int(inputlines[2]) + 3].split()
        enddtime = inputlines[3+int(inputlines[2]) + 4].split()
        startdate = datetime.datetime.strptime(starttime[0], "%Y-%m-%d")
        enddate = datetime.datetime.strptime(enddtime[0], "%Y-%m-%d")
        deltadate = enddate - startdate
        # 1--CPU 2--MEM
        if inputlines[3+int(inputlines[2]) + 1] == 'CPU\n':
            optimize_dim = 1
        else:
            optimize_dim = 2
        
        # physical machine
        PM_imform = inputlines[0].split()
        PMware = [int(PM_imform[0]), int(PM_imform[1]), int(PM_imform[2])]
        inputdict = {}
        inputdict['flavorids'] = reqflavorids
        inputdict['deltadate'] = int(deltadate.days)
        inputdict['optimize_dim'] = optimize_dim
        inputdict['PMware'] = PMware
        inputdict['flavers'] = reqflavers
        return inputdict
        
        
    def list2TimeSeries(self, data):
        TimeSeriesData = []
        flavor_ids = ['flavor' + str(i) for i in range(self.flavornum + 1) if i!= 0]
        feature_vector = []
        for eachdaydata in data:
            for i in range(len(flavor_ids)):
                if flavor_ids[i] in eachdaydata:
                    feature_vector.append(eachdaydata[flavor_ids[i]])
                else:
                    feature_vector.append(0)
            TimeSeriesData.append(feature_vector)
            feature_vector = []
        return TimeSeriesData
    
    def data2regulation(self):
        originaldata, stardate, enddate = self.loaddatatxt()
        TimeSeriesData = self.list2TimeSeries(originaldata)
        TimeSeriesData_T = list(map(list, zip(*TimeSeriesData)))
        filt_TimeSeriesData = self.filter_TimeSeries(TimeSeriesData_T)
        #TimeSeriesData_T = list(map(list, zip(*TimeSeriesData)))
        #self.fig_plot(TimeSeriesData_T, filt_TimeSeriesData)
        return filt_TimeSeriesData
    ''''
    @staticmethod
    def fig_plot(TimeSeriesData_T, filt_TimeSeriesData):
        for i in range(len(TimeSeriesData_T)):
            plt.figure(i)
            plt.subplot(2, 1, 1)
            plt.plot(TimeSeriesData_T[i])
            plt.subplot(2, 1, 2)
            plt.plot(filt_TimeSeriesData[i])
            plt.show()
    ''' 
    @staticmethod
    def filter_TimeSeries(TimeSeriesData_T):
        #TimeSeriesData_T = list(map(list, zip(*TimeSeriesData)))
        TimeSeriesData_T_copy = copy.deepcopy(TimeSeriesData_T)
        tmpTimeSeriesData_T = copy.deepcopy(TimeSeriesData_T)
        for i in range(len(TimeSeriesData_T)):
            tmpTimeSeriesData_T[i].sort()
            maxindex = TimeSeriesData_T_copy[i].index(tmpTimeSeriesData_T[i][-1])
            submaxindex = TimeSeriesData_T_copy[i].index(tmpTimeSeriesData_T[i][-2])
            TimeSeriesData_T_copy[i][maxindex] = tmpTimeSeriesData_T[i][-3]
            TimeSeriesData_T_copy[i][submaxindex] = tmpTimeSeriesData_T[i][-3]
        return TimeSeriesData_T_copy
    
    @staticmethod
    def TimeSeriesData2TrainData(TimeSeriesData_T, flavorid = [1, 2, 3, 4], num = 7):
        # TimeSeriesData_T = list(map(list, zip(*TimeSeriesData)))
        trainsets = []
        flavorid = list(map(lambda x: x-1, flavorid))
        for i in flavorid:
            samples = []
            for j in range(num, len(TimeSeriesData_T[i])):
                sample_pairs = []
                sample = TimeSeriesData_T[i][j-num:j]
                lab = TimeSeriesData_T[i][j:j+1]
                sample_pairs.append(sample)
                sample_pairs.append(lab)
                samples.append(sample_pairs)
            trainsets.append(samples)
        return trainsets
    
    @staticmethod
    def TimeSeriesData2TrainData1(TimeSeriesData, flavorid = [1, 2, 3, 4], num = 7):
        TimeSeriesData_T = list(map(list, zip(*TimeSeriesData)))
        trainsets = []
        labels = []
        for i in flavorid:
            samples = []
            labs = []
            for j in range(num, len(TimeSeriesData_T[i]) - 1):
                sample = TimeSeriesData_T[i][j-num:j]
                lab = TimeSeriesData_T[i][j:j+1]
                samples.append(sample)
                labs.append(lab)
            trainsets.append(samples)
            labels.append(labs)

        return trainsets, labels
## ---------------------------------------------------------------------------------
        
def training_sets2plit(training_sets):
    X = []
    Y = []
    for sample in training_sets:
        train = sample[0]
        lab =  sample[1]
        X.append(train)
        Y.append(lab)
    return X, Y


def training_sets2concat(training_sets):
    X, Y = training_sets2plit(training_sets[0])
    for i in range(1, len(training_sets)):
        x, y = training_sets2plit(training_sets[i])
        X = list(map(lambda x: x[0] + x[1], zip(X, x)))
        Y = list(map(lambda x: x[0] + x[1], zip(Y, y)))
    return X, Y


def data2norm(data, reqflavorids):
    reqflavorids = list(map(lambda x: x-1, reqflavorids))
    normparamlist = []
    datacopy = copy.deepcopy(data)
    for i in range(len(datacopy)):
        maxvalue = max(datacopy[i])
        if maxvalue != 0:
            datacopy[i] = list(map(lambda x : x*1.0/maxvalue, data[i]))
        normparamlist.append(maxvalue)
            
    return datacopy, normparamlist   
  
def write2output(predictsum, reqinformdict, PMalloc):
    # predictsum -- Number list of predictions for each corresponding flavor such as [2, 4, 5, 7]  
    # reqinformdict -- input's information
    # PMalloc -- applied pmware information
    results = []
    # first line
    results.append(int(sum(predictsum)))
    # predict flavor information
    for i in range(len(predictsum)):
        results.append('flavor' + str(reqinformdict['flavorids'][i]) + ' ' + str(int(predictsum[i])))
    results.append(' ')
    # require pmnumbers
    results.append(str(len(PMalloc)))
    for i in range(len(PMalloc)):
        list_unique = list(set(PMalloc[i]))
        # flavor require information in each pmware
        linestr = str(i+1) 
        #results.append(str(i + 1) + ' ')
        for flavorid in list_unique:
            linestr = linestr + ' ' + 'flavor' + flavorid + ' ' + str(int(PMalloc[i].count(flavorid))) + ' '
        results.append(linestr)
        linestr = ''
    return results
            
'''            
def pack2alloc(predictsum, reqinformdict):
    reqflavorinform  = {}
    for i in range(len(predictsum)):
        reqflavorinform[str(reqinformdict['flavorids'][i])] = [predictsum[i], reqinformdict['flavers'][i]]        
    # only consider cpu and mem
    PMware = reqinformdict['PMware'][:-1]
    #reqcopy = copy.deepcopy(reqflavorinform)
    danymic_PM = copy.deepcopy(PMware)
    PMalloc = []
    tmpPMlist = []
    # loop place predict all flavor
    for key, value in reqflavorinform.items():
        while True:
            if value[0] != 0:
                # Dynamically update PMware memory and CPU
                danymic_PM[0] -=  value[1][0]
                danymic_PM[1] -=  value[1][1]
                if danymic_PM[0] < 0 or danymic_PM[1] < 0:
                    # Filled pmware put into PMalloc
                    PMalloc.append(tmpPMlist)
                    # apply for new PMware and empty tmpPMlist
                    danymic_PM = copy.deepcopy(PMware)
                    tmpPMlist = []
                else:
                    value[0]-=1
                    tmpPMlist.append(key)
            else:
                break
    PMalloc.append(tmpPMlist)
    return PMalloc
'''
class WLJ:
    def __init__(self, ID, wl_cpu, wl_mem, remaincpu = 0, remainmem = 0):
        self.ID = ID        
        self.wl_cpu = wl_cpu
        self.wl_mem = wl_mem
        self.remaincpu = wl_cpu
        self.remainmem = wl_mem
        self.loadedlist = []
        
class XNJ:
    def __init__(self, ID, cpuSize, memSize, num):
        self.ID = ID
        self.cpuSize = cpuSize
        self.memSize = memSize
        self.num = num
        
        
   
#    XNJsort = []
#    flaver_pairs = copy.deepcopy(reqinformdict['flavers'])
#    countt = len(flaver_pairs)
#    while countt:
#        max_XNJ = flaver_pairs[0]
#        max_index = 0
#        for i in range(len(flaver_pairs)):
#            if len(flaver_pairs[i]) != 0:
#                tmpXNJ = flaver_pairs[i]
#                if reqinformdict['optimize_dim']==1:# CPU
#                    if tmpXNJ[0] > max_XNJ[0] or (tmpXNJ[0] == max_XNJ[0] and tmpXNJ[1] > max_XNJ[1]):
#                        max_XNJ = tmpXNJ
#                        max_index = i
#        flaver_pairs[max_index][0] = 0
#        flaver_pairs[max_index][1] = 0
#        XNJsort.append(max_index)
#        countt -= 1
#   
           
        
def pack2alloc(predictsum, reqinformdict):
    XNJs = []
    for i in range(len(reqinformdict['flavorids'])):
        XNJ_ID = reqinformdict['flavorids'][i]
        XNJ_cpu = reqinformdict['flavers'][i][0]
        XNJ_mem = reqinformdict['flavers'][i][1]
        XNJ_num = predictsum[i]
        XNJs.append(XNJ(XNJ_ID, XNJ_cpu, XNJ_mem, XNJ_num))


    DE_XNJ = []
    if reqinformdict['optimize_dim']==1:# CPU
        DE_XNJ = sorted(list(range(16)), reverse = True)
    else:
        DE_XNJ = [15,14,12,13,11,9,10,8,6,7,5,3,4,2,1]

    XNJsort = []
    for xxx in DE_XNJ:
        for i in range(len(XNJs)):
            if xxx == XNJs[i].ID:
                XNJsort.append(i)
                break
    
    WLJs = []
    WLJ0 = WLJ(1, reqinformdict['PMware'][0], reqinformdict['PMware'][1])
    WLJs.append(WLJ0)

    for i in XNJsort:
        while XNJs[i].num != 0:
            flag = False
            for j in range(len(WLJs)):
                if WLJs[j].remaincpu - XNJs[i].cpuSize >= 0 and  WLJs[j].remainmem - XNJs[i].memSize >= 0:
                   WLJs[j].remaincpu = WLJs[j].remaincpu - XNJs[i].cpuSize 
                   WLJs[j].remainmem = WLJs[j].remainmem - XNJs[i].memSize
                   WLJs[j].loadedlist.append(XNJs[i].ID)
                   flag = True
                   XNJs[i].num -=1
                   break
            if flag ==False:
                newWLJ = WLJ(1, reqinformdict['PMware'][0], reqinformdict['PMware'][1])
                newWLJ.remaincpu = newWLJ.remaincpu - XNJs[i].cpuSize 
                newWLJ.remainmem = newWLJ.remainmem - XNJs[i].memSize
                newWLJ.loadedlist.append(XNJs[i].ID)
                XNJs[i].num -=1
                WLJs.append(newWLJ)
    PMalloc = []
    for i in range(len(WLJs)):
        PMalloc.append(list(map(lambda x: str(x), WLJs[i].loadedlist)))
    return PMalloc

def flitdata2XY(traindata, flavor_indexs):
    X = []
    Y = []
    for i in range(len(flavor_indexs)):
        x, y = training_sets2plit(traindata[i])
        X.append(x)
        Y.append(y)
    return X, Y

def singleclspredict(X, Y, maxvalue, hiddennodes, lr, starttime, deadtime, predictnum = 14):
    networks = NeuralNetwork(len(X[0]), hiddennodes, len(Y[0]), hidden_layer_bias = 0.0, output_layer_bias = 0.0)
    count = 1
    #errors = []
    #lr = 0.5
    while 1:
        if count % 100 == 0 and lr > 0.2:
            lr/=2
        for i in range(len(X)):
            training_inputs = X[i]
            training_outputs = Y[i]
            networks.train(training_inputs, training_outputs, lr)
        count = count + 1
        if time.time() - starttime > deadtime:
            break
    nextdata = Y[-1]
    predictdata = X[-1][len(Y[0]):] + Y[-1]
    predictlabs = []
    for i in range(predictnum):
        nextdata = networks.feed_forward(predictdata)
        predictdata = predictdata[len(Y[0]):] + nextdata
        predictlabs.append(nextdata[0])
    predictlabs = list(map(lambda x: round(x*maxvalue), predictlabs))    
    return sum(predictlabs)

def trainAR(X, Y, normparamlist, costtime, starttime = 0, hiddennodes = 5, lr = 0.5, showerrors = True, predictnum = 14):
    #hiddennodes = 15
    predicts = []
    for i in range(len(X)):
        starttime = time.time()
        predict = singleclspredict(X[i], Y[i], normparamlist[i], hiddennodes, lr, starttime, costtime/len(X), predictnum)
        predicts.append(int(predict))
    return predicts

def predict_vm00(ecs_lines, input_lines):
    starttime = time.time()
    flavor_sumNum = 15  # flavor params
    TSA = Data_Analysis(ecs_lines, input_lines, flavor_sumNum) 
    reqinformdict = TSA.loadinput()
    data = TSA.data2regulation()
    #reqinformdict['flavorids'].sort()
    #req_flavorids = list(range(1, 16))
    #datanorm, normparamlist = data2norm(data, reqinformdict['flavorids']) 
    datanorm, maxvaluelist = data2norm(data, reqinformdict['flavorids'])
    normparamlist = []
    for FlaID in reqinformdict['flavorids']:
        normparamlist.append(maxvaluelist[FlaID-1])

    num = 10
    traindata = TSA.TimeSeriesData2TrainData(datanorm, reqinformdict['flavorids'], num)
    X, Y = flitdata2XY(traindata, reqinformdict['flavorids'])
    costtime = 5
    predict = trainAR(X, Y, normparamlist, costtime, starttime = starttime, hiddennodes = 1, lr = 0.5, showerrors = True, predictnum = reqinformdict['deltadate'])
    predict = list(map(lambda x: x + 30, predict))
    #predict = list(map(int, predict))
    for i in range(len(predict)):
        if predict[i] > normparamlist[i]*reqinformdict['deltadate']:
            predict[i] = normparamlist[i]*reqinformdict['deltadate'] - int(normparamlist[i]/2)
    
    pmwarealloc = pack2alloc(predict, reqinformdict)
    results = write2output(predict, reqinformdict, pmwarealloc)
    return results

def predict_vm(ecs_lines, input_lines):
    # Do your work from here#
    result = []
    if ecs_lines is None:
        #print 'ecs information is none'
        return result
    if input_lines is None:
        #print 'input file information is none'
        return result
    result = predict_vm00(ecs_lines, input_lines)
    return result