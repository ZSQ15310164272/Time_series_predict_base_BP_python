# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 23:00:22 2018

@author: ZSQ
"""
import sys
import os
import predictor
#import matplotlib.pyplot as plt
#import numpy as np

#def main():
#    #print 'main function begin.'
##    if len(sys.argv) != 4:
##        #print 'parameter is incorrect!'
##        #print 'Usage: python esc.py ecsDataPath inputFilePath resultFilePath'
##        exit(1)
#    # Read the input files
#    ecsDataPath = 'data_2015_12.txt'
#    inputFilePath = 'input_5flavors_cpu_7days1.txt'
#    resultFilePath = 'output.txt'
#
#    ecs_infor_array = read_lines(ecsDataPath)
#    input_file_array = read_lines(inputFilePath)
#    # implementation the function predictVm
#    predic_result = predictor.predict_vm(ecs_infor_array, input_file_array)
#    # write the result to output file
#    if len(predic_result) != 0:
#        write_result(predic_result, resultFilePath)
#    else:
#        predic_result.append("NA")
#        write_result(predic_result, resultFilePath)
#    #print 'main function end.'


def write_result(array, outpuFilePath):
    with open(outpuFilePath, 'w') as output_file:
        for item in array:
            output_file.write("%s\n" % item)


def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        #print 'file not exist: ' + file_path
        return None
'''
def loaddata(ecs_infor_array):
    datalist = []
    for line in ecs_infor_array:
        linedata = line.split()
        if int(linedata[1][6:]) > 15:
            pass
        else:
            datalist.append(int(linedata[1][6:]))
    #datalist[np.where(datalist!=3)] = 0
    plt.figure()
    plt.plot(datalist)
    plt.show()
    return datalist
'''
        
if __name__ == "__main__":
    
    ecsDataPath = 'TrainData_2015.1.1_2015.2.19.txt'
    inputFilePath = 'input_5flavors_cpu_7days.txt'
    resultFilePath = 'output1.txt'
    
    ecs_infor_array = read_lines(ecsDataPath)
    input_file_array = read_lines(inputFilePath)
    
    ## DEBUG
    # implementation the function predictVM
    predic_result = predictor.predict_vm(ecs_infor_array, input_file_array)
    # write the result to output file
    if len(predic_result) != 0:
        write_result(predic_result, resultFilePath)
    else:
        predic_result.append("NA")
        write_result(predic_result, resultFilePath)
