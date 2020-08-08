#Translates the h5 files to txt or csv files for C to read (Can't find a h5 library for C)

import h5py
import matplotlib.pyplot as plt
import numpy as np

def translate():
    train_data = h5py.File('train_catvnoncat.h5','r')
    test_data = h5py.File('test_catvnoncat.h5','r')


    #TRAIN X
    ''' for i in range(len(train_data['train_set_x'])):
        print(i)
        myFile = open('Plain_Files/Train/x/' + str(i) + '.txt', 'w')
        for a in range(3):
            myFile.write('\n\n')
            for b in range(64):
                myFile.write('\n')
                for c in range(64):
                    myFile.write(str(train_data['train_set_x'][i][c][b][a]) + " ")'''
    
    #TRAIN Y
    '''for i in range(len(train_data['train_set_y'])):
        print(i)
        myFile = open('Plain_Files/Train/y/' + str(i) + '.txt', 'w')
        #print(str(train_data['train_set_y'][i]))
        myFile.write(str(train_data['train_set_y'][i]))'''
    
    #TEST X
    for i in range(len(test_data['test_set_x'])):
        print(i)
        myFile = open('Plain_Files/Test/x/' + str(i) + '.txt', 'w')
        for a in range(3):
            myFile.write('\n\n')
            for b in range(64):
                myFile.write('\n')
                for c in range(64):
                    myFile.write(str(test_data['test_set_x'][i][c][b][a]) + " ")
    #TEST Y
    for i in range(len(test_data['test_set_y'])):
        print(i)
        myFile = open('Plain_Files/Test/y/' + str(i) + '.txt', 'w')
        #print(str(test_data['test_set_y'][i]))
        myFile.write(str(test_data['test_set_y'][i]))

translate()