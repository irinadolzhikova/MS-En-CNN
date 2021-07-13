#!/usr/bin/env python
# coding: utf-8
'''Subject-independent evaluation using Multi-Subject Ensemble CNN with DeepConvNet as base classifier
'''


''' The implementation of the base classifiers in this work is based on [1], where the DeepConvNet architecture [2] is used.
References
----------
.. 
[1] K.  Zhang,  N.  Robinson,  S.-W.  Lee,  and  C.  Guan, 
 “Adaptive  transferlearning  for  eeg  motor  imagery  classification  with  deep  convolutionalneural network,”
 Neural Networks, vol. 136, pp. 1–10, 2021.
 
[2] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
   Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
   Deep learning with convolutional neural networks for EEG decoding and
   visualization.
   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
'''

import argparse
import json
import logging
import sys
from os.path import join as pjoin

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from braindecode.models.deep4 import Deep4Net



from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds


datapath = 'Directory for the data'# Set the path to the data

modelpath = 'Directory for the models, models are saved as S_{test_subject}_cv{K in K-fold CV}.pt'  ## set the model path
dfile = h5py.File(datapath, 'r')
torch.cuda.set_device(args.gpu)
set_random_seeds(seed=20200205, cuda=True)
BATCH_SIZE = 16





# Get data from single subject.
def get_data(subj):
    dpath = '/s' + str(subj)
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]
    return X[:], Y[:]


X, Y = get_data(0)
n_classes = 2
in_chans = X.shape[1]
# final_conv_length = auto ensures we only get a single output in the time dimension
model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                 input_time_length=X.shape[2],
                 final_conv_length='auto').cuda()

# Dummy train data to set up the model.
X_train = np.zeros(X[:2].shape).astype(np.float32)
Y_train = np.zeros(Y[:2].shape).astype(np.int64)


def reset_model(checkpoint):
    # Load the state dict of the model.
    model.network.load_state_dict(checkpoint['model_state_dict'])

    # Only optimize parameters that requires gradient.
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.network.parameters()),
                      lr=1*0.01, weight_decay=0.5*0.001)
    model.compile(loss=F.nll_loss, optimizer=optimizer,
                  iterator_seed=20200205, )

modelpath=''## set the model path

subj=subjs[3]
accuracies=[]
subjects_list=[]
majority_results=[]

for test_subject in range(0,54):
   
    fold_predictions=[]
    subjects_list.append(test_subject)
        
    for cv in range(0,13):  # cv in range (0, K), where K is based on K-fold CV

        checkpoint = torch.load(pjoin(modelpath, 'S' + str(test_subject) +'_cv'+str(cv)+ '.pt'),
                                map_location='cuda:' + str(args.gpu))
  
        reset_model(checkpoint)
        model.fit(X_train, Y_train, 0, BATCH_SIZE)
        
        X, Y = get_data(test_subject)
        X_test, Y_test = X[200:400], Y[200:400]
        print(X_test.shape)
        test_loss = model.evaluate(X_test, Y_test)

    
        ss=model.predict_classes(X_test)
        fold_predictions.append(ss)
        print(ss)
    
        corrects = np.sum(ss==Y_test)
        print(corrects)
        test_acc = corrects/ X_test.shape[0]
        print(test_acc)
    
        accuracies.append( test_acc)
        
        
    fold_predictions=torch.where(torch.tensor(fold_predictions) == 0, torch.tensor(-1), torch.tensor(fold_predictions))
    print(fold_predictions)
    majority=sum(fold_predictions)
    pred_output_maj = torch.where(majority< 0, torch.tensor(0), majority)
    pred_output_maj = torch.where(pred_output_maj > 0, torch.tensor(1), pred_output_maj)
    corrects = torch.sum(pred_output_maj == torch.tensor(Y_test))
    test_acc_maj = corrects.cpu().numpy() / X_test.shape[0]
    majority_results.append(test_acc_maj)




list1=subjects_list
list2=majority_results

zipped_lists = zip(list1, list2)
sorted_pairs = sorted(zipped_lists)

tuples = zip(*sorted_pairs)
list1, list2 = [ list(tuple) for tuple in  tuples]
import pandas as pd
method2 = {'#Subjects as a test':list1,
'Test_acc':list2
        
        }

df = pd.DataFrame(method2, columns = ['#Subjects as a test',
'Test_acc'
])

print (df) 


print(df.to_latex(index=False))  
print(model.network)
