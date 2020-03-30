import math
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import tifffile
import sys
    
device = torch.device("cuda")

def printNow(string,a="",b="",c="",d="",e="",f=""):
    print(string,a,b,c,d,e,f)
    sys.stdout.flush()

def lossFunctionMAE(dist, labels, tol=1e-10):
    loss=nn.L1Loss(reduction='none')
    prediction=dist
    targets=labels
    maeloss=loss(prediction, targets).mean(dim=1, keepdim=True)
    return maeloss #B1YX

def lossFunctionReg(dist, labels):
    loss=nn.L1Loss(reduction='none')
    prediction=dist
    targets=labels # just to have MAE aspect of the loss
    return loss(prediction, 0*targets).mean(dim=1, keepdim=True)
    
def lossFunctionBCE(prob, masks):
    weight_rebal = torch.ones_like (masks) / 10.0  +  (1.0 - 1.0 / 10.0) * masks
    loss=nn.BCEWithLogitsLoss(reduction='none', weight=weight_rebal)
    prediction=prob[:, 0:1, ...]
    targets=masks[:, 0:1, ...]
    bceloss=loss(prediction, targets)
    return bceloss
    
def lossFunction(dist, prob, labels, masks):
    loss1=lossFunctionBCE(prob, masks)
    loss1=loss1.mean()
    errors=lossFunctionMAE(dist, labels, tol=1e-10)
    loss2=(errors*masks).sum()/masks.sum()
    masksbg=torch.where(masks==0, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device)) # only do regularization for background pixels
    errors3=lossFunctionReg(dist, labels)
    loss3=(errors3*masksbg).sum()/masksbg.sum()
    loss=1*loss1+0.2*(loss2+1e-4*loss3)
    return loss
    
def shuffle(data, dataGT, dataMask, counter=None):
    if counter is None:
        index=np.random.randint(0, data.shape[0])
    else:
        if counter>=data.shape[0]:
            counter=0
            p = np.random.permutation(data.shape[0])
            data=data[p]
            dataGT=dataGT[p]
            dataMask=dataMask[p]
        index=counter
        counter+=1
    
    img=data[index, ...]
    imgClean=dataGT[index, ...]
    imgMask=dataMask[index, ...]
    
    return img, imgClean, imgMask, counter

def trainingPred(my_train_data, my_train_data_GT, my_train_data_Mask,  net, dataCounter, ps, bs, device):

    inputs= torch.zeros(bs, 1, ps, ps)
    labels= torch.zeros(bs, my_train_data_GT.shape[1], ps, ps)
    masks= torch.zeros(bs, 1, ps, ps)
    

    for j in range(bs):
        im, l, m, dataCounter=shuffle(my_train_data,
                                          my_train_data_GT,
                                          my_train_data_Mask,
                                          counter=dataCounter)

        inputs[j, ...]=torch.from_numpy(im.astype(np.float32)).float()
        labels[j, ...]=torch.from_numpy(l.astype(np.float32)).float()
        masks[j, ...]=torch.from_numpy(m.astype(np.float32)).float()
        
    inputs_raw, labels, masks= inputs.to(device), labels.to(device), masks.to(device)
    dist, prob=net(inputs_raw)
    return dist, prob, labels, masks,  dataCounter
    
    

def trainNetwork(net, trainData , valData, 
               trainDataGT , valDataGT , 
               trainDataMask , valDataMask ,
               postfix, directory, device , 
               verbose=False, numOfEpochs= 400, stepsPerEpoch=100,   
               miniBatchSize=4, learningRate=3e-4, 
               supervised=True, valSize=4
               ):
    combined=np.concatenate((trainData, valData))
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.5,  patience=40, verbose = True)
    running_loss = 0.0
    dataCounter=0

    trainHist=[]
    valHist=[]
    switch=False
    for epoch in range(numOfEpochs):
        lossesEpochTrn=[]
        for step in tqdm(range(stepsPerEpoch)):
            
            optimizer.zero_grad()
            dist, prob, labels, masks, dataCounter = trainingPred(trainData, 
                                                                  trainDataGT,
                                                                  trainDataMask, 
                                                                  net, 
                                                                  dataCounter, 
                                                                  256, #TODO
                                                                  miniBatchSize, 
                                                                  device)

            loss=lossFunction(dist, prob, labels, masks)
            loss.backward()
            lossesEpochTrn.append(loss.item())
            optimizer.step()
            
        lossesEpochTrn=np.array(lossesEpochTrn)
        printNow("Epoch "+str(int(epoch))+" finished")
        printNow("Avg. Loss in Epoch : "+str(np.mean(lossesEpochTrn))+"+-(2SEM)"+str(2.0*np.std(lossesEpochTrn)/np.sqrt(lossesEpochTrn.size)))
        trainHist.append(np.mean(lossesEpochTrn))
        torch.save(net, os.path.join(directory,"last_"+postfix+".net"))
        net.train(False)
        
        with torch.no_grad():
            valCounter=0
            lossesEpochVal=[]
            for step in tqdm(range(valSize)):
                dist, prob, labels, masks, valCounter = trainingPred(valData,
                                                              valDataGT,
                                                              valDataMask,
                                                              net,
                                                              valCounter,
                                                              256, #TODO
                                                              miniBatchSize,
                                                              device)
                loss=lossFunction(dist, prob, labels, masks)
                lossesEpochVal.append(loss.item())
            
            
            lossesEpochVal=np.array(lossesEpochVal)
            printNow("Avg. Validation Loss in Epoch : "+str(np.mean(lossesEpochVal))+"+-(2SEM)"+str(2.0*np.std(lossesEpochVal)/np.sqrt(lossesEpochVal.size)))
            if len(valHist)==0 or np.mean(lossesEpochVal) < np.min(np.array(valHist)):
                torch.save(net,os.path.join(directory,"best_"+postfix+".net"))
            valHist.append(np.mean(lossesEpochVal))
            scheduler.step(np.mean(lossesEpochVal))
            np.save(os.path.join(directory,"history"+postfix+".npy"), (np.array( [np.arange(epoch),trainHist,valHist ] ) ) )
        net.train(True)
    printNow('Finished Training')
    return trainHist, valHist

from stardistPytorch.prediction import predict

def optimize_thresholds(X_val, Y_val, net, nms_threshs=[0.3,0.4,0.5], iou_threshs=[0.3,0.5,0.7], predict_kwargs=None, optimize_kwargs=None, save_to_json=True):
    Yhat_val = [predict(x, net) for x in X_val]
    opt_prob_thresh, opt_measure, opt_nms_thresh = None, -np.inf, None
    for _opt_nms_thresh in nms_threshs:
        _opt_prob_thresh, _opt_measure = optimize_threshold(Y_val, Yhat_val, model=net, nms_thresh=_opt_nms_thresh, iou_threshs=iou_threshs, **optimize_kwargs)
        if _opt_measure > opt_measure:
            opt_prob_thresh, opt_measure, opt_nms_thresh = _opt_prob_thresh, _opt_measure, _opt_nms_thresh
    opt_threshs = dict(prob=opt_prob_thresh, nms=opt_nms_thresh)
    thresholds = opt_threshs
    print(end='', file=sys.stderr, flush=True)
    print("Using optimized values: prob_thresh={prob:g}, nms_thresh={nms:g}.".format(prob=thresholds.prob, nms=thresholds.nms))
    return opt_threshs

def optimize_threshold(Y, Yhat, model, nms_thresh, measure='accuracy', iou_threshs=[0.3,0.5,0.7], bracket=None, tol=1e-2, maxiter=20, verbose=1):
    np.isscalar(nms_thresh) or _raise(ValueError("nms_thresh must be a scalar"))
    iou_threshs = [iou_threshs] if np.isscalar(iou_threshs) else iou_threshs
    values = dict()

    if bracket is None:
        max_prob = max([np.max(prob) for prob, dist in Yhat])
        bracket = max_prob/2, max_prob
    with tqdm(total=maxiter, disable=(verbose!=1), desc="NMS threshold = %g" % nms_thresh) as progress:

        def fn(thr):
            prob_thresh = np.clip(thr, *bracket)
            value = values.get(prob_thresh)
            if value is None:
                Y_instances = [model._instances_from_prediction(y.shape, *prob_dist, prob_thresh=prob_thresh, nms_thresh=nms_thresh)[0] for y,prob_dist in zip(Y,Yhat)]
                stats = matching_dataset(Y, Y_instances, thresh=iou_threshs, show_progress=False, parallel=True)
                values[prob_thresh] = value = np.mean([s._asdict()[measure] for s in stats])
            if verbose > 1:
                print("{now}   thresh: {prob_thresh:f}   {measure}: {value:f}".format(
                    now = datetime.datetime.now().strftime('%H:%M:%S'),
                    prob_thresh = prob_thresh,
                    measure = measure,
                    value = value,
                ), flush=True)
            else:
                progress.update()
                progress.set_postfix_str("{prob_thresh:.3f} -> {value:.3f}".format(prob_thresh=prob_thresh, value=value))
                progress.refresh()
            return -value

        opt = minimize_scalar(fn, method='golden', bracket=bracket, tol=tol, options={'maxiter': maxiter})
    verbose > 1 and print('\n',opt, flush=True)
    return opt.x, -opt.fun



