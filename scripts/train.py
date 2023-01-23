from torch.utils import data
import sys
import os
import random
import argparse
from pathlib import Path
import torch
import time
from torch.autograd import Variable
import numpy as np 
import cv2
from datasets import get_dataset
from models import get_model
from loss import *
from scripts.utils import get_pose_err

def train(args):
    print(args)
    print('--------------------------')
    if args.dataset == '7S':
        dataset = get_dataset('7S')
    if args.dataset == '12S':
        dataset = get_dataset('12S')
    dataset_train = dataset(args.data_path, args.dataset, args.scene,
                      model=args.model, aug=args.aug)
    test_dataset = dataset(args.data_path, args.dataset, args.scene, split='test', model=args.model, aug='False')
    
    trainloader = data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last = True, pin_memory=True)
    testloader = data.DataLoader(test_dataset, batch_size=1, num_workers=16, shuffle=False)
    reg_loss = EuclideanLoss_with_Uncertainty()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.dataset).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, eps=1e-8,
                                 betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

    best_acc = 0
    min_median_tr = 10000
    min_median_ro = 10000
    for epoch in range(args.n_epoch + 1):
        model.train()
        scheduler.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("learning rate", lr)

        loss_list = []
        corr_list = []
        for _, (img, coord, mask) in enumerate(trainloader):
            model.train()
            torch.set_grad_enabled(True)

            start = time.time()
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            img = Variable(img).cuda()
            coord = Variable(coord).cuda()
            mask = Variable(mask).cuda()

            coord_pred, uncertainty_map_pred = model(img)
            total_loss, accuracy, loss_reg = reg_loss(coord_pred, coord, mask, uncertainty_map_pred)

            loss_list.append(total_loss.detach().cpu().numpy())
            corr_list.append(accuracy.detach().cpu().numpy())
            total_loss.backward()

            optimizer.step()
            end = time.time()
            print('Epoch:{},step:{}, total_loss:{}, residual_loss:{}, accuracy:{},time:{}'.format
                  (epoch, _, total_loss, total_loss, accuracy, end - start))
        print(f"Epoch{epoch}| Loss: {np.mean(loss_list)}| Acc:{np.mean(corr_list)} ")
        if epoch>=100:
            from pnpransac import pnpransac
            intrinsics_color = test_dataset.intrinsics_color   
            pose_solver = pnpransac.pnpransac(intrinsics_color[0,0], intrinsics_color[1,1], intrinsics_color[0,2], intrinsics_color[1,2])
            model.eval()
            if args.dataset == 'my':
                x = np.linspace(4, 640-4, 80)
                y = np.linspace(4, 480-4, 60)
            else:
                x = np.linspace(4, 640-4, 80)
                y = np.linspace(4, 480-4, 60)
            xx, yy = np.meshgrid(x, y)      # [60 80]
            pcoord = np.concatenate((np.expand_dims(xx,axis=2), np.expand_dims(yy,axis=2)), axis=2)
            rot_err_list = []
            transl_err_list = []

            for _, (img, pose) in enumerate(testloader):
                img = Variable(img).cuda()

                coord, uncertainty = model(img)
                coord = np.transpose(coord.cpu().data.numpy()[0,:,:,:], (1,2,0))
                uncertainty = np.transpose(uncertainty[0].cpu().data.numpy(), (1,2,0))
                coord = np.concatenate([coord,uncertainty],axis=2)
                coord = np.ascontiguousarray(coord)             
                pcoord = np.ascontiguousarray(pcoord)           
        
                pcoord = pcoord.reshape(-1,2)                 
                coords = coord[:,:,0:3].reshape(-1,3)           
                confidences = coord[:,:,3].flatten().tolist()  
            
                coords_filtered = []        
                coords_filtered_2D = []

                for i in range(len(confidences)):
                    if confidences[i]>0:
                        coords_filtered.append(coords[i])
                        coords_filtered_2D.append(pcoord[i])

                coords_filtered = np.vstack(coords_filtered)                
                coords_filtered_2D = np.vstack(coords_filtered_2D)

                rot, transl = pose_solver.RANSAC_loop(coords_filtered_2D.astype(np.float64),
                                                      coords_filtered.astype(np.float64), 256)
                pose_gt = pose.data.numpy()[0, :, :]
                pose_est = np.eye(4)
                pose_est[0:3, 0:3] = cv2.Rodrigues(rot)[0].T
                pose_est[0:3, 3] = -np.dot(pose_est[0:3, 0:3], transl)

                transl_err, rot_err = get_pose_err(pose_gt, pose_est)
                rot_err_list.append(rot_err)
                transl_err_list.append(transl_err)
                
                print('step:{}, Pose error: {}m, {}\u00b0，changdu:{}'.format(_ ,transl_err, rot_err, len(coords_filtered_2D))) 

            results = np.array([transl_err_list, rot_err_list]).T 
        
            print('Median pose error: {}m, {}\u00b0'.format(np.median(results[:,0]), 
                np.median(results[:,1]))) 
            print('Average pose error: {}m, {}\u00b0'.format(np.mean(results[:,0]), 
                    np.mean(results[:,1])))
            print('stddev: {}m, {}\u00b0'.format(np.std(results[:,0],ddof=1),
                    np.std(results[:,1],ddof=1)))
            print("ACC:{}", np.sum((results[:,0] <= 0.050) * (results[:,1] <= 5)) * 1. / len(results) * 100)

            best_acc = max(best_acc, np.sum((results[:,0] <= 0.050) * (results[:,1] <= 5)) * 1. / len(results) * 100)
            min_median_tr = min(min_median_tr, np.median(results[:,0]))
            min_median_ro = min(min_median_ro, np.median(results[:,1]))
            state = {'model_state': model.state_dict()}
            
            
            if best_acc == (np.sum((results[:,0] <= 0.050) * (results[:,1] <= 5)) * 1. / len(results) * 100) or \
                    min_median_tr == np.median(results[:,0]) or min_median_ro == np.median(results[:,1]) or \
                    np.median(results[:,0]) < 10 or np.median(results[:,1]) < 100.0 :
                root_path = Path(args.savemodel_path)
                save_path = root_path / args.scene
                save_path.mkdir(parents=True, exist_ok=True)
                print(f'save path to {save_path}')
                torch.save(state, save_path / f'epoch{epoch}_acc{np.sum((results[:,0] <= 0.050) * (results[:,1] <= 5)) * 1. / len(results) * 100}_trans{np.median(results[:,0])}_rot{np.median(results[:,1])}.pth')




