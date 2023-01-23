from torch.utils import data
import sys
import os
import cv2


from datasets import get_dataset
from models import get_model
from loss import *
from scripts.utils import get_pose_err

def test(args, save_pose=False):
    if True:
        sys.path.insert(0, '/home/dk/FDANet/pnpransac')
        from pnpransac import pnpransac
    if args.dataset == '7S':
        dataset = get_dataset('7S')
    if args.dataset == '12S':
        dataset = get_dataset('12S')
    test_dataset = dataset(args.data_path, args.dataset, args.scene, split='test', model=args.model, aug='False')

    testloader = data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)
    intrinsics_color = test_dataset.intrinsics_color
    pose_solver = pnpransac(intrinsics_color[0,0], intrinsics_color[1,1], intrinsics_color[0,2], intrinsics_color[1,2])

    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.dataset)
    model_state = torch.load(args.resume,
                map_location=device)['model_state']
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    x = np.linspace(4, 640 - 4, 80)
    y = np.linspace(4, 480 - 4, 60)
    xx, yy = np.meshgrid(x, y)  # [60 80]
    pcoord = np.concatenate((np.expand_dims(xx, axis=2), np.expand_dims(yy, axis=2)), axis=2)
    rot_err_list = []
    transl_err_list = []

    for _, (img, pose) in enumerate(testloader):
        with torch.no_grad():
            img = img.to(device)
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
            if confidences[i] < 0.2:
                coords_filtered.append(coords[i])
                coords_filtered_2D.append(pcoord[i])

        coords_filtered = np.vstack(coords_filtered)
        coords_filtered_2D = np.vstack(coords_filtered_2D)
        rot, transl = pose_solver.RANSAC_loop(coords_filtered_2D.astype(np.float64), coords_filtered.astype(np.float64), 256)

        pose_gt = pose.data.numpy()[0,:,:]  # [4 4]
        pose_est = np.eye(4)        # [4 4]
        pose_est[0:3,0:3] = cv2.Rodrigues(rot)[0].T             # Rwc
        pose_est[0:3,3] = -np.dot(pose_est[0:3,0:3], transl)    # twc

        transl_err, rot_err = get_pose_err(pose_gt, pose_est)
        rot_err_list.append(rot_err)
        transl_err_list.append(transl_err)
        print('step:{}, Pose error: {}m, {}\u00b0，changdu:{}'.format(_ ,transl_err, rot_err,len(coords_filtered_2D)))

    results = np.array([transl_err_list, rot_err_list]).T   # N 2
    np.savetxt(os.path.join(args.output,
            'pose_err_{}_{}_{}_coord.txt'.format(args.dataset,
            args.scene.replace('/','.'), args.model)), results)
    print('Accuracy: {}%'.format(np.sum((results[:,0] <= 0.050)
                * (results[:,1] <= 5)) * 1. / len(results) * 100))
    print('Median pose error: {}m, {}\u00b0'.format(np.median(results[:,0]),
            np.median(results[:,1])))
    print('Average pose error: {}m, {}\u00b0'.format(np.mean(results[:,0]),
            np.mean(results[:,1])))
    print('stddev: {}m, {}\u00b0'.format(np.std(results[:,0],ddof=1),
            np.std(results[:,1],ddof=1)))








































