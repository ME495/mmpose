import yaml
import numpy as np
import cv2
import os



def get_ir_intrinsics(path):
    '''cam0: color
       cam1: ir1
       cam2: ir2
       
    :param path: path to the camchain file
    :return: Ts: [view,4,4] transformation matrices
                Ds: [view,4] distortion coefficients
                Ks: [view,3,3] intrinsic matrices
    '''
    with open(path, 'r') as f:
        camchain = yaml.load(f, Loader=yaml.FullLoader)
    T_1_1 = np.eye(4)
    T_2_1 = np.array(camchain['cam2']['T_cn_cnm1'])

    
    
    Ts = np.stack([T_1_1, T_2_1], axis=0) # color to color, color to ir1, color to ir2
    
    D1 = np.array(camchain['cam1']['distortion_coeffs'])
    D2 = np.array(camchain['cam2']['distortion_coeffs'])
    
    Ds = np.stack([D1, D2], axis=0)
    
    intrinsics1 = np.array(camchain['cam1']['intrinsics'])
    intrinsics2 = np.array(camchain['cam2']['intrinsics'])
    
    intrinsics_list = [intrinsics1, intrinsics2]
    K_list = []
    for intrinsics in intrinsics_list:
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0., cx],
                    [0., fy, cy],
                    [0., 0., 1.]])
        K_list.append(K)
    Ks = np.stack(K_list, axis=0)
    
    return Ts[:, :3, :], Ds, Ks


def triangulate_joints(joint: np.ndarray, intr: np.ndarray, T: np.ndarray, mask: np.ndarray):
    '''
    :param joint: [*,view,N,2]
    :param intr: [*,view*3*3]
    :param T: [*,view,3,4]
    :param mask: [*,view]
    :return: [*,N,3]
    '''
    Pm = intr @ T  # [*,view,3,4]
    # Am=np.concatenate([Pm[mask,None,2,:]*normalized_joints[mask,:,0,None]-Pm[mask,None,0,:],
    #                     Pm[mask,None,2,:]*normalized_joints[mask,:,1,None]-Pm[mask,None,1,:]],axis=0).transpose(1,0,2)
    Au = Pm[..., None, 2, :] * joint[..., 0, None] - Pm[..., None, 0, :]  # [*,view,N,4]
    Av = Pm[..., None, 2, :] * joint[..., 1, None] - Pm[..., None, 1, :]  # [*,view,N,4]
    Au[np.logical_not(mask)] = 0
    Av[np.logical_not(mask)] = 0
    Am = np.concatenate([Au, Av], axis=-3)
    Am = Am.swapaxes(-3, -2)
    _, _, vh = np.linalg.svd(Am, full_matrices=False)
    x3d_h = vh[..., -1, :]
    x3d = x3d_h[..., :3] / x3d_h[..., 3:]

    return x3d