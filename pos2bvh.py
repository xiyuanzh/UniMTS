import numpy as np
from Quaternions import Quaternions
from scipy_motion import myBVH
import BVH
from scipy_motion import myAnimation
import Animation
from scipy_motion import myInverseKinematics as myIK
import InverseKinematics as IK
from tqdm import tqdm
import multiprocessing
import os
import os.path as osp
from scipy.spatial.transform import Rotation as R


parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
# names = ['root','leftleg1','leftleg2','leftleg3','leftleg4','rightleg1','rightleg2','rightleg3','rightleg4',\
#         'spline1','spline2','spline3','spline4','spline5','rightarm1','rightarm2','rightarm3','rightarm4',\
#         'leftarm1','lertarm2','leftarm3','leftarm4']

def process_file(f):

    fk_positions = np.load('/path/to/joint/pos/%s.npy' % (f))
    
    frametime = 1 / 20
   
    anim_ik, _, _, save_file = IK.animation_from_positions(fk_positions, parents=parents)

    if save_file:
        BVH.save('bvh/%s.bvh' % f, anim_ik, frametime=frametime)

source_dir = '/path/to/joint/pos'
error_file = ['M005836.npy', 'M000990.npy', '000990.npy', '005836.npy']
npy_files = [file[:-4] for file in os.listdir(source_dir) if file.endswith('.npy') and file not in error_file]

# Process files in parallel
pool = multiprocessing.Pool(processes=8)
for _ in tqdm(pool.imap_unordered(process_file, npy_files), total=len(npy_files)):
    pass
pool.close()
pool.join()