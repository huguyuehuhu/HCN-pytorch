import numpy as np
import os

def get_num_frames(file):
    f=open(file,'r')
    nonempty_lines = [line.strip("\n") for line in  f if line != "\n"]
    line_count = len(nonempty_lines)
    f.close()
    return line_count

def read_xyz(file,num_joint=15,max_body=2):
    num_frames=get_num_frames(file)
    
    print(num_frames) 
    data = np.zeros((3, num_frames, num_joint, max_body))
       
    with open(file, 'r') as f:
        for n, line in enumerate(f):
            mat=np.array(line.split(',')[1:],dtype=np.float32).reshape(-1,3)
            
            ### undo normalization see: http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt
            mat[:,0]=1280-(mat[:,0]*2560)
            mat[:,1]=960-(mat[:,1]*1920)
            mat[:,2]=mat[:,2]*(10000/7.8125)

            mat=mat.transpose()
            person_1_joints=mat[:,:num_joint]
            person_2_joints=mat[:,num_joint:]
            data[:, n, :, 0] = person_1_joints
            data[:, n, :, 1] = person_2_joints
    
    return data    



if __name__ == '__main__':
    
    print(read_xyz("../SBU-Kinect-Interaction/s01s02/08/001/skeleton_pos.txt"))
