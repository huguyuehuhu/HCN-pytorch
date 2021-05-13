import data_loader
import matplotlib.pyplot as plt
import argparse
from utils import utils
from utils.utils import str2bool
import os 
from mpl_toolkits import mplot3d
import numpy as np

joints_map={ 'HEAD' :0,"NECK":1,"TORSO":2,"LEFT_SHOULDER":3,"LEFT_ELBOW":4,"LEFT_HAND":5,"RIGHT_SHOULDER":6,"RIGHT_ELBOW":7,"RIGHT_HAND":8,"LEFT_HIP":9,"LEFT_KNEE":10,"LEFT_FOOT":11,"RIGHT_HIP":12,"RIGHT_KNEE":13,"RIGHT_FOOT":14}

members=(('HEAD','NECK'), ('NECK','TORSO'),('NECK','LEFT_SHOULDER'),('NECK','RIGHT_SHOULDER'),('RIGHT_SHOULDER','RIGHT_ELBOW'),('RIGHT_ELBOW','RIGHT_HAND'),('LEFT_SHOULDER','LEFT_ELBOW'),('LEFT_ELBOW','LEFT_HAND'),('TORSO','LEFT_HIP'),('TORSO','RIGHT_HIP'),('RIGHT_HIP','RIGHT_KNEE'),('LEFT_HIP','LEFT_KNEE'),('RIGHT_KNEE','RIGHT_FOOT'),('LEFT_KNEE','LEFT_FOOT'))

actions={0:"APPROACHING",1:"DEPARTING",2:"KICKING",7:"PUNCHING",3:"PUSHING",5:"HUGGING",4:"ShakingHands",6:"Exchanging"}
def plot(dataloader):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plt.interactive(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
       
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        for j in range(data_batch.shape[i]):
            for k in range(data_batch.shape[2]):
                
                ax.clear() 
                ax.set_xlim([0,1])               
                ax.set_ylim([0,1])               
                ax.set_zlim([0,1])            

                ax.text(0, 0, 0, str(actions[labels_batch[j].item()]))
                for p in range(data_batch.shape[-1]):
                    for parts in members:
                        x1=data_batch[j,0,k,joints_map[parts[0]],p]
                        y1=data_batch[j,1,k,joints_map[parts[0]],p]
                        z1=data_batch[j,2,k,joints_map[parts[0]],p]
                        
                        x2=data_batch[j,0,k,joints_map[parts[1]],p]
                        y2=data_batch[j,1,k,joints_map[parts[1]],p]
                        z2=data_batch[j,2,k,joints_map[parts[1]],p]
                        
                        ax.plot(np.array([x1,x2]),np.array([y1,y2]),np.array([z1,z2]))
                              
                plt.draw()
                plt.pause(0.1)
                    


if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='/data0/', help="root directory for all the datasets")
    parser.add_argument('--dataset_name', default='NTU-RGB-D-CV', help="dataset name ") # NTU-RGB-D-CS,NTU-RGB-D-CV
    parser.add_argument('--model_dir', default='./',
                    help="parents directory of model")

    parser.add_argument('--model_name', default='HCN',help="model name")
    parser.add_argument('--mode', default='train', help='train,test,or load_train')
    parser.add_argument('--num', default='01', help='num of trials (type: list)')
    
    # Load the parameters from json file
    args = parser.parse_args()
    experiment_path =  os.path.join(args.model_dir,'experiments',args.dataset_name,args.model_name+args.num)
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)


    json_file = os.path.join(experiment_path,'params.json')
    if not os.path.isfile(json_file):
        with open(json_file,'w') as f:
            print("No json configuration file found at {}".format(json_file))
            f.close()
            print('successfully made file: {}'.format(json_file))

    params = utils.Params(json_file)



    params.dataset_dir = args.dataset_dir
    params.dataset_name = args.dataset_name
    params.model_version = args.model_name
    params.experiment_path = experiment_path
    params.mode = args.mode

    
    test_dl = data_loader.fetch_dataloader('train', params)
    plot(test_dl)
