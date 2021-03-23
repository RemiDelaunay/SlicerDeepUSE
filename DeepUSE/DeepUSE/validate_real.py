import torch
import argparse
import time
import math
import scipy.io as io
import pandas as pd
import matplotlib.pyplot as plt

from dataset import create_dataloader
from utils import parse_configuration, to_bmode, render_mpl_table
from models import create_model
from utils.visualizer import Visualizer
from utils.losses import SNRe

def validate(config_file, load_epoch):
    """Performs validation of a specified model.
    
    Input params:
    config_file: Either a string with the path to the JSON 
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and 
        model-specific settings.
    """
    print('Reading config file...')
    configuration = parse_configuration(config_file)
    if load_epoch:
        configuration['model_params']['load_checkpoint'] = int(load_epoch)
    print('Initializing dataset...')
    val_dataset = create_dataloader(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))
    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()
    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params'])   # create a visualizer that displays images and plots
    train_batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
    validation_iterations = len(val_dataset)
    epoch = 1
    num_epochs = 1 
    df = pd.DataFrame(columns=['image_id', 'similarity', 'smooth', 'consistency', 'SNRe'])
    for i, data in enumerate(val_dataset):
        image_list, image_id = data
        nb_sequence = 0
        for image in image_list:
            nb_sequence+=1
            model.set_input(image)
            model.test()
            model.backward_val()
            losses = model.get_current_losses()
            visualizer.print_losses(epoch, num_epochs, i, math.floor(validation_iterations / train_batch_size), losses, image_id, name='validation')
            
            snr = [SNRe(-model.strain_list[t][:,:,143:-143,:]) for t in range(0,len(model.strain_list))]
            bmode = [to_bmode(image[0,i,:,:]) for i in range(0,image.size()[1]-1)]
            axial_displacement = [disp[0, 1, :, :] for disp in model.disp_list]
            
            #save each sequence individualy
            data = {'bmode': bmode,
                    'snr':[data.cpu().numpy() for data in snr],
                    'consistency':[data.cpu().numpy() for data in model.loss_consistency_strain],
                    'similarity': [data.cpu().numpy() for data in model.loss_similarity],
                    'displacement':[data.cpu().numpy() for data in axial_displacement],
                    'strain':[data[0,0,:,:].cpu().numpy() for data in model.strain_list],
                    'strain_compensated':[data[0,0,:,:].cpu().numpy() for data in model.strain_compensated_list]}

            io.savemat(configuration["model_params"]["checkpoint_path"]+'result_{}_{}.mat'.format(image_id[0],nb_sequence),data)

            df.loc[len(df)] = [ image_id[0]+'_{}'.format(nb_sequence),
                                model.loss_similarity_mean.cpu().numpy(),
                                model.loss_smooth_mean.cpu().numpy(),
                                model.loss_consistency_strain_mean.cpu().numpy(),
                                torch.mean(torch.stack(snr)).cpu().numpy()
                                ]
                        
            to_visualise = bmode + axial_displacement + [torch.squeeze(strain) for strain in model.strain_compensated_list]          
            visualizer.save_validation_images(to_visualise, model.loss_similarity, image_id[0]+'_{}'.format(nb_sequence), result_type='displacement')

    df = df.set_index('image_id')
    df = df.astype(float)
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df = df.round(4)
    df = df.reset_index()
    render_mpl_table(df)
    plt.savefig(configuration['visualization_params']['log_path']+'metrics_result')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')
    parser.add_argument('--epoch', dest= 'epoch', help='load model at n epoch for validation')
    args = parser.parse_args()
    validate(args.configfile, args.epoch)
