###########################################
# Author: Mathis Konarski                 #
# Date: 21/06/2022                        #
###########################################

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch_geometric as tog

def flow_data(data_fn_df, grid_size, min_flow):
    '''
    Create a numpy array representing the flow for each period
    
    Parameters
    ----------
    data_fn_df : pandas.DataFrame with  start_lat_zone, start_lon_zone, end_lat_zone, end_lon_zone, stoptime_period
    grid_size : grid parameters of data_fn_df
    min_flow : if the flow is too low we are not considering it
    
    Returns
    -------
    numpy.array of shape (n periods, grid lat size, grid lon size, grid lat size, grid lon size)
    '''
    start_grid_df = data_fn_df.filter(['start_lat_zone','start_lon_zone'])
    end_grid_df = data_fn_df.filter(['end_lat_zone','end_lon_zone'])
    start_grid_df.columns = ['lat', 'lon']
    end_grid_df.columns = ['lat', 'lon']
    grid = np.array(pd.concat((start_grid_df, end_grid_df)).drop_duplicates())

    full_grid_flow = np.zeros((data_fn_df.stoptime_period.max()+1,grid_size[0],grid_size[1],grid_size[0],grid_size[1]), dtype='uint8')
    for square in tqdm(grid):
        data_square_df = data_fn_df[(data_fn_df.end_lat_zone==square[0]) & (data_fn_df.end_lon_zone==square[1])]
        data_count_df = data_square_df.value_counts(['start_lat_zone', 'start_lon_zone','stoptime_period'], sort=False)
        data_count_df = data_count_df.reset_index().rename(columns={0:'n_trips', 'stoptime_period':'period'})
        for x in data_count_df.iterrows():
            if x[1].n_trips >= min_flow:
                full_grid_flow[x[1].period, square[0]-1, square[1]-1, x[1].start_lat_zone-1, x[1].start_lon_zone-1] = x[1].n_trips
    return full_grid_flow


def flow_graphs(data_fn_df, grid_size):
    '''
    Create a pandas.Series of graph with the flow (start zone -> end zone) for each period
    
    Parameters
    ----------
    data_fn_df : pandas.DataFrame with flow information (start_lat_zone, start_lon_zone, end_lat_zone, end_lon_zone) and stoptime_period
    grid_size : grid parameters of data_fn_df
    
    Return
    ------
    pandas.Series with graph flow associated with the period
    '''
    graph_flow_ser = pd.Series(index = data_fn_df.stoptime_period.unique(), dtype=object)
    for per in tqdm(data_fn_df.stoptime_period.unique()): # Treat the data period by period
        data_per_df = data_fn_df[(data_fn_df.stoptime_period==per)] # Select the data associated with the period
        flow_per_ser = data_per_df.value_counts(['start_lat_zone','start_lon_zone','end_lat_zone','end_lon_zone'])
        start_nodes_lst, end_nodes_lst, edges_weight_lst = ([], [], []) # Graph data
        for x in flow_per_ser.iteritems(): # Each flow data is filled separately
            start_nodes_lst.append((x[0][0]-1)*(grid_size[1])+x[0][1]-1)
            end_nodes_lst.append((x[0][2]-1)*(grid_size[1])+x[0][3]-1)
            edges_weight_lst.append(x[1])
        graph_per = tog.data.Data(x=torch.tensor(np.ones((grid_size[0]*grid_size[1],1)), dtype=torch.float),
                                  edge_index=torch.tensor((start_nodes_lst, end_nodes_lst), dtype=torch.long),
                                  edge_weight=torch.tensor(edges_weight_lst, dtype=torch.float)) # Graph creation using the data
        graph_flow_ser[per] = graph_per.__copy__()
    return graph_flow_ser.sort_index()


def volume_data(data_fn_df):
    '''
    Create a numpy image of the volume demand for each period with 2 channels: start and end
    
    Parameters
    ----------
    data_fn_df : pandas.DataFrame with  start_lat_zone, start_lon_zone, end_lat_zone, end_lon_zone,
                                        starttime_period, stoptime_period, weekday, hour
    
    Returns
    -------
    numpy.array of shape (n periods, grid lat size, grid lon size, 2)
    '''
    volume_start_ser = data_fn_df.value_counts(['start_lat_zone', 'start_lon_zone', 'starttime_period'])
    volume_start_ser.name = 'volume_start'
    volume_start_ser.index.names = ['lat', 'lon', 'period']

    volume_stop_ser = data_fn_df.value_counts(['end_lat_zone', 'end_lon_zone', 'stoptime_period'])
    volume_stop_ser.name = 'volume_stop'
    volume_stop_ser.index.names = ['lat', 'lon', 'period']

    volume_df = pd.DataFrame(volume_start_ser).join(volume_stop_ser, how='outer')
    volume_df = volume_df.fillna(0).reset_index()

    time_period_df = data_fn_df.filter(['starttime_period', 'weekday',
                                        'hour']).drop_duplicates()
    time_period_df = time_period_df.set_index('starttime_period')

    volume_df = volume_df.join(time_period_df, on='period')

    volume_np = np.zeros((volume_df.period.max()+1, volume_df.lat.max(), volume_df.lon.max(), 2))

    for row in tqdm(volume_df.itertuples()):
        volume_np[row.period, row.lat-1, row.lon-1, 0] = row.volume_start
        volume_np[row.period, row.lat-1, row.lon-1, 1] = row.volume_stop
    return volume_np


def scoring(y_arr, pred_arr, intro_str, norm, minv):
    '''
    Print performances for one epoch of train or test loop
    
    Parameters
    ----------
    y_arr : numpy.array with real values
    pred_arr : numpy.array with predicted values
    intro_str : string to explain what is scored, possibility to add informations
    minv : int with the minimal volume to be considered for scoring
    norm : int with the maximal value for y_arr during the training period
    '''
    y_start_arr, pred_start_arr = y_arr[:,:,:,0], pred_arr[:,:,:,0] # Separation of starting and finishing informations
    y_stop_arr, pred_stop_arr = y_arr[:,:,:,1], pred_arr[:,:,:,1]
    y_start_minv = y_start_arr[y_start_arr*norm >= minv] *norm # Denormalization
    pred_start_minv = pred_start_arr[y_start_arr*norm >= minv] *norm
    y_stop_minv = y_stop_arr[y_stop_arr*norm >= minv] *norm 
    pred_stop_minv = pred_stop_arr[y_stop_arr*norm >= minv] *norm 
    print(intro_str,
          '| Start RMSE:', "%.2f" %np.sqrt(((pred_start_minv - y_start_minv)**2).mean()),
          '| Start MAPE:', "%.2f" %((abs(pred_start_minv - y_start_minv) / y_start_minv).mean()*100), '%',
          '| Stop RMSE:', "%.2f" %np.sqrt(((pred_stop_minv - y_stop_minv)**2).mean()),
          '| Stop MAPE:', "%.2f" %((abs(pred_stop_minv - y_stop_minv) / y_stop_minv).mean()*100), '%')


def train_loop(dataloader, model, loss_fn, optimizer, norm, min_vol_metrics):
    '''
    GNN Model training
    
    Parameters
    ----------
    dataloader : pytorch_geometic.loader.DataLoader with training data
    model : torch.nn.Module
    loss_fn : torch loss
    optimizer : torch.optim adapted optimizer for the training
    norm : maximal volume for the volume data
    min_vol_metrics : int with minimal number of trips demand to be considered for scoring
    
    Returns
    -------
    loss : loss average across training
    '''
    size = len(dataloader.dataset)
    y_all_train_arr = np.zeros((size, 10,20,2)) # Used to copmute final metrics
    pred_all_train_arr = np.zeros((size, 10,20,2))
    for batch, (V, F, y) in enumerate(tqdm(dataloader)):
        # Compute prediction and loss
        pred = model(V, F)
        weight = (y*norm>min_vol_metrics)+1e-3  # Focus on the more important data
        loss = (loss_fn(pred, y)*weight).mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save prediction
        y_all_train_arr[batch*dataloader.batch_size:min(size,(batch+1)*dataloader.batch_size)] = y[:,-1]
        pred_all_train_arr[batch*dataloader.batch_size:min(size,(batch+1)*dataloader.batch_size)] = pred[:,-1].detach()
    
    loss = loss.item()
    scoring(y_all_train_arr, pred_all_train_arr, f"Train : Avg loss: {loss:>7f}", norm, min_vol_metrics)
    return loss
            

def test_loop(dataloader, model, loss_fn, norm, min_vol_metrics):
    '''
    Validate the model performances via screen printing
    
    Parameters
    ----------
    dataloader : pytorch_geometic.loader.DataLoader with training data
    model : torch.nn.Module
    loss_fn : torch loss
    norm : maximal volume for the volume data
    '''
    size = len(dataloader.dataset)
    test_loss = 0
    y_all_val_arr = np.zeros((size, 10,20,2)) # Used to copmute final metrics
    pred_all_val_arr = np.zeros((size, 10,20,2))
    with torch.no_grad():
        for i, (V, F, y) in enumerate(dataloader):
            pred = model(V, F)
            weight = (y*norm>min_vol_metrics)+1e-3
            test_loss += (loss_fn(pred, y)*weight).mean()
            y_all_val_arr[i] = y[0,-1]
            pred_all_val_arr[i] = pred[0,-1].detach()

    test_loss /= size
    scoring(y_all_val_arr, pred_all_val_arr, f"Test : Avg loss: {test_loss:>8f}", norm, min_vol_metrics)
    
    
def model_training(loader_tuple, model, loss_fn, optimizer, epochs, min_vol_metrics, patience=5):
    '''
    Train a model and validate it on unseen data
    
    Parameters
    ----------
    loader_tuple : tuple with (pytorch dataloader of training data,
                               pytorch dataloader of testing data,
                               norm used in the dataset)
    model : pytorch model to train and test
    loss_fn : torch loss
    optimizer : torch optimizer
    epochs : maximal number of training loop to realize
    min_vol_metrics : minimal number of trip to be considered inside training functions
    patience : number of epochs without loss reduction before early stopping
    '''
    loss_min, trigger = 1000,0
    for t in range(epochs):
        print(f"Epoch {t+1}:")
        loss = train_loop(loader_tuple[0], model, loss_fn, optimizer, loader_tuple[2], min_vol_metrics)
        test_loop(loader_tuple[1], model, loss_fn, loader_tuple[2], min_vol_metrics)
        print('')
        if loss>loss_min:
            trigger+=1
            if trigger==patience:
                print('Early stopping: loss minimal=', loss_min)
                break
        else:
            loss_min = loss
            trigger = 0
            

def combined_train_loop(dataloaders_tuple, GNN_models_tuple, combined_model, loss_fn, optimizers_tuple, norms_tuple, min_vol_metrics):
    '''
    Combined models training
    Train in parralel GNN models on the different transportation mode then combine the prediction with a final model to take into account the influence of one mode of transport on the others
    
    Parameters
    ----------
    dataloaders_tuple : tuple containing pytorch_geometic.loader.DataLoader with training data
    GNN_models_tuple : tuple containing torch.nn.Module with the model for each mode of transportation
    combined_model : final model combining the different transportation modes
    loss_fn : torch loss
    optimizers_tuple : tuple torch.optim optimizer for each model
    norms_tuple : tuple with the maximal volume for the volume data by transportation type
    min_vol_metrics : int with minimal number of trips demand to be considered for scoring

    Returns
    -------
    loss : loss average across training
    '''
    size = len(dataloaders_tuple[0].dataset)
    batch_size_fn = dataloaders_tuple[0].batch_size
    y_bike_train_arr = np.zeros((size, 10,20,2)) # Used to copmute final metrics
    pred_bike_train_arr = np.zeros((size, 10,20,2))
    y_gtaxi_train_arr = np.zeros((size, 10,20,2)) # Used to copmute final metrics
    pred_gtaxi_train_arr = np.zeros((size, 10,20,2))
    y_ytaxi_train_arr = np.zeros((size, 10,20,2)) # Used to copmute final metrics
    pred_ytaxi_train_arr = np.zeros((size, 10,20,2))
    pred_combined_bike_train_arr = np.zeros((size, 10,20,2))
    pred_combined_gtaxi_train_arr = np.zeros((size, 10,20,2))
    pred_combined_ytaxi_train_arr = np.zeros((size, 10,20,2))
    
    iter_loader_bike = iter(dataloaders_tuple[0])
    iter_loader_gtaxi = iter(dataloaders_tuple[1])
    iter_loader_ytaxi = iter(dataloaders_tuple[2])
    
    bike_model = GNN_models_tuple[0]
    gtaxi_model = GNN_models_tuple[1]
    ytaxi_model = GNN_models_tuple[2]
    
    optimizer_bike = optimizers_tuple[0]
    optimizer_gtaxi = optimizers_tuple[1]
    optimizer_ytaxi = optimizers_tuple[2]
    optimizer_final = optimizers_tuple[3]

    for batch in tqdm(range(len(iter_loader_bike))):
        (V_bike, F_bike, y_bike) = iter_loader_bike.next()
        (V_gtaxi, F_gtaxi, y_gtaxi) = iter_loader_gtaxi.next()
        (V_ytaxi, F_ytaxi, y_ytaxi) = iter_loader_ytaxi.next()
        # Compute prediction and loss
        pred_bike = bike_model(V_bike, F_bike)
        pred_gtaxi = gtaxi_model(V_gtaxi, F_gtaxi)
        pred_ytaxi = ytaxi_model(V_ytaxi, F_ytaxi)
        combined_pred = combined_model(pred_bike.detach(), pred_gtaxi.detach(), pred_ytaxi.detach())
        
        bike_weight = (y_bike*norms_tuple[0]>min_vol_metrics)+1e-3  # Focus on the more important data
        bike_loss = (loss_fn(pred_bike, y_bike)*bike_weight).mean()
        gtaxi_weight = (y_gtaxi*norms_tuple[1]>min_vol_metrics)+1e-3  # Focus on the more important data
        gtaxi_loss = (loss_fn(pred_gtaxi, y_gtaxi)*gtaxi_weight).mean()
        ytaxi_weight = (y_ytaxi*norms_tuple[2]>min_vol_metrics)+1e-3  # Focus on the more important data
        ytaxi_loss = (loss_fn(pred_ytaxi, y_ytaxi)*ytaxi_weight).mean()
        combined_weight = torch.concat((((y_bike*norms_tuple[0]>min_vol_metrics)+1e-3).unsqueeze(dim=0),
                                        ((y_gtaxi*norms_tuple[1]>min_vol_metrics)+1e-3).unsqueeze(dim=0),
                                        ((y_ytaxi*norms_tuple[2]>min_vol_metrics)+1e-3).unsqueeze(dim=0)), axis=0)
        combined_loss = (loss_fn(combined_pred, torch.concat((y_bike.unsqueeze(dim=0), y_gtaxi.unsqueeze(dim=0), y_ytaxi.unsqueeze(dim=0)), axis=0))*combined_weight).mean()

        # Backpropagation
        optimizer_bike.zero_grad()
        optimizer_gtaxi.zero_grad()
        optimizer_ytaxi.zero_grad()
        optimizer_final.zero_grad()
        
        bike_loss.backward()
        gtaxi_loss.backward()
        ytaxi_loss.backward()
        combined_loss.backward()
        
        optimizer_bike.step()
        optimizer_gtaxi.step()
        optimizer_ytaxi.step()
        optimizer_final.step()
        
        # Save prediction
        y_bike_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = y_bike[:,-1]
        pred_bike_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = pred_bike[:,-1].detach()
        y_gtaxi_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = y_gtaxi[:,-1]
        pred_gtaxi_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = pred_gtaxi[:,-1].detach()
        y_ytaxi_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = y_ytaxi[:,-1]
        pred_ytaxi_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = pred_ytaxi[:,-1].detach()
        pred_combined_bike_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = combined_pred[0,:,-1].detach()
        pred_combined_gtaxi_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = combined_pred[1,:,-1].detach()
        pred_combined_ytaxi_train_arr[batch*batch_size_fn:min(size,(batch+1)*batch_size_fn)] = combined_pred[2,:,-1].detach()
    
    loss = combined_loss.item()
    scoring(y_bike_train_arr, pred_bike_train_arr, f"Train bike : Avg loss: {bike_loss.item():>7f}", norms_tuple[0], min_vol_metrics)
    scoring(y_bike_train_arr, pred_combined_bike_train_arr, f"Train bike combination : Avg loss: {loss:>7f}", norms_tuple[0], min_vol_metrics)
    scoring(y_gtaxi_train_arr, pred_gtaxi_train_arr, f"Train gtaxi : Avg loss: {gtaxi_loss.item():>7f}", norms_tuple[1], min_vol_metrics)
    scoring(y_gtaxi_train_arr, pred_combined_gtaxi_train_arr, f"Train gtaxi combination : Avg loss: {loss:>7f}", norms_tuple[1], min_vol_metrics)
    scoring(y_ytaxi_train_arr, pred_ytaxi_train_arr, f"Train ytaxi : Avg loss: {ytaxi_loss.item():>7f}", norms_tuple[2], min_vol_metrics)
    scoring(y_ytaxi_train_arr, pred_combined_ytaxi_train_arr, f"Train ytaxi combination : Avg loss: {loss:>7f}", norms_tuple[2], min_vol_metrics)
    return loss


def combined_test_loop(dataloaders_tuple, GNN_models_tuple, combined_model, loss_fn, norms_tuple, min_vol_metrics):
    '''
    Combined models validation
    
    Parameters
    ----------
    dataloaders_tuple : tuple containing pytorch_geometic.loader.DataLoader with training data
    GNN_models_tuple : tuple containing torch.nn.Module with the model for each mode of transportation
    combined_model : final model combining the different transportation modes
    loss_fn : torch loss
    norms_tuple : tuple with the maximal volume for the volume data by transportation type
    min_vol_metrics : int with minimal number of trips demand to be considered for scoring
    '''
    size = len(dataloaders_tuple[0].dataset)
    bike_loss = 0
    gtaxi_loss = 0
    ytaxi_loss = 0
    combined_loss = 0
    y_bike_all_arr = np.zeros((size, 10,20,2)) # Used to copmute final metrics
    pred_bike_all_arr = np.zeros((size, 10,20,2))
    y_gtaxi_all_arr = np.zeros((size, 10,20,2)) # Used to copmute final metrics
    pred_gtaxi_all_arr = np.zeros((size, 10,20,2))
    y_ytaxi_all_arr = np.zeros((size, 10,20,2)) # Used to copmute final metrics
    pred_ytaxi_all_arr = np.zeros((size, 10,20,2))
    pred_combined_bike_all_arr = np.zeros((size, 10,20,2))
    pred_combined_gtaxi_all_arr = np.zeros((size, 10,20,2))
    pred_combined_ytaxi_all_arr = np.zeros((size, 10,20,2))
    
    iter_loader_bike = iter(dataloaders_tuple[0])
    iter_loader_gtaxi = iter(dataloaders_tuple[1])
    iter_loader_ytaxi = iter(dataloaders_tuple[2])
    
    bike_model = GNN_models_tuple[0]
    gtaxi_model = GNN_models_tuple[1]
    ytaxi_model = GNN_models_tuple[2]
    with torch.no_grad():
        for i in tqdm(range(len(iter_loader_bike))):
            (V_bike, F_bike, y_bike) = iter_loader_bike.next()
            (V_gtaxi, F_gtaxi, y_gtaxi) = iter_loader_gtaxi.next()
            (V_ytaxi, F_ytaxi, y_ytaxi) = iter_loader_ytaxi.next()
            
            pred_bike = bike_model(V_bike, F_bike)
            pred_gtaxi = gtaxi_model(V_gtaxi, F_gtaxi)
            pred_ytaxi = ytaxi_model(V_ytaxi, F_ytaxi)
            combined_pred = combined_model(pred_bike.detach(), pred_gtaxi.detach(), pred_ytaxi.detach())
            
            bike_weight = (y_bike*norms_tuple[0]>min_vol_metrics)+1e-3  # Focus on the more important data
            bike_loss += (loss_fn(pred_bike, y_bike)*bike_weight).mean()
            gtaxi_weight = (y_gtaxi*norms_tuple[1]>min_vol_metrics)+1e-3  # Focus on the more important data
            gtaxi_loss += (loss_fn(pred_gtaxi, y_gtaxi)*gtaxi_weight).mean()
            ytaxi_weight = (y_ytaxi*norms_tuple[2]>min_vol_metrics)+1e-3  # Focus on the more important data
            ytaxi_loss += (loss_fn(pred_ytaxi, y_ytaxi)*ytaxi_weight).mean()
            combined_weight = torch.concat((((y_bike*norms_tuple[0]>min_vol_metrics)+1e-3).unsqueeze(dim=0),
                                            ((y_gtaxi*norms_tuple[1]>min_vol_metrics)+1e-3).unsqueeze(dim=0),
                                            ((y_ytaxi*norms_tuple[2]>min_vol_metrics)+1e-3).unsqueeze(dim=0)), axis=0)
            combined_loss += (loss_fn(combined_pred, torch.concat((y_bike.unsqueeze(dim=0), y_gtaxi.unsqueeze(dim=0), y_ytaxi.unsqueeze(dim=0)), axis=0))*combined_weight).mean()
            
            y_bike_all_arr[i] = y_bike[:,-1]
            pred_bike_all_arr[i] = pred_bike[:,-1].detach()
            y_gtaxi_all_arr[i] = y_gtaxi[:,-1]
            pred_gtaxi_all_arr[i] = pred_gtaxi[:,-1].detach()
            y_ytaxi_all_arr[i] = y_ytaxi[:,-1]
            pred_ytaxi_all_arr[i] = pred_ytaxi[:,-1].detach()
            pred_combined_bike_all_arr[i] = combined_pred[0,:,-1].detach()
            pred_combined_gtaxi_all_arr[i] = combined_pred[1,:,-1].detach()
            pred_combined_ytaxi_all_arr[i] = combined_pred[2,:,-1].detach()       
    
    bike_loss /= size
    gtaxi_loss /= size
    ytaxi_loss /= size
    combined_loss /= size
    scoring(y_bike_all_arr, pred_bike_all_arr, f"Test bike : Avg loss: {bike_loss.item():>7f}", norms_tuple[0], min_vol_metrics)
    scoring(y_bike_all_arr, pred_combined_bike_all_arr, f"Test bike combination : Avg loss: {combined_loss:>7f}", norms_tuple[0], min_vol_metrics)
    scoring(y_gtaxi_all_arr, pred_gtaxi_all_arr, f"Test gtaxi : Avg loss: {gtaxi_loss.item():>7f}", norms_tuple[1], min_vol_metrics)
    scoring(y_gtaxi_all_arr, pred_combined_gtaxi_all_arr, f"Test gtaxi combination : Avg loss: {combined_loss:>7f}", norms_tuple[1], min_vol_metrics)
    scoring(y_ytaxi_all_arr, pred_ytaxi_all_arr, f"Test ytaxi : Avg loss: {ytaxi_loss.item():>7f}", norms_tuple[2], min_vol_metrics)
    scoring(y_ytaxi_all_arr, pred_combined_ytaxi_all_arr, f"Test ytaxi combination : Avg loss: {combined_loss:>7f}", norms_tuple[2], min_vol_metrics)
    
    
def combined_training(train_loaders_tuple, test_loaders_tuple, GNN_models_tuple, combined_model, loss_fn, optimizers_tuple, epochs, norms_tuple, min_vol_metrics, patience):
    '''
    Combined models training
    Train and validate a model who take into account the influence of one mode of transport on the others
    
    Parameters
    ----------
    train_loaders_tuple : tuple containing pytorch_geometic.loader.DataLoader with training data
    test_loaders_tuple : tuple containing pytorch_geometic.loader.DataLoader with training data
    GNN_models_tuple : tuple containing torch.nn.Module with the model for each mode of transportation
    combined_model : final model combining the different transportation modes
    loss_fn : torch loss
    optimizers_tuple : tuple torch.optim optimizer for each model
    norms_tuple : tuple with the maximal volume for the volume data by transportation type
    min_vol_metrics : int with minimal number of trips demand to be considered for scoring
    '''
    loss_min, trigger = 1000,0
    for t in range(epochs):
        print(f"Epoch {t+1}:")
        loss = combined_train_loop(train_loaders_tuple, GNN_models_tuple, combined_model, loss_fn, optimizers_tuple, norms_tuple, min_vol_metrics)
        combined_test_loop(test_loaders_tuple, GNN_models_tuple, combined_model, loss_fn, norms_tuple, min_vol_metrics)
        print('')
        if loss>loss_min:
            trigger+=1
            if trigger==patience:
                print('Early stopping: loss minimal=', loss_min)
                break
        else:
            loss_min = loss
            trigger = 0
