import torch
import numpy as np
def validate(
    model,
    loader,
    loss_function,
    metric,
    step=None,
    tb_logger=None,
    device=None,
):
    """
    Evaluate model performance on validation data.

    Args:
        model: PyTorch model to validate
        loader: DataLoader containing validation data
        loss_function: Loss function to compute validation loss between prediction and target.
        metric: Metric function to evaluate segmentation against ground-truth labels.
        step: Current training step for logging (required if tb_logger provided)
        tb_logger: TensorBoard logger for recording metrics (optional)
        device: Torch device to run validation on (optional)

    Returns:
        None
    """

    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric =[]

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            predictioncopy=prediction.clone().detach()
            predictioncopy[predictioncopy>=0.5]=1
            predictioncopy[predictioncopy<0.5]=0
            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            val_loss += loss_function(prediction,y) 
            
            val_metric.append(metric(prediction,y)) # TODO

    # normalize loss and metric
    val_loss /= len(loader)
    #print(val_metric.dtype)
    #print(val_metric)
    val_metric_ch0= [sublist[0].cpu().numpy() for sublist in val_metric]
    val_metric_ch1= [sublist[1].cpu().numpy() for sublist in val_metric]
    val_metric_ch2= [sublist[2].cpu().numpy() for sublist in val_metric]
    val_metric_ch3= [sublist[3].cpu().numpy() for sublist in val_metric]
    val_metric_ch4= [sublist[4].cpu().numpy() for sublist in val_metric]
    average_val_metric_ch0= np.mean(val_metric_ch0)
    average_val_metric_ch1=np.mean(val_metric_ch1)
    average_val_metric_ch2=np.mean(val_metric_ch2)
    average_val_metric_ch3=np.mean(val_metric_ch3)
    average_val_metric_ch4=np.mean(val_metric_ch4)
    average_of_four_class=(average_val_metric_ch0+average_val_metric_ch1+average_val_metric_ch2+average_val_metric_ch3+average_val_metric_ch4)/5
    
    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        #tb_logger.add_scalar(
        #    tag="val_metric", scalar_value=val_metric, global_step=step
        #)
        
        # we always log the last validation images
        tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
        channel1=y.detach().to('cpu')[:,0,:,:]
        channel2=y.detach().to('cpu')[:,1,:,:]
        channel3=y.detach().to('cpu')[:,2,:,:]
        channel4=y.detach().to('cpu')[:,3,:,:]
        channel5=y.detach().to('cpu')[:,4,:,:]
               
        concat_channel1_images= torch.cat([channel1,channel2,channel3,channel4,channel5],dim=0)
        concat_channel1_reshaped=concat_channel1_images.unsqueeze(dim=1)
        
        tb_logger.add_images(tag="val_target", img_tensor=concat_channel1_reshaped, global_step=step)
        pred_ch1=predictioncopy.to('cpu')[:,0,:,:]
        pred_ch2=predictioncopy.to('cpu')[:,1,:,:]
        pred_ch3=predictioncopy.to('cpu')[:,2,:,:]
        pred_ch4=predictioncopy.to('cpu')[:,3,:,:]
        pred_ch5=predictioncopy.to('cpu')[:,4,:,:]
        concat_pred_channel1_images= torch.cat([pred_ch1,pred_ch2,pred_ch3,pred_ch4,pred_ch5],dim=0)
        concat_pred_channel1_reshaped=concat_pred_channel1_images.unsqueeze(dim=1)
        
        tb_logger.add_images(
            tag="val_prediction", img_tensor=concat_pred_channel1_reshaped, global_step=step
        )

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric across 5 channel: {:.4f}\n, Average Metric Ch1: {:.4f}\n,Average Metric Ch2: {:.4f}\n, Average Metric Ch3: {:.4f}\n, Average Metric Ch4: {:.4f}\n, Average Metric Ch5: {:.4f}\n".format(
            val_loss,average_of_four_class, average_val_metric_ch0,average_val_metric_ch1,average_val_metric_ch2,average_val_metric_ch3,average_val_metric_ch4
        )
    )