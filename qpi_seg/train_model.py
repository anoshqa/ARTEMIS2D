import torch
def train_model(
    model,
    loader,
    optimizer,
    loss_function,
    epoch,
    log_interval=100,
    log_image_interval=20,
    tb_logger=None,
    device=None,
    early_stop=False,
):
    if device is None:
        if torch.cuda.is_available():
            device=torch.device("cuda")
        else:
            device=torch.device("cpu")
    model.train()
    model = model.to(device)

    # iterate over the batches of this epoch
    for batch_id, (x, y) in enumerate(loader):
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)
        #print(y.dtype)
        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        predictioncopy=prediction.clone().detach()
        predictioncopy[predictioncopy>=0.5]=1
        predictioncopy[predictioncopy<0.5]=0
        # if necessary, crop the masks to match the model output shape
        if prediction.shape[-2:] != y.shape[-2:]:
            raise RuntimeError
        #print(prediction.shape)
        #print(prediction.dtype)
        #print(y.shape)
        #print(y.dtype)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()
        
        if batch_id % 20 == 0 or batch_id == len(loader) - 1:
            print(f"Epoch: {epoch} - Batch: {batch_id}/{len(loader)} - Loss: {loss.item()}")
            tb_logger.add_scalar(tag="train_loss", scalar_value=loss, global_step=epoch * len(loader) + batch_id)
        #if batch_id % log_interval == 0:
        #    print(
        #        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #            epoch,
        #            batch_id * len(x),
        #            len(loader.dataset),
        #            100.0 * batch_id / len(loader),
        #            loss.item(),
        #        )
        #    )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            #tb_logger.add_scalar(
            #    tag="train_loss", scalar_value=loss.item(), global_step=step
            #)
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                #x = unnormalize(x)
                #print(x.shape)
                #print(y.shape)
                tb_logger.add_images(
                    tag="image", img_tensor=x.to("cpu"), global_step=step
                )
                #print(y.shape)
                channel1=y.detach().to('cpu')[:,0,:,:]
                channel2=y.detach().to('cpu')[:,1,:,:]
                channel3=y.detach().to('cpu')[:,2,:,:]
                channel4=y.detach().to('cpu')[:,3,:,:]
                channel5=y.detach().to('cpu')[:,4,:,:]
               
                concat_channel1_images= torch.cat([channel1,channel2,channel3,channel4,channel5],dim=0)
                concat_channel1_reshaped=concat_channel1_images.unsqueeze(dim=1)
                
                #print(concat_channel1_reshaped.shape)
                tb_logger.add_images(
                    tag="masks", img_tensor=concat_channel1_reshaped, global_step=step
                )
                pred_ch1=predictioncopy.to('cpu')[:,0,:,:]
                pred_ch2=predictioncopy.to('cpu')[:,1,:,:]
                pred_ch3=predictioncopy.to('cpu')[:,2,:,:]
                pred_ch4=predictioncopy.to('cpu')[:,3,:,:]
                pred_ch5=predictioncopy.to('cpu')[:,4,:,:]
                concat_pred_channel1_images= torch.cat([pred_ch1,pred_ch2,pred_ch3,pred_ch4,pred_ch5],dim=0)
                concat_pred_channel1_reshaped=concat_pred_channel1_images.unsqueeze(dim=1)
                tb_logger.add_images(tag="prediction",img_tensor=concat_pred_channel1_reshaped,global_step=step)
                #combined_image = torch.cat(
                #    [x, pad_to_size(y, x.size()), pad_to_size(prediction, x.size())],
                #    dim=3,
                #)

                #tb_logger.add_images(
                #    tag="input_target_prediction",
                #    img_tensor=combined_image,
                #    global_step=step,
                #)

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break