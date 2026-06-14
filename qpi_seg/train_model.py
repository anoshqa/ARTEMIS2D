import torch
def center_crop(x, target_spatial_shape):
    """Center-crop ``x`` so its spatial dimensions match ``target_spatial_shape``.

    The first two dimensions of ``x`` (batch and channel) are kept as-is; only
    the trailing spatial dimensions are cropped. The crop is centered: any
    excess on each side is removed symmetrically (when the size difference is
    odd, the extra pixel is taken off the trailing side because we use
    floor-division).

    Args:
        x: Input tensor shaped ``(batch, channels, *spatial)``.
        target_spatial_shape: Target sizes for the spatial dimensions only
            (e.g. ``(H, W)`` for 2D or ``(D, H, W)`` for 3D). Every entry must
            be ``<=`` the corresponding spatial size of ``x``.

    Returns:
        A view of ``x`` cropped to ``(batch, channels, *target_spatial_shape)``.
    """

    x_target_size = x.size()[:2] + torch.Size(target_spatial_shape)

    offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

    slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

    return x[slices]
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

        # zero the gradients for this iteration
        optimizer.zero_grad()

        # apply model and calculate loss
        prediction = model(x)
        # if necessary, crop the masks to match the model output shape
        if prediction.shape != y.shape:
            y = center_crop(y, prediction.size()[2:])
        if y.dtype != prediction.dtype:
            y = y.type(prediction.dtype)
        loss = loss_function(prediction, y)

        # backpropagate the loss and adjust the parameters
        loss.backward()
        optimizer.step()
        if batch_id % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_id * len(x),
                    len(loader.dataset),
                    100.0 * batch_id / len(loader),
                    loss.item(),
                )
            )

        # log to tensorboard
        if tb_logger is not None:
            step = epoch * len(loader) + batch_id
            tb_logger.add_scalar(
                tag="train_loss", scalar_value=loss.item(), global_step=step
            )
            # check if we log images in this iteration
            if step % log_image_interval == 0:
                #x = unnormalize(x)
                tb_logger.add_images(
                    tag="input", img_tensor=x.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="target", img_tensor=y.to("cpu"), global_step=step
                )
                tb_logger.add_images(
                    tag="prediction",
                    img_tensor=prediction.to("cpu").detach(),
                    global_step=step,
                )
                combined_image = torch.cat(
                    [x, pad_to_size(y, x.size()), pad_to_size(prediction, x.size())],
                    dim=3,
                )

                tb_logger.add_images(
                    tag="input_target_prediction",
                    img_tensor=combined_image,
                    global_step=step,
                )

        if early_stop and batch_id > 5:
            print("Stopping test early!")
            break