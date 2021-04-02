def train_autoencoder(model, train_loader, val_loader, optimizer, args):

    # Init
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    for epoch in range(args.epochs):

        running_loss = 0
        running_val_loss = 0

        for idx, batch in enumerate(train_loader):
            # Prediction
            target = to_device(batch, args.device)
            reconstruction = model(target)

            # Loss
            optimizer.zero_grad()
            loss = model.criterion(reconstruction, target)
            running_loss += loss.item()

            # Backprop and params' update
            loss.backward()
            optimizer.step()

        for idx, batch in enumerate(val_loader):
            # Prediction
            target = to_device(batch, args.device)
            reconstruction = model(target)

            # Loss
            loss = model.criterion(reconstruction, target)
            running_val_loss += loss.item()

        # Average loss over the batches during the training
        model.logs["train loss"].append(running_loss/train_size)
        model.logs["val loss"].append(running_val_loss/train_size)
