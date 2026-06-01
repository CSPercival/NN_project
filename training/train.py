import torch

def model_train(model, loader, optimizer, device, loss_function):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)

        loss = loss_function(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        print(f"\r{num_batches} batch done", end='')
        print(f"\nloss  = {loss.item()}")

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def model_evaluate(model, loader, device, loss_function):
    model.eval()
    total_loss = 0.0
    total_loss = 0.0
    num_batches = 0

    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)

        loss = loss_function(predictions, targets)

        total_loss += loss.item()
        num_batches += 1
        print(f"\r{num_batches} batch done", end='')
    avg_loss = total_loss / num_batches
    return avg_loss