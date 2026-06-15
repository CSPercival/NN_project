import torch

from tqdm import tqdm

def model_train(model, loader, optimizer, device, loss_function):
    model.train()
    total_loss = 0.0
    num_batches = 0

    loop = tqdm(loader, leave=True)
    for images, targets in loop:

        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)

        loss = loss_function(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
        num_batches += 1

    avg_loss = total_loss.item() / num_batches
    print(f"\nAvg Training Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def model_evaluate(model, loader, device, loss_function):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    loop = tqdm(loader, leave=True)
    for images, targets in loop:

        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)

        loss = loss_function(predictions, targets)

        total_loss += loss.detach()
        num_batches += 1

    avg_loss = total_loss.item() / num_batches
    print(f"\nAvg Eval Loss: {avg_loss:.4f}")
    return avg_loss