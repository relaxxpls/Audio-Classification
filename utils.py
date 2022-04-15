import torch


def train(model, train_loader, criterion, optimizer, device="cpu"):
    model.train()
    train_loss = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # ? zero the parameter gradients
        optimizer.zero_grad()

        # ? forward pass, backward pass & optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def valid(model, valid_loader, criterion, device="cpu"):
    model.eval()
    valid_loss = 0
    num_correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            num_correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_loader)
    accuracy = 100 * num_correct / len(valid_loader.dataset)

    return valid_loss, accuracy
