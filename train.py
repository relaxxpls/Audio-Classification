import torch
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def train(
    model, train_loader, criterion, optimizer, epoch, log_interval=10, debug_interval=25
):
    model.train()
    train_loss = 0
    num_correct = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # ? zero the parameter gradients
        optimizer.zero_grad()

        # ? forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)

        # ? backward pass
        loss.backward()

        # ? optimize
        optimizer.step()

        train_loss += loss.item()
        num_correct += (predicted == labels).sum().item()

        # ? print training stats
        iteration = epoch * len(train_loader) + batch_idx

        if (batch_idx + 1) % log_interval == 0:
            writer.add_scalar("training loss/loss", loss, iteration)
            writer.add_scalar(
                "learning rate/lr", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar(
                "training accuracy/acc",
                num_correct / (batch_idx * len(labels)),
                iteration,
            )

        #     print(
        #         f'Epoch: {epoch}\tLoss: {loss:.6f}'
        #         f'[{batch_idx * len(inputs)}/{len(train_loader.dataset)} '
        #         f'({100. * batch_idx / len(train_loader):.0f}%)]'
        #     )

        # ? report debug image every `debug_interval` mini-batches
        # if batch_idx % debug_interval == 0:
        #     for n, (inp, pred, label) in enumerate(zip(inputs, predicted, labels)):
        #         series = (
        #             f'label_{classes[label.cpu()]}'
        #             f'_pred_{classes[pred.cpu()]}'
        #         )

        #         writer.add_image(
        #             f'Train MelSpectrogram samples/{batch_idx}_{n}_{series}',
        #             plot_signal(inp.cpu().numpy().squeeze(), series, 'hot'),
        #             iteration
        #         )

    train_loss /= len(train_loader)
    accuracy = 100 * num_correct / len(train_loader.dataset)

    return train_loss, accuracy
