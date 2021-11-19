import torch
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


def valid(model, valid_loader, criterion, debug_interval=25):
    model.eval()
    valid_loss = 0
    num_correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            print(inputs.shape)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            valid_loss += loss.item()
            num_correct += (predicted == labels).sum().item()

            # iteration = (epoch + 1) * len(valid_loader)
            # if batch_idx % debug_interval == 0:
            #     for n, (inp, pred, label) in enumerate(zip(inputs, predicted, labels)):
            #         series = f'label_{classes[label.cpu()]}_pred_{classes[pred.cpu()]}'

            #         writer.add_image(
            #             f'Test MelSpectrogram samples/{batch_idx}_{n}_{series}',
            #             plot_signal(inp.cpu().numpy().squeeze(), series, 'hot'), iteration
            #         )

    valid_loss /= len(valid_loader)
    accuracy = 100 * num_correct / len(valid_loader.dataset)

    return valid_loss, accuracy


def predict(model, data, classes=None):
    model.eval()

    with torch.no_grad():
        data = data.unsqueeze(1).to(device)
        output = model(data)
        accuracy, [predicted] = torch.max(output.data, 1).item()
        accuracy = 100 * accuracy

        if classes:
            predicted = classes[predicted]

        return predicted, accuracy
