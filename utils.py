import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as aT
import torchvision.transforms as vT


def audio_loader(path, max_length_in_seconds=4):
    waveform, sample_rate = torchaudio.load(path)
    _, num_frames = waveform.shape
    max_frames = sample_rate * max_length_in_seconds

    # ? Pad audio with zeros if too short or cut audio if too long
    if num_frames < max_frames:
        waveform = torch.nn.functional.pad(waveform, (0, max_frames - num_frames))
    elif num_frames > max_frames:
        waveform = waveform.narrow(dim=1, start=0, length=max_frames)

    transforms = vT.Compose(
        [
            aT.Resample(44100, 22050),
            aT.MFCC(sample_rate=sample_rate, n_mfcc=64),
            aT.AmplitudeToDB(),
        ]
    )
    waveform = transforms(waveform)

    return waveform


def train(model, train_loader, criterion, optimizer, device="cpu"):
    model.train()
    train_loss = 0
    num_correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # ? zero the parameter gradients
        optimizer.zero_grad()

        # ? forward pass, backward pass & optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        num_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    accuracy = 100 * num_correct / len(train_loader.dataset)

    return train_loss, accuracy


def valid(model, valid_loader, criterion, device="cpu"):
    model.eval()
    valid_loss = 0
    num_correct = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            num_correct += (predicted == labels).sum().item()

    valid_loss /= len(valid_loader)
    accuracy = 100 * num_correct / len(valid_loader.dataset)

    return valid_loss, accuracy


def predict(model, input_tensor, classes, device="cpu"):
    model.eval()
    inputs = input_tensor.unsqueeze(1)
    inputs = inputs.to(device)

    with torch.no_grad():
        output = model(inputs)

        output = output.squeeze()
        output = F.softmax(output, dim=-1)

        accuracy, predicted = torch.max(output.data, -1)
        accuracy *= 100

        # ? provide class labels
        predicted = classes[predicted]

        return predicted, accuracy
