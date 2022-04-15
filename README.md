# Audio-Classification

## [Problem Statement](IE643_challengequestion.pdf)

Audio classification using Torch Audio and Librosa

## Preprocessing

* All input audio files are processed to equal lengths using zero padding or cutting
* Audio downsampled to 22050 from 41000 to reduce complexity
* Mel Frequency Cepstral Coefficients (MFCC) used for collecting features
* MFCC converted the audio classification task to an image classification task

## Training procedure

* Transfer learning using pretrained DenseNet for classifying the data. Resnet18, various CNN and RNN models were also tried out, but the results were unsatisfactory.
* Optimizer used is Adam with learning rate = 1e-3.
* Criterion used is Cross Entropy Loss
* Scheduler used StepLR whose parameters are from the PyTorch docs
* Training epochs: 20

## Final results

* Training & validation accuracy: 98% (similar due to small train test split)
* Plots have been made towards the end of the report
