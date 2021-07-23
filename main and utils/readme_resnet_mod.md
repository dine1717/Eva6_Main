# Assingment 9 - EVA 6 Modified Resnet Architecture <br>

### Bharath Kumar Bolla, Sabeesh and Dinesh ###

The following are the modifications done in the resnet architecture:<br>
There are three convolutional blocks. First and Third have a residual block. While the second does not have one. There are a total of 7,105,138 neurons in the architecture<br>
1) incorporation of a convolutional layer before the residual block.<br>
2) Addition of max pooling in the last layer and in the custom first layer.<br>
3) Addition of a second max pooling layer at the end of all layers. <br>

The network architecture is as shown below. <br>
![res mod1](https://user-images.githubusercontent.com/48343095/126528374-9cea368e-69cf-443c-b127-688abac9717f.PNG)
![arch2](https://user-images.githubusercontent.com/48343095/126528381-364b0146-564f-4954-b490-b29dc8f06bfb.PNG) <br>

4) Two types of augmentations are used here. In layer - augmentation as pary of the architecture and Albumentations as shown below
![aug2](https://user-images.githubusercontent.com/48343095/126529551-f960f19b-c394-4ddf-98ef-fbbb7091b361.PNG)
![aug layer1](https://user-images.githubusercontent.com/48343095/126529562-452f38fe-a9af-4d24-bcc2-78777325a099.png)


5) The one cycle lr policy is impleemted as follows. The LR finder library is used to find the optimal LR for the optimizer and the min and max lr for the scheduler cycles between 0.0001 and 0.1 at the end of every epoch. <br>
![onecycle lr](https://user-images.githubusercontent.com/48343095/126528658-225988a0-3858-4b52-82b3-78b8786b59b5.PNG)
![lr](https://user-images.githubusercontent.com/48343095/126528640-015427dd-4aba-4028-af3d-529b6fd17c0f.PNG)

6) The train accuracy attaines is approx 94% and test is 89% at the end of 24 epochs with just the processing steps as mentioned above.
![train accuracy](https://user-images.githubusercontent.com/48343095/126528650-d20682b8-7ccf-4a86-8631-c1626f3a36d6.PNG)

7) The repo consists of the following files - custom_resnet.py, train_test_normal.py and main2.py



