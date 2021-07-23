import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch 

def generate_gradcam(predicted_class,pred,model, img, true_class_label, predicted_class_label, heat_gradient):
    pred[:,predicted_class].backward()
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(img).detach()
    for i in range(256):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    # relu on top of the heatmap
    heatmap = np.maximum(heatmap.cpu(), 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    # plt.matshow(heatmap)

    heatmap1 = cv2.resize(np.array(heatmap), (32, 32))
    heatmap1 = np.uint8(255 * heatmap1)
    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
    superimposed_img = heatmap1*heat_gradient  + np.transpose(img[0].cpu().numpy(),(1,2,0))
    a = f'Original_class {true_class_label} predicted_class {predicted_class_label}'
    plt.figure(figsize=(5,5))
    
    plt.subplot(1,2,1)
    plt.imshow(np.transpose(img[0].cpu().numpy(),(1,2,0))/2+0.5)
    plt.subplot(1,2,2)
    plt.imshow(superimposed_img)
    plt.title(a)
    
def get_true_pred(testloader,activated_mod,count_images,class_dictionary, heat_gradient):
    passed = 0
    fail = 0
    count = 0
    activated_mod.eval()
    batch = iter(testloader)
    for i in range(len(testloader)):

        img, label = next(batch)
        img = img.to("cuda")
        pred = activated_mod(img)
        predicted_class = np.argmax(np.array(pred.detach().cpu()))
        if int(label.item()) == int(predicted_class):
            # print(f'true_class={label.item()}',f'predicted_class={predicted_class}', True)
            passed+=1
        else:
            count+=1
            true_class_label = class_dictionary[int(label.item())]
            predicted_class_label = class_dictionary[int(predicted_class)]
            # print(f'true_class={label.item()}',f'predicted_class={predicted_class}', False)
            generate_gradcam(predicted_class, pred,activated_mod, img, true_class_label, predicted_class_label,heat_gradient)
            fail+=1
            if count >= count_images:
                break

