from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, densenet121
import torch
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import PIL
import time

st = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_resnet_cam(model_path):
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=3)
    model.load_state_dict(
        torch.load(model_path, map_location=device))
    target_layer_resnet = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer_resnet], use_cuda=False)
    return cam, model.to(device)


def get_densenet_cam(model_path):
    model = densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(in_features=1024, out_features=3)
    model.load_state_dict(
        torch.load(model_path, map_location=device))
    target_layer_densenet = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer_densenet], use_cuda=True)
    return cam, model.to(device)


def get_image(path):
    return PIL.Image.open(path).convert('RGB')


def convert_img(image_rgb):
    convert = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        # Normalize to ImageNet
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    input_tensor = convert(image_rgb).float()

    v = Variable(input_tensor, requires_grad=True)
    v = v.unsqueeze(0)
    return v.to(device)


def get_visualization(path_to_image, cam_obj):
    image_rgb = get_image(path_to_image)
    converted_img = convert_img(image_rgb)
    grayscale_cam = cam_obj(input_tensor=converted_img)
    res_conv = torchvision.transforms.Resize(size=(224, 224))
    image_rgb = res_conv(image_rgb)
    img_np = np.array(image_rgb) / 255
    visualization = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)
    return visualization


df = pd.read_csv('/net/archive/groups/plggsano/fl_msc_classification/rsna/test_labels_stage_1.csv')
root_f = '/net/archive/groups/plggsano/fl_msc_classification/rsna/stage_1_test/full'
root_s = '/net/archive/groups/plggsano/fl_msc_classification/rsna/stage_1_test/segmented'
df['segmented'] = df['patient_id'].apply(lambda id: f"{root_s}/{id}.png")
df['full'] = df['patient_id'].apply(lambda id: f"{root_f}/{id}.png")

images = [(s, f, label) for s, f, label in zip(df['segmented'].values, df['full'].values, df['label'].values)]
cam_resnet_segmented, resnet_segmented = get_resnet_cam(
    '/net/archive/groups/plggsano/fl_msc_classification/iccs_models/d_rsna-segmented_m_ResNet50_r_10-c_3_bs_8_le_3_mf_3_ff_0.1_lr_0.0001_image_224_IID/best_model/ResNet50_7_acc_0.736_loss_0.624')
cam_resnet_full, resnet_full = get_resnet_cam(
    '/net/archive/groups/plggsano/fl_msc_classification/iccs_models/d_rsna-full_m_ResNet50_r_10-c_2_bs_8_le_3_mf_2_ff_0.1_lr_0.0001_image_224_IID/best_model/ResNet50_10_acc_0.757_loss_0.645')
cam_densenet_segmented, densenet_segmented = get_densenet_cam(
    '/net/archive/groups/plggsano/fl_msc_classification/iccs_models/d_rsna-segmented_m_DenseNet121_r_10-c_2_bs_8_le_3_mf_2_ff_0.1_lr_0.0001_image_224_IID/best_model/DenseNet121_7_acc_0.742_loss_0.65')
cam_densenet_full, densenet_full = get_densenet_cam(
    '/net/archive/groups/plggsano/fl_msc_classification/iccs_models/d_rsna-full_m_DenseNet121_r_10-c_3_bs_8_le_3_mf_3_ff_0.1_lr_0.0001_image_224_IID/best_model/DenseNet121_7_acc_0.737_loss_0.707')

for image_path_tuple in images:
    image_path_segmented, image_path_full, label = image_path_tuple

    converted_img_segmented = convert_img(get_image(image_path_segmented))
    converted_img_full = convert_img(get_image(image_path_full))

    pred1 = np.argmax(resnet_segmented(converted_img_segmented).cpu().detach().numpy())
    pred2 = np.argmax(densenet_segmented(converted_img_segmented).cpu().detach().numpy())
    pred3 = np.argmax(resnet_full(converted_img_full).cpu().detach().numpy())
    pred4 = np.argmax(densenet_full(converted_img_full).cpu().detach().numpy())

    # print(pred1, pred2, pred3, pred4)
    pred_set = {pred1, pred2, pred3, pred4}
    if len(pred_set) != 1 or pred1 != label:
        # print('Mismatch: ', label, pred_set)
        continue
    classes = ["Normal", "No Lung Opacity", "Lung Opacity"]
    # print("Predicted class: ", classes[pred1], label)
    vis1 = get_visualization(image_path_segmented, cam_resnet_segmented)
    vis2 = get_visualization(image_path_full, cam_resnet_full)
    vis3 = get_visualization(image_path_segmented, cam_densenet_segmented)
    vis4 = get_visualization(image_path_full, cam_densenet_full)
    plt.figure()
    plt.axis('off')
    f, axarr = plt.subplots(1, 4)

    fig1 = axarr[0].imshow(vis1)
    fig2 = axarr[1].imshow(vis2)
    fig3 = axarr[2].imshow(vis3)
    fig4 = axarr[3].imshow(vis4)

    figs = [fig1, fig2, fig3, fig4]
    for f, description in zip(figs, ['ResNet50 (segmented)', 'ResNet50', 'DenseNet121 (segmented)', 'DenseNet121']):
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        f.axes.set_title(description, {'fontsize': 7, 'fontweight': '10'})
    # plt.show()
    if not os.path.exists('results'):
        os.mkdir('results')
    cls = classes[pred1].replace(' ', '_')
    cls = classes[pred1].replace('/', '_')
    plt.savefig(f'results/{os.path.basename(image_path_segmented)[:-4]}_{cls}.pdf', bbox_inches='tight')
    plt.close()
    et = time.time()

    print(et - st)
