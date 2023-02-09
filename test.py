# coding:utf-8
from PIL import Image
import partnet
from torch.utils import data
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import warnings
import cv2
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def mask_find_bboxs(mask):
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    mask = (mask * 255.0).astype("uint8")
    mask = np.where(mask >= mask.mean(), mask, 0)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,
                                                                        connectivity=8)  # connectivity参数的默认值为8
    act_area = []
    for index, stat in enumerate(stats):
        stat_mask = np.where(labels == index, 1, 0)
        act_area.append((mask * stat_mask).sum() / stat_mask.sum())
    index_area = np.argmax(np.array(act_area))
    return stats[index_area]


def complement(heatmap, img):
    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    b = mask_find_bboxs(heatmap)

    # cv2.waitKey()
    x0, y0 = b[0], b[1]
    x1 = b[0] + b[2]
    y1 = b[1] + b[3]
    left_width_margin = img[y0:y1, 0:x0 + 1]
    right_width_margin = img[y0:y1, x1 - 1:]
    top_height_margin = img[0:y0 + 1, x0:x1]
    bottom_height_margin = img[y1 - 1:, x0:x1]
    # img = np.array(img)

    # print(img_path)
    # print(left_width_margin.shape, right_width_margin.shape, top_height_margin.shape, bottom_height_margin.shape)
    if b[2] > b[3]:
        top_scale = top_height_margin.shape[0] / (top_height_margin.shape[0] + bottom_height_margin.shape[0])
        bottom_scale = bottom_height_margin.shape[0] / (top_height_margin.shape[0] + bottom_height_margin.shape[0])
        # print(1, max(int(top_scale * (b[2] - b[3])), 1), max(int(bottom_scale * (b[2] - b[3])), 1))
        img_cmp = np.concatenate([cv2.resize(top_height_margin, (b[2], max(int(top_scale * (b[2] - b[3])), 1))),
                                  np.float32(img[y0:y1, x0:x1]),
                                  cv2.resize(bottom_height_margin, (b[2], max(int(bottom_scale * (b[2] - b[3])), 1)))],
                                 axis=0)
    else:
        left_scale = left_width_margin.shape[1] / (left_width_margin.shape[1] + right_width_margin.shape[1])
        right_scale = right_width_margin.shape[1] / (left_width_margin.shape[1] + right_width_margin.shape[1])
        # print(2, max(int(left_scale * (b[2] - b[3])), 1), max(int(left_scale * (b[2] - b[3])), 1))
        img_cmp = np.concatenate([cv2.resize(left_width_margin, (max(int(left_scale * (b[3] - b[2])), 1), b[3])),
                                  np.float32(img[y0:y1, x0:x1]),
                                  cv2.resize(right_width_margin, (max(int(right_scale * (b[3] - b[2])), 1), b[3]))],
                                 axis=1)
    return cv2.resize(img_cmp, (img.shape[0], img.shape[1]))


def val(model, img_batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        inputs = img_batch.to(device)

        attention_logits, logits1, logits2, logits3, logits4, batch_attention_mask = model(inputs, part=True)

        batch_object_mask = batch_attention_mask.mean(1)
        batch_object_image = []
        for batch_index in range(batch_object_mask.shape[0]):
            object_mask = batch_object_mask[batch_index].detach().cpu().numpy()
            image = inputs[batch_index].permute(1, 2, 0).cpu().numpy()
            object_image = complement(object_mask, image)
            object_image = torch.tensor(object_image).permute(2, 0, 1)
            batch_object_image.append(object_image)
        batch_object_image = torch.stack(batch_object_image).to(device)

        attention_logits_object, logits_object1, logits_object2, logits_object3, logits_object4, batch_attention_mask_part = model(
            batch_object_image, part=True)

        logits_all = attention_logits + logits1 + logits2 + logits3 + logits4

        logits_all_object = attention_logits_object + logits_object1 + logits_object2 + logits_object3 + logits_object4

        predict_value, predict__indexs = torch.sort(F.softmax(logits_all + logits_all_object, dim=-1), descending=True)

        return predict_value[:, 0:3], predict__indexs[:, 0:3]


def data(image):
    data_transform = transforms.Compose(
        [transforms.Resize((512, 512)),
         transforms.CenterCrop((448, 448)),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )

    img = np.array(image)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, 2)
    x0 = 0
    x1 = img.shape[1]
    y0 = 0
    y1 = img.shape[0]
    if img.shape[0] < img.shape[1]:
        img_cmp = np.concatenate(
            [
                cv2.resize(img[y0:y0 + 1, x0:x1], (x1, max(int(0.5 * (x1 - y1)), 1))),
                np.float32(img),
                cv2.resize(img[y1 - 1:y1, x0:x1], (x1, max(int(0.5 * (x1 - y1)), 1)))
            ],
            axis=0)
    elif img.shape[0] > img.shape[1]:
        img_cmp = np.concatenate(
            [
                cv2.resize(img[y0:y1, x0:x0 + 1], (max(int(0.5 * (y1 - x1)), 1), y1)),
                np.float32(img),
                cv2.resize(img[y0:y1, x1 - 1:x1], (max(int(0.5 * (y1 - x1)), 1), y1))
            ],
            axis=1)
    else:
        img_cmp = img
    img_cmp = Image.fromarray(np.uint8(img_cmp))
    return data_transform(img_cmp).unsqueeze(0)


def load(model_name):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'partnet50':
        model = partnet.partnet(backbone_name='resnet50', parts=4, num_classes=200)

    elif model_name == 'partnet101':
        model = partnet.partnet(backbone_name='resnet101', parts=4, num_classes=200)

    elif model_name == 'partnet_conv_tiny':
        model = partnet.partnet(backbone_name='convext_tiny', parts=4, num_classes=200)

    if torch.cuda.device_count() > 0:
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(
        'model/partnet_conv_tiny_448_0.0001_0_0_0_4_attention_pool_complement1-2_image_object2_p2p-23-34_p2pdata-bs16-cub5-epoch300_0_0.9118.pth'))
    return model


def main(model, image_path):
    print('start testing')
    image = Image.open(image_path)
    data_batch = data(image)
    top3_value, top3_index = val(model, data_batch)

    return top3_value.cpu().numpy(), top3_index.cpu().numpy()


if __name__ == '__main__':
    image = Image.open('D:/datasets/cubbirds/train/001.Black_footed_Albatross/Black_Footed_Albatross_0007_796138.jpg').convert('RGB')
    model_name = 'partnet_conv_tiny'
    model = load(model_name)
    max_epoch, max_acc = main(model, image)
