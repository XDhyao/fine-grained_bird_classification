import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
import resnet
import convnext
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=(dilation, dilation), groups=groups, bias=False, dilation=dilation)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class partnet(nn.Module):
    def __init__(self, backbone_name='resnet50', parts=4, num_classes=200):
        super(partnet, self).__init__()
        if backbone_name == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=False)
            stage_channel = [256, 512, 1024, 2048]
        elif backbone_name == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=False)
            stage_channel = [256, 512, 1024, 2048]
        elif backbone_name == 'convext_tiny':
            self.backbone = convnext.convnext_tiny(pretrained=False, in_22k=True, num_classes=21841)
            stage_channel = [96, 192, 384, 768]
        elif backbone_name == 'convext_small':
            self.backbone = convnext.convnext_small(pretrained=False, in_22k=True, is_384=True, drop_path_rate=0.1)
            stage_channel = [96, 192, 384, 768]
        elif backbone_name == 'convext_base':
            self.backbone = convnext.convnext_base(pretrained=False, in_22k=True, is_384=True, drop_path_rate=0.2)
            stage_channel = [128, 256, 512, 1024]
        else:
            raise ValueError('no this backbone')

        block_expansion = 4
        self.parts = parts

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.adp_maxpool = nn.AdaptiveMaxPool2d(1)
        # self.adp_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.stage_attention = nn.Parameter(torch.randn(1, 3))
        # self.fc = nn.Linear(512 * block_expansion, num_classes)
        # self.attention_fc = nn.Linear(512 * block_expansion, num_classes)
        '''
        # stage 1
        self.conv_block1 = nn.Sequential(
            BasicConv(512 * block_expansion // 4, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 512 * block_expansion // 2, kernel_size=3, stride=1, padding=1, relu=True),
            # nn.AdaptiveMaxPool2d(1)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(512 * block_expansion // 2),
            nn.Linear(512 * block_expansion // 2, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # stage 2
        self.conv_block2 = nn.Sequential(
            BasicConv(512 * block_expansion // 2, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 512 * block_expansion // 2, kernel_size=3, stride=1, padding=1, relu=True),
            # nn.AdaptiveMaxPool2d(1)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(512 * block_expansion // 2),
            nn.Linear(512 * block_expansion // 2, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # stage 3
        self.conv_block3 = nn.Sequential(
            BasicConv(512 * block_expansion, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 512 * block_expansion // 2, kernel_size=3, stride=1, padding=1, relu=True),
            # nn.AdaptiveMaxPool2d(1)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(512 * block_expansion // 2),
            nn.Linear(512 * block_expansion // 2, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # concat features from different stages
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(512 * block_expansion // 2 * 3),
            nn.Linear(512 * block_expansion // 2 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_classes),
        )
        '''
        feature_size = 512
        num_ftrs = 2048
        self.num_ftrs = num_ftrs
        # stage 2
        self.conv_block2 = nn.Sequential(
            BasicConv(stage_channel[1], stage_channel[2], kernel_size=1, stride=1, padding=0,
                      relu=True),
            # BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(stage_channel[2]),
            nn.Linear(stage_channel[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, num_classes),
        )
        # self.classifier2 = nn.Linear(self.num_ftrs // 2, num_classes)

        # stage 3
        self.conv_block3_1 = nn.Sequential(
            BasicConv(stage_channel[2], stage_channel[2], kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
        )

        self.conv_block2_3 = nn.Sequential(
            BasicConv(stage_channel[2], stage_channel[2], kernel_size=3, stride=1, padding=1, relu=True),
            # BasicConv(feature_size, self.num_ftrs // 2, kernel_size=3, stride=1, padding=1, relu=True),
        )

        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(stage_channel[2]),
            nn.Linear(stage_channel[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, num_classes),
        )

        # stage 4
        self.conv_block4_1 = nn.Sequential(
            BasicConv(stage_channel[3], stage_channel[2], kernel_size=1, stride=1, padding=0, relu=True),
            # BasicConv(feature_size, stage_channel[2], kernel_size=3, stride=1, padding=1, relu=True),
        )

        self.conv_block3_4 = nn.Sequential(
            BasicConv(stage_channel[2], stage_channel[2], kernel_size=3, stride=1, padding=1,
                      relu=True),
            # BasicConv(feature_size, stage_channel[2], kernel_size=3, stride=1, padding=1, relu=True),
        )

        self.classifier4 = nn.Sequential(
            nn.BatchNorm1d(stage_channel[2]),
            nn.Linear(stage_channel[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, num_classes),
        )

        self.attention_classifier = nn.Sequential(
            nn.BatchNorm1d(stage_channel[2]),
            nn.Linear(stage_channel[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, num_classes),
        )

        # concat features from different stages
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(stage_channel[2] + stage_channel[2] + stage_channel[2]),
            nn.Linear(stage_channel[2] + stage_channel[2] + stage_channel[2], feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )
        '''
        self.attention_classifier = nn.Sequential(
            nn.BatchNorm1d(512 * block_expansion // 2),
            nn.Linear(512 * block_expansion // 2, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.part_classifier = nn.Sequential(
            nn.BatchNorm1d(512 * block_expansion // 2),
            nn.Linear(512 * block_expansion // 2, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.part_concat_classifier = nn.Sequential(
            nn.BatchNorm1d(512 * block_expansion // 2 * parts),
            nn.Linear(512 * block_expansion // 2 * parts, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        '''



    def forward(self, x, part):
        batch = x.shape[0]
        _, feature_map2, feature_map3, feature_map4 = self.backbone(x)

        feature_map2 = self.conv_block2(feature_map2)
        feature2 = self.adp_maxpool(feature_map2).view(batch, -1)
        x2 = self.classifier2(feature2)

        feature_map3 = self.conv_block3_1(feature_map3)
        new_feature_map3 = self.conv_block2_3((F.interpolate(feature_map3, scale_factor=2.0) + feature_map2))
        feature3 = self.adp_maxpool(new_feature_map3).view(batch, -1)
        x3 = self.classifier3(feature3)

        feature_map4 = self.conv_block4_1(feature_map4)
        new_feature_map4 = self.conv_block3_4((F.interpolate(feature_map4, scale_factor=2.0) + feature_map3))
        feature4 = self.adp_maxpool(new_feature_map4).view(batch, -1)
        x4 = self.classifier4(feature4)

        # feature_map2 = self.conv_block1(feature_map2)
        # feature2 = self.adp_maxpool(feature_map2).view(batch, -1)
        #
        # feature_map3 = self.conv_block2(feature_map3)
        # feature3 = self.adp_maxpool(feature_map3).view(batch, -1)
        #
        # feature_map4 = self.conv_block3(feature_map4)
        # feature4 = self.adp_maxpool(feature_map4).view(batch, -1)
        # x2 = self.classifier1(feature2)
        # x3 = self.classifier2(feature3)
        # x4 = self.classifier3(feature4)
        x_concat = self.classifier_concat(torch.cat((feature2, feature3, feature4), -1))

        # feature2_mask = feature_map2.mean(1, keepdim=True)
        # feature3_mask = feature_map3.mean(1, keepdim=True)
        # feature4_mask = feature_map4.mean(1, keepdim=True)
        # object_attention_mask = torch.cat((feature2_mask, feature3_mask, feature4_mask), 1)

        query_map = feature_map4
        query_map = query_map * query_map.mean(dim=1, keepdim=True)
        attention_scores = torch.einsum('bxwh, bywh->bxy', query_map, query_map)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # 频繁模式选注意力
        A = torch.sum(attention_probs, dim=-2)
        channel_value, channel_indexs = torch.sort(A, descending=True)
        channel_indexs = channel_indexs[:, :4]
        # channel_value, channel_indexs = channel_value[:, :self.parts], channel_indexs[:, :self.parts]
        batch_mask = []
        for batch_index in range(batch):
            # attention_pool_features = []
            attention_mask = []
            for part_index in range(channel_indexs.shape[1]):
                attention = query_map[batch_index][channel_indexs[batch_index][part_index]]
                attention_mask.append(attention)
                # attention_feature = (F.sigmoid(attention) * query_map[batch_index]).mean(dim=-1).mean(dim=-1)
            #     attention_pool_features.append(attention_feature)
            # attention_pool_features = torch.stack(attention_pool_features)

            # batch_avg_feature.append(attention_pool_features.mean(0))
            attention_mask = torch.stack(attention_mask)
            batch_mask.append(attention_mask)
        batch_mask = torch.stack(batch_mask)

        # part_scores = torch.einsum('bxwh, bywh->bxy', batch_mask, batch_mask)
        # part_scores = part_scores.mean(dim=-1)
        # part_values, part_indexs = torch.sort(part_scores, descending=False)
        # part_indexs = part_indexs[:, :self.parts]
        # batch_part_mask = []
        # for batch_index in range(batch):
        #     part_attention_mask = []
        #     for part_index in range(part_indexs.shape[1]):
        #         part_attention = batch_mask[batch_index][part_indexs[batch_index][part_index]]
        #         part_attention_mask.append(part_attention)
        #     part_attention_mask = torch.stack(part_attention_mask)
        #     batch_part_mask.append(part_attention_mask)
        # batch_part_mask = torch.stack(batch_part_mask)

        # 注意力特征池化
        batch_avg_feature = (F.sigmoid(batch_mask).mean(1, keepdim=True)*query_map).mean(dim=-1).mean(dim=-1)
        # batch_avg_feature = torch.stack(batch_avg_feature)
        attention_x = self.attention_classifier(batch_avg_feature)

        # x_concat = self.classifier_concat(torch.cat((feature2, feature3, feature4), -1))
        return attention_x, x2, x3, x4, x_concat, batch_mask
