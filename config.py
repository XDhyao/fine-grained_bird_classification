import resnet
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import partnet
from torch.utils import data
# from scipy import ndimage
import convnext
from dataset import my_dataset
class Config:
    def __init__(self, arg):
        self.lr = arg.lr
        self.gpu_id = arg.gpu_id
        self.dim = 2048
        self.dataset_name = arg.dataset_name
        self.num_class = 200
        self.model_path = arg.model_path
        self.batch_size = arg.bs
        self.img_size = arg.img_size
        self.epochs = arg.epochs
        self.loss = arg.loss
        self.lmd_0 = arg.lmd_0
        self.lmd_1 = arg.lmd_1
        self.lmd_2 = arg.lmd_2
        self.lmd_3 = arg.lmd_3
        self.lmd_4 = arg.lmd_4
        self.lmd_5 = arg.lmd_5
        self.step_distance = arg.psd
        self.time = arg.time
        self.text = arg.text

        # self.mean = arg.mean
        # self.std = arg.std
        self.parts = arg.parts
        self.model_name = arg.model
        self.train_dataloader, self.val_dataloader, self.num_class = self.set_dataloader()
        self.model = self.set_model()

    def set_model(self):
        if self.model_name == 'resnet50':
            net = resnet.resnet50(pretrained=True, alpha=self.lmd_0)
            net.fc = nn.Linear(net.fc.in_features, self.num_class)

        elif self.model_name == 'resnet50fine':
            net = resnet.resnet50(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)
            model_trained = '/home/20181214363/models/resnet50_tsinghuadog_all_448_0.01.pth'
            print(model_trained)
            net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_trained).items()})
            
        elif self.model_name == 'resnet50ISDA':
            net = resnet.resnet50(pretrained=True, isda=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'seresnet50':
            net = resnet.seresnet50(pretrained=True)
            # state_dict_path = "/home/20181214363/PycharmProjects/models/seresnet50-pretrained-weights.pkl"
            # state_params = torch.load(state_dict_path)
            # state_params['weight'].pop('module.fc.weight')
            # state_params['weight'].pop('module.fc.bias')
            # net.load_state_dict(state_params['weight'], strict=False)
            # net.load_state_dict(state_params)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'seresnet50ISDA':
            net = resnet.seresnet50(pretrained=True, isda=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'resnet101':
            net = resnet.resnet101(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'resnet152':
            net = resnet.resnet152(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'resnet101ISDA':
            net = resnet.resnet101(pretrained=True, isda=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'seresnet101':
            net = resnet.seresnet101(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'seresnet101ISDA':
            net = resnet.seresnet101(pretrained=True, isda=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'resnext50_32x4d':
            net = resnet.resnext50_32x4d(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'seresnext50_32x4d':
            net = resnet.seresnext50_32x4d(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'resnext101_32x8d':
            net = resnet.resnext101_32x8d(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'seresnext101_32x8d':
            net = resnet.seresnext101_32x8d(pretrained=True)
            net.fc = nn.Linear(2048, self.num_class)

        elif self.model_name == 'densenet121':
            net = densenet.densenet121(pretrained=True)
            feature_num = net.classifier.in_features
            net.classifier = nn.Linear(feature_num, self.num_class)
            self.dim = 1024
        elif self.model_name == 'densenet161':
            net = densenet.densenet161(pretrained=True)
            feature_num = net.classifier.in_features
            net.classifier = nn.Linear(feature_num, self.num_class)

        elif self.model_name == 'inceptionv3':
            net = inception.inception_v3(pretrained=True)
            feature_num = net.fc.in_features
            net.fc = nn.Linear(feature_num, self.num_class)

        elif self.model_name == 'partnet50':
            net = partnet.partnet(backbone_name='resnet50', parts=self.parts, num_classes=self.num_class)
        elif self.model_name == 'partnet101':
            net = partnet.partnet(backbone_name='resnet101', parts=self.parts, num_classes=self.num_class)

        elif self.model_name == 'partnet_conv_tiny':
            net = partnet.partnet(backbone_name='convext_tiny', parts=self.parts, num_classes=self.num_class)
        elif self.model_name == 'partnet_conv_small':
            net = partnet.partnet(backbone_name='convext_small', parts=self.parts, num_classes=self.num_class)
        elif self.model_name == 'partnet_conv_base':
            net = partnet.partnet(backbone_name='convext_base', parts=self.parts, num_classes=self.num_class)

        elif self.model_name == 'partnet_swin_small':
            net = partnet.partnet(backbone_name='swin_small', parts=self.parts, num_classes=self.num_class)
        elif self.model_name == 'partnet_swin_base':
            net = partnet.partnet(backbone_name='swin_base', parts=self.parts, num_classes=self.num_class)

        elif self.model_name == 'my_resnet50':
            print('my_resnet50')
            net = resnet.resnet50(pretrained=False)
            pretrain_state_dict = torch.load('/home/20181214363/resnet/pretrained_models/resnet50_8xb32_in1k_20210831-ea4938fc.pth')['state_dict']
            import collections
            model_state_dict = collections.OrderedDict()
            for k in pretrain_state_dict:
                key = '.'.join(k.split('.')[1:])
                model_state_dict[key] = pretrain_state_dict[k]
            net.load_state_dict(model_state_dict)
            net.fc = nn.Linear(net.fc.in_features, self.num_class)

        elif self.model_name == 'my_seresnet50':
            print('my_seresnet50')
            net = resnet.seresnet50(pretrained=False)
            pretrain_state_dict = \
            torch.load('/home/20181214363/resnet/pretrained_models/se-resnet50_batch256_imagenet_20200804-ae206104.pth')['state_dict']
            import collections
            model_state_dict = collections.OrderedDict()
            for k in pretrain_state_dict:
                key = '.'.join(k.split('.')[1:])
                if 'se_layer' in key:
                    key = key.replace('.conv.', '.')
                model_state_dict[key] = pretrain_state_dict[k]
            net.load_state_dict(model_state_dict)
            net.fc = nn.Linear(net.fc.in_features, self.num_class)
        elif self.model_name == 'convnext_base':
            net = convnext.convnext_base(pretrained=True, in_22k=False, is_384=False, drop_path_rate=0.1)
            net.head = nn.Linear(net.head.in_features, self.num_class)

        elif self.model_name == 'convnext_tiny':
            net = convnext.convnext_tiny(pretrained=True, in_22k=True, drop_path_rate=0.1)
            net.head = nn.Linear(net.head.in_features, self.num_class)

        return net

    def set_dataloader(self):
        # data_transform = {
        #     "train": transforms.Compose(
        #         [transforms.Resize((600, 600)),
        #          transforms.RandomCrop((self.img_size, self.img_size)),
        #          transforms.RandomHorizontalFlip(),
        #          # transforms.RandomVerticalFlip(),
        #          # transforms.RandomGrayscale(p=0.1),
        #          # transforms.RandomRotation((0, 30), center=(224, 224)),
        #          # transforms.ColorJitter(brightness=0.5, contrast=0.5),
        #          transforms.ToTensor(),
        #          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        #          # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
        #     ),
        #
        #     "val": transforms.Compose(
        #         [transforms.Resize((600, 600)),
        #          transforms.CenterCrop((self.img_size, self.img_size)),
        #          transforms.ToTensor(),
        #          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        #          # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]
        #     )
        # }
        if self.dataset_name in ['cubbirds']:
            # data_transform = {
            #     "train": transforms.Compose(
            #         [transforms.Resize((600, 600)),
            #          transforms.RandomCrop((self.img_size, self.img_size)),
            #          transforms.RandomHorizontalFlip(),
            #          transforms.ToTensor(),
            #          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            #     ),
            #     "val": transforms.Compose(
            #         [transforms.Resize((600, 600)),
            #          transforms.CenterCrop((self.img_size, self.img_size)),
            #          transforms.ToTensor(),
            #          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            #     )}
            data_transform = {
                "train": transforms.Compose(
                    [transforms.Resize((512, 512)),
                     transforms.RandomCrop((self.img_size, self.img_size)),
                     transforms.ColorJitter(brightness=0.126, saturation=0.5),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                ),
                "val": transforms.Compose(
                    [transforms.Resize((512, 512)),
                     transforms.CenterCrop((self.img_size, self.img_size)),
                     transforms.ToTensor(),
                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                )}
        elif self.dataset_name in ['aircrafts']:
            data_transform = {
                "train": transforms.Compose(
                    [transforms.Resize((512, 512)),
                     transforms.RandomCrop((self.img_size, self.img_size)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ColorJitter(brightness=0.126, saturation=0.5),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                ),
                "val": transforms.Compose(
                    [transforms.Resize((512, 512)),
                     transforms.CenterCrop((self.img_size, self.img_size)),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                )}
        elif self.dataset_name in ['stcars']:
            data_transform = {
                "train": transforms.Compose(
                    [transforms.Resize((512, 512)),
                     transforms.RandomCrop((self.img_size, self.img_size)),
                     transforms.RandomHorizontalFlip(),
                     transforms.ColorJitter(brightness=0.126, saturation=0.5),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                ),
                "val": transforms.Compose(
                    [transforms.Resize((512, 512)),
                     transforms.CenterCrop((self.img_size, self.img_size)),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                )}
            # data_transform = {
            #     "train": transforms.Compose(
            #         [transforms.Resize((600, 600)),
            #          transforms.RandomCrop((self.img_size, self.img_size)),
            #          transforms.RandomHorizontalFlip(),
            #          transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
            #          transforms.ToTensor(),
            #          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            #     ),
            #     "val": transforms.Compose(
            #         [transforms.Resize((600, 600)),
            #          transforms.CenterCrop((self.img_size, self.img_size)),
            #          transforms.ToTensor(),
            #          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            #     )}
        # elif self.dataset_name in ['stcars', 'stdogs']:
        #     data_transform = {
        #         'train': transforms.Compose([
        #             transforms.Resize((self.img_size, self.img_size)),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.ToTensor(),
        #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #         ]),
        #         'val': transforms.Compose([
        #             transforms.Resize((self.img_size, self.img_size)),
        #             transforms.ToTensor(),
        #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #         ])}
        # elif self.dataset_name in ['aircrafts', 'nabirds']:
        #     data_transform = {
        #         "train": transforms.Compose(
        #             [transforms.Resize((600, 600)),
        #              transforms.RandomCrop((self.img_size, self.img_size)),
        #              transforms.RandomHorizontalFlip(),
        #              transforms.ToTensor(),
        #              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        #         ),
        #
        #         "val": transforms.Compose(
        #             [transforms.Resize((600, 600)),
        #              transforms.CenterCrop((self.img_size, self.img_size)),
        #              transforms.ToTensor(),
        #              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        #         )
        #     }
        else:
            raise ValueError("no this dataset")
        
        if self.dataset_name == 'trainbalance300':
            dataset_path_train = '/home/20181214363/datasets/tsinghuaDog/train_balance_300'
            dataset_path_val = '/home/20181214363/datasets/tsinghuaDog/val'
            num_class = 130

        elif self.dataset_name == 'allbalance500':
            dataset_path_train = '/home/20181214363/datasets/tsinghuaDog/train_val_balance'
            dataset_path_val = '/home/20181214363/datasets/tsinghuaDog/val'
            num_class = 130

        elif self.dataset_name == 'trainbalance500':
            dataset_path_train = '/home/20181214363/datasets/tsinghuaDog/train_balance_500'
            dataset_path_val = '/home/20181214363/datasets/tsinghuaDog/val'
            num_class = 130

        elif self.dataset_name == 'tsinghuadogs':
            dataset_path_train = '/home/20181214363/datasets/tsinghuaDog/train'
            dataset_path_val = '/home/20181214363/datasets/tsinghuaDog/val'
            num_class = 130

        elif self.dataset_name == 'all':
            dataset_path_train = '/home/20181214363/datasets/tsinghua_dogs/low-resolution'
            dataset_path_val = '/home/20181214363/datasets/tsinghuaDog/val'
            num_class = 130

        elif self.dataset_name == 'cubbirds':
            dataset_path_train = '/home/20181214363/datasets/cubbirds/train'
            dataset_path_val = '/home/20181214363/datasets/cubbirds/val'
            num_class = 200

        elif self.dataset_name == 'cubbirds_complement':
            dataset_path_train = '/home/20181214363/datasets/cubbirds_complement/train'
            dataset_path_val = '/home/20181214363/datasets/cubbirds_complement/val'
            num_class = 200

        elif self.dataset_name == 'cub_complement':
            dataset_path_train = '/home/20181214363/datasets/cubbirds/train'
            dataset_path_val = '/home/20181214363/datasets/cubbirds/val'
            num_class = 200

        elif self.dataset_name == 'nabirds':
            dataset_path_train = '/home/20181214363/datasets/nabirds/train'
            dataset_path_val = '/home/20181214363/datasets/nabirds/val'
            num_class = 555

        elif self.dataset_name == 'stcars':
            dataset_path_train = '/home/20181214363/datasets/stcars/train'
            dataset_path_val = '/home/20181214363/datasets/stcars/val'
            num_class = 196

        elif self.dataset_name == 'stdogs':
            dataset_path_train = '/home/20181214363/datasets/stdogs/train'
            dataset_path_val = '/home/20181214363/datasets/stdogs/val'
            num_class = 120

        elif self.dataset_name == 'aircrafts':
            dataset_path_train = '/home/20181214363/datasets/aircrafts/train'
            dataset_path_val = '/home/20181214363/datasets/aircrafts/val'
            num_class = 100
        else:
            raise ValueError("no this dataset")
        self.num_class = num_class
        # print(dataset_path_train)
        # print(dataset_path_val)

        # train_dataset = TsinghuaDog(root_dir=dataset_path_train, transform=data_transform["train"])
        # validate_dataset = TsinghuaDog(root_dir=dataset_path_val, transform=data_transform["val"])
        # train_dataset = datasets.ImageFolder(root=dataset_path_train, transform=data_transform["train"])
        # validate_dataset = datasets.ImageFolder(root=dataset_path_val, transform=data_transform["val"])
        #
        # if self.dataset_name == 'cub_complement':
        train_dataset = my_dataset(root_dir=dataset_path_train, transform=data_transform["train"])
        validate_dataset = my_dataset(root_dir=dataset_path_val, transform=data_transform["val"])

        train_loader = data.DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=8)
        val_loader = data.DataLoader(validate_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=8)
        return train_loader, val_loader, num_class

    def print_info(self):
        info = '+---------------------------------------------------+\n' \
               '|| init lr : {:<10f}|| batch size   : {:<10d}||\n' \
               '|| img size: {:<10d}|| epochs       : {:<10d}||\n' \
               '|| parts   : {:<10d}|| step distance: {:<10d}||\n' \
               '|| model   : {:<10s}|| loss         : {:<10s}||\n' \
               '|| dataset : {:<10s}|| gpu id       : {:<10s}||\n' \
               '|| lmd_0   : {:<10f}|| lmd_1        : {:<10f}||\n' \
               '|| lmd_2   : {:<10f}|| lmd_3        : {:<10f}||\n' \
               '|| lmd_4   : {:<10f}|| lmd_5        : {:<10f}||\n' \
               '|| data augment: {:<33s}||\n' \
               '+--------------------------------------------------+'.format(self.lr, self.batch_size, self.img_size,
                                                                             self.epochs, self.parts,
                                                                             self.step_distance,
                                                                             self.model_name, self.loss,
                                                                             self.dataset_name,
                                                                             self.gpu_id, self.lmd_0, self.lmd_1,
                                                                             self.lmd_2, self.lmd_3,
                                                                             self.lmd_4, self.lmd_5,
                                                                             'RandomCrop, RandomHorizontalFlip')
        print(info)
