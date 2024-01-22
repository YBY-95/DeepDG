from train import get_args, set_random_seed, train

method_list = ['ANDMask', 'CORAL', 'DANN', 'ERM', 'Mixup', 'MLDG', 'MMD', 'GroupDRO',
               'RSC', 'VREx', 'DIFEX']
speed_list = ['50', '50-100', '100', '100-150', '150', '150-200', '200', '200-250',
               '250', '250-300', '300', '300-350', '350']
stable_speed = ['50', '100', '150', '200', '250', '300', '350']
test_speed = ['200', '150', '250', '300', '350']
channel_list = ['B1-4_CH1-8', 'B5-9_CH9-16']

args = get_args()
set_random_seed(args.seed)

# 设置样本相关参数
setattr(args, 'dataset', 'ZXJ_GD')
setattr(args, 'data_dir', r'D:\DATABASE\ZXJ_GD\var_speed_sample\DG\\')

# 设置domain
domains = [speed + '_' + channel_list[0] for speed in test_speed]
setattr(args, 'domains', domains)

#

train(args)