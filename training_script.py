from train import get_args, set_random_seed, train
import time, sys, os
from utils.util import Tee

if __name__ == '__main__':

    method_list = ['ANDMask', 'CORAL', 'DANN', 'ERM', 'Mixup', 'MLDG', 'MMD', 'GroupDRO',
                   'RSC', 'VREx', 'DIFEX']
    speed_list = ['50', '50-100', '100', '100-150', '150', '150-200', '200', '200-250',
                   '250', '250-300', '300', '300-350', '350']
    var_speed = ['50-100', '100-150', '150-200', '200-250', '250-300', '300-350']
    stable_speed = ['50', '100', '150', '200', '250', '300', '350']
    channel_list = ['B1-4_CH1-8', 'B5-9_CH9-16']

    #载入参数
    args = get_args()
    set_random_seed(args.seed)

    # 设置样本相关参数
    setattr(args, 'dataset', 'ZXJ_GD')
    setattr(args, 'data_dir', r'D:\DATABASE\ZXJ_GD\var_speed_sample\DG\\')

    # 设置泛化方法

    # 针对每个变速工况，使用相邻稳速工况进行泛化
    for method in method_list:
        # 设置泛化方法
        setattr(args, 'algorithm', method)
        for i in range(len(var_speed)):

            # 确定泛化domain
            test_speed = [var_speed[i], speed_list[i*2], speed_list[i*2+2]]
            domains = [speed + '_' + channel_list[0] for speed in test_speed]
            setattr(args, 'domains', domains)
            args.img_dataset["ZXJ_GD"] = domains

            # 设置记录存档
            """
            或许是由于输出变量出现了一些重叠的原因，训练记录文件的保存出现了一个非常搞笑的状况：日期最早的训练记录里会包含本次便利所有的
            全部记录（目前不能确定保存的model文件否也发生了未知的重叠）
            """
            args.output = r'D:\python_workfile\TL-comparsion\DeepDG\train_output\Target-VarSpeed_Source-StableSpeed//' \
                          + time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
            if os.path.exists(args.output) is False:
                os.makedirs(os.path.dirname(args.output + '\\'))
            sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
            sys.stderr = Tee(os.path.join(args.output, 'err.txt'))

            # 开始训练
            train(args)