import os
import re
import pandas as pd

def str2dataframe(data_str):
    # 首先按换行符分割字符串成行列表
    rows = data_str.strip().split('\n')

    # 准备一个空列表来存储分割后的数据
    data1 = []
    data2 = []

    # 对每一行按冒号分割成列，并存储到data列表中
    for row in rows:
        if 'data_dir' in row:
            continue
        elif 'img_dataset' in row:
            continue
        elif 'complet time' in row:
            continue
        elif 'start training fft teacher net' in row:
            continue
        elif 'cls loss' in row:
            continue
        elif 'done' in row:
            continue
        elif row == '':
            continue
        else:
            columns = row.split(':')
            data1.append(columns[0])
            data2.append(columns[1])
        # 使用分割后的数据创建DataFrame
    df = pd.DataFrame([data2], columns=data1)

    return df

def add_colume():
    # 为了能够识别out文件中的最后一行，添加一个结束标识符
    # 设置你想要搜索的目录
    base_directory = r'D:\python_workfile\TL-comparsion\DeepDG\train_output'

    # 遍历目录及其子目录中的所有文件
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            # 检查文件名是否为 'out.txt'
            if file == 'out.txt':
                # 构建文件的完整路径
                file_path = os.path.join(root, file)
                # 以追加模式打开文件，并在末尾添加新行
                with open(file_path, 'a') as f:
                    f.write('\n=======hyper-parameter used========')  # 添加新行

    print("All 'out.txt' files have been updated.")

class model_info:
    def __init__(self, record_dir):
        self.record_dir = record_dir
        self.record_name = os.path.basename(record_dir)
        self.hyper_parameter_list = pd.DataFrame()
        self.result_list = pd.DataFrame()

    def outfile_read(self):
        for subdir_name in os.listdir(self.record_dir):
            record_dir = self.record_dir
            subdir_path = os.path.join(record_dir, subdir_name)
            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith('out.txt'):
                        file_path = os.path.join(subdir_path, file_name)
                        # 收集所有模型参数
                        with open(file_path, 'r') as txt_file:
                            content = txt_file.read()
                            start_marker = '=======hyper-parameter used========\n=========================================='
                            end_marker = '===========start training==========='
                            # 使用正则表达式匹配开始和结束标记之间的内容
                            pattern = re.escape(start_marker) + '(.*?)' + re.escape(end_marker)
                            matches = re.findall(pattern, content, re.DOTALL)

                        for block in matches:
                            df = str2dataframe(block)
                            self.hyper_parameter_list = self.hyper_parameter_list.append(df)

                        #收集所有模型训练最终结果
                        with open(file_path, 'r') as txt_file:
                            content = txt_file.read()
                            start_marker = '===========epoch 199==========='
                            end_marker = '=======hyper-parameter used========'
                            # 使用正则表达式匹配开始和结束标记之间的内容
                            pattern = re.escape(start_marker) + '(.*?)' + re.escape(end_marker)
                            matches = re.findall(pattern, content, re.DOTALL)

                        for block in matches:
                            df = str2dataframe(block)
                            self.result_list = self.result_list.append(df)

            # 这里是记录文件的重叠，所以简单粗暴的直接读取最新的文件，后续可以考虑加上一个最早时间记录的判定
            break

    def info2csv(self):



if __name__ == '__main__':
    dir = r'D:\python_workfile\TL-comparsion\DeepDG\train_output\Target-VarSpeed_Source-StableSpeed'
    info = model_info(dir)
    info.outfile_read()
    print(info.hyper_parameter_list)





