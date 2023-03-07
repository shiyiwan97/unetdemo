import os

import torch


class SaveWeightUtil:

    @classmethod
    def save_weight(cls, name, evaluation_param, best_value_type, weight_path, weight):
        """
        保存权重及记录
        :param name: 参数名
        :param evaluation_param: 评价标准值
        :param best_value_type: 0 = min;1 = max
        :param weight_path: 权重保存位置（权重文件夹上一级，\weight\iou\max_iou_weight.pth 传入\weight）
        :param record_path: 记录保存位置
        :return: 
        """
        record_path = os.path.join(weight_path, name, name)
        save_weight_path = os.path.join(weight_path, name, 'weight_' + ('max_' if best_value_type == 1 else 'min_') + name + '.pth')
        record_file_r = open(record_path, 'r+')
        line = record_file_r.readline().replace('\n', '')
        splitArray = line.split(' ')
        best_value = splitArray[3]
        best_value_epoch = splitArray[2]
        epoch = int(splitArray[0]) + 1

        record_file_r.seek(0)
        if evaluation_param > float(best_value) if best_value_type == 1 else evaluation_param < float(best_value):
            record_file_r.write(str(epoch) + ' max ' if best_value_type == 1 else ' min ' + str(epoch) + ' ' + str(
                evaluation_param.item()) + '\n')
            torch.save(weight, save_weight_path)
        else:
            record_file_r.write(
                str(epoch) + ' max ' if best_value_type == 1 else ' min ' + best_value_epoch + ' ' + best_value + '\n')
        record_file_r.close()
        record_file_w = open(record_path, 'a')
        record_file_w.write(str(epoch) + ' ' + str(evaluation_param.item()) + '\n')
        record_file_w.close()


if __name__ == '__main__':
    SaveWeightUtil.save_weight('iou1', 1, 1, r'..\weight', 2)
