import numpy as np
import time
import paddle
import paddle.nn as nn
from tqdm import tqdm
from utils.eval_model import eval_turn
from utils.utils import LabelSmoothingCrossEntropy


def train(config, model, epoch_num, start_epoch, optimizer, scheduler, data_loader, date_suffix, ls):
    # 日志
    if date_suffix is None:
        date_suffix = time.strftime('%Y%m%d-%H%M%S')
    log_file = f'{config.log_folder}/{config.describe}-{date_suffix}.log'
    save_params = f'{config.save_dir}/{config.describe}-{date_suffix}.pdparams'
    best_acc = 0

    # 训练
    beta = 1
    gamma = 0.01 if config.dataset == 'STCAR' or config.dataset == 'AIR' else 1
    get_ce_loss = LabelSmoothingCrossEntropy(epsilon=ls)
    add_loss = nn.L1Loss()
    for epoch in range(start_epoch, epoch_num):
        scheduler.step(epoch)
        model.train()
        losses = [0] * 4
        for _, data in enumerate(tqdm(data_loader['train'], desc=f'Training epoch {epoch + 1}...')):
            inputs, labels, labels_swap, swap_law, img_names = data
            labels = paddle.to_tensor(np.array(labels))
            labels_swap = paddle.to_tensor(np.array(labels_swap))
            swap_law = paddle.to_tensor(np.array(swap_law).astype(np.float32))
            optimizer.clear_grad()
            outputs = model(inputs)
            ce_loss = get_ce_loss(outputs[0], labels)
            swap_loss = get_ce_loss(outputs[1], labels_swap) * beta
            law_loss = add_loss(outputs[2], swap_law) * gamma
            loss = ce_loss + swap_loss + law_loss
            loss.backward()
            optimizer.step()
            losses[0] += ce_loss.item()
            losses[1] += swap_loss.item()
            losses[2] += law_loss.item()
            losses[3] += 1  # batch个数
        losses[0] /= losses[3]
        losses[1] /= losses[3]
        losses[2] /= losses[3]

        # 保存损失
        info = f'Train {epoch + 1} Ce Loss {losses[0]} Swap Loss {losses[1]} Law Loss {losses[2]}\n'
        print(info, flush=True)
        with open(log_file, 'a+') as f:
            f.write(info)
        # 进行测试，保存测试集正确率，并保存最好的权重
        acc = eval_turn(model, data_loader['test'], epoch + 1)
        info = f'Test {epoch + 1} Accuracy={acc}\n'
        print(info, flush=True)
        with open(log_file, 'a+') as f:
            f.write(info)
        if acc > best_acc:
            best_acc = acc
            paddle.save(model.state_dict(), save_params)
