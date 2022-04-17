import numpy as np
import paddle
from tqdm import tqdm


def eval_turn(model, data_loader, show_info=''):
    model.eval()
    total = data_loader.total_item_len
    num_cls = data_loader.num_cls
    correct = 0
    with paddle.no_grad():
        desc = 'Test...' if show_info == '' else f'Testing epoch {show_info} ...'
        for batch_cnt_val, data_val in enumerate(tqdm(data_loader, desc=desc)):
            inputs = data_val[0]
            labels = paddle.to_tensor(np.array(data_val[1]).astype(np.long))
            outputs = model(inputs)
            outputs_pred = outputs[0] + outputs[1][:, 0:num_cls] + outputs[1][:, num_cls:2 * num_cls]
            predicted = outputs_pred.argmax(axis=1)
            correct += (predicted == labels).sum().cpu().numpy()
        acc = 100.0 * correct / total
    return acc[0]
