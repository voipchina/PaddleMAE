import os
import numpy as np
import paddle
import paddle.vision.transforms as T
from tqdm import tqdm
from data import ImageNetDataset
from model import MAE

def train(places, dataset_root = '/home/aistudio/data/data89857/ILSVRC2012mini', batch_size = 128, image_size = 256, epochs = 400):
    mae_train_trans = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.Transpose([2, 0, 1]),
        T.Normalize(mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    data_format='CHW')])
    dataset = ImageNetDataset(f'{dataset_root}/train', f'{dataset_root}/train_list.txt',
                                mode='train', transforms=mae_train_trans)
    trainlen = dataset.__len__()
    loader = paddle.io.DataLoader(dataset,
                places=places,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0)
    model = MAE(image_size = image_size)
    optimizer = paddle.optimizer.AdamW(1e-4, weight_decay = 0.3, parameters=model.parameters())

    for e in range(epochs):
        epochloss = 0.0
        model.train()
        pbar = tqdm(total = trainlen)
        for i, data in enumerate(loader()):
            img, label = data
            img = paddle.cast(img, 'float32')
            _, routput, _, loss= model(img)
            epochloss = epochloss + loss.numpy()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            pbar.update(batch_size)
            pbar.set_postfix({'epoch': e, 'batch': f'{i}/{trainlen // batch_size}','loss': float(epochloss) / (i+1) / batch_size})
            if (i % 100 == 0 and e % 5 == 0):
                t = np.concatenate((img.numpy()[0],routput.numpy()[0]), axis = 0)
                t.tofile('images/pic%d_%d.raw' %(e,i))
        pbar.close()
        epochloss = epochloss /  trainlen
        print ("epoch:", e , "loss:",epochloss)
        if e % 5 == 0:
            paddle.save(model.state_dict(), "model/model%d.pdparams"%(e % 5))
            paddle.save(optimizer.state_dict(), "model/adam%d.pdopt"%(e % 5))

if __name__ == '__main__':
    USE_GPU = False
    if USE_GPU:
        places = paddle.fluid.cuda_places()
    else:
        os.environ['CPU_NUM'] = str(6)
        paddle.set_device('cpu')
        places = paddle.fluid.cpu_places()

    train(places = places, dataset_root = '/Users/steven/my/code/ai/PaddleMAE/data/ILSVRC2012mini')
