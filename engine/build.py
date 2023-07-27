import os
import data
import json
import torch
import solver
import modeling
import numpy as np
import time
import pickle
import shutil

from utils.visualization import Visualization
from utils.miscellaneous import mkdir

from tensorboardX import SummaryWriter

def get_vis_step(len_dataloader):
    vis_step = len_dataloader/10 
    diff = []
    step_size = 1
    while vis_step > step_size:
        diff += [vis_step-step_size]
        step_size = step_size*10
    index_min = np.argmin(np.array(diff))

    step_ = 10**(index_min)

    return step_

def clean_models(models_to_keep, filepath):
    print("Clean models. Keep only {}".format(models_to_keep))
    # Search for models and adapters (if they exit)
    aux = [x[0] for x in os.walk(filepath)]

    adapter_modules = [x for x in aux if "adapter_epoch" in x]

    models = [x for x in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, x)) and ".log" not in x and ".pickle" not in x ]

    # Clean them and leave only model to keep

    for module in adapter_modules:
        if int(module.split("/")[-1].split("_")[-1]) not in models_to_keep:
            print("Clean adapter {}".format(module))
            shutil.rmtree(module)

    for model in models:
        if int(model.split("_")[-1]) not in models_to_keep:
            print("Clean model {}".format(model))
            os.remove(os.path.join(filepath, model))

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, result_path: str = "./results"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.result_path = result_path
        self.min_val_loss = np.inf
        self.epoch_min_val = 0
        best_metrics = {}
        best_metrics['val_loss'] = np.inf
        best_metrics['epoch'] = 0
        best_metrics['mIoU'] = 0
        best_metrics['tIoU'] = 0
        self.best_metrics = best_metrics

    def set_best_metrics(self, best_metrics):
        self.best_metrics = best_metrics
        
    def early_stop(self, metrics):
        if metrics['val_loss'] < self.min_val_loss:
            self.counter = 0
            self.epoch_min_val = metrics['epoch']
            self.min_val_loss = metrics['val_loss']

            if metrics['mIoU'] > self.best_metrics['mIoU']:
                self.best_metrics['val_loss'] = metrics['val_loss']
                self.best_metrics['epoch'] = metrics['epoch']
                self.best_metrics['mIoU'] = metrics['mIoU']
                self.best_metrics['tIoU'] = metrics['tIoU']

                with open(os.path.join(self.result_path,"best_metrics.pickle"), "wb") as f:
                    pickle.dump(self.best_metrics, f)

            # Delete all epoch checkpoints and leave only the best one
            models_to_keep = [self.epoch_min_val, self.best_metrics['epoch']]
            clean_models(models_to_keep, self.result_path)

        elif metrics['val_loss']  > (self.min_val_loss + self.min_delta):
            self.counter += 1
            models_to_keep = [metrics["epoch"], self.best_metrics['epoch'], self.epoch_min_val]
            clean_models(models_to_keep, self.result_path)

            if self.counter >= self.patience:
                return True
        print("Best metrics: epoch: {} Loss: {} MIoU: {}".format(self.best_metrics['epoch'], self.best_metrics['val_loss'], self.best_metrics['mIoU']))
        print("Best loss: epoch: {} Loss: {}".format(self.epoch_min_val, self.min_val_loss))
        return False
    
def trainer(cfg):
    print('trainer')
    dataloader_train, dataset_size_train = data.make_dataloader(cfg, is_train=True)
    dataloader_test, dataset_size_test   = data.make_dataloader(cfg, is_train=False)

    model = modeling.build(cfg) 
    model.cuda()
    
    optimizer = solver.make_optimizer(cfg, model)
    scheduler, interval = solver.make_scheduler(cfg, optimizer, len(dataloader_train))

    total_iterations = 0
    total_iterations_val = 0
    last_epoch = 0
    best_metrics = {"best_epoch": 0, "best_mIoU": 0, "best_tIoU": None}

    early_stopper = EarlyStopper(patience=cfg.EARLY_STOPPER.PATIENCE, 
                                 min_delta=cfg.EARLY_STOPPER.MIN_DELTA, 
                                 result_path="./checkpoints/{}".format(cfg.EXPERIMENT_NAME))

    if len(cfg.TRAIN.CHECKPOINT) > 0: # Continue training from checkpoint
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT)
        with open("./checkpoints/{}/best_metrics.pickle".format(cfg.EXPERIMENT_NAME), "rb") as f:
            best_metrics = pickle.load(f)
            early_stopper.set_best_metrics(best_metrics)

        if cfg.LANGUAGE.TRAINING == "finetuning":
            print("Language model training: Finetuning")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Language model training: No finetuning")
            lm_dict =  {}
            for key, item in model.state_dict().items():
                if "language_model" in key:
                    lm_dict[key] = item
            
            # Merge with the checkpoint state dictionary:
            full_state_dict = {**lm_dict, **checkpoint["model_state_dict"]}

            # Load the full state dictionary
            model.load_state_dict(full_state_dict)

            if cfg.LANGUAGE.TRAINING == "adapter":
                print("with adapters")
                if "adapter" in checkpoint.keys():
                    model.language_model.load_adapter(checkpoint["tvg_adapter"])
                else:
                    print("Warning: No adapter path provided in checkpoint. Proceed with randomly initialized adapter.")
                    
        dataloader_train = checkpoint['dataloader_train']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if cfg.SOLVER.BASE_LR == checkpoint['optimizer_state_dict']['param_groups'][0]['lr']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # In case we want to change the base lr
            print("Change base LR from {} to {}".format(checkpoint['optimizer_state_dict']['param_groups'][0]['lr'], cfg.SOLVER.BASE_LR))
            for i in range(len(checkpoint['optimizer_state_dict']['param_groups'])):
                checkpoint['optimizer_state_dict']['param_groups'][i]['lr'] = cfg.SOLVER.BASE_LR
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        last_epoch = checkpoint['epoch'] + 1
        
        loss = checkpoint['loss']
        total_iterations = checkpoint['total_iterations']
        total_iterations_val = checkpoint['total_iterations_val']
        print("Continue training from epoch {}.".format(last_epoch))

    vis_train = Visualization(cfg, dataset_size_train)
    vis_test  = Visualization(cfg, dataset_size_test, is_train=False)

    writer_path = os.path.join(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
    writer = SummaryWriter(writer_path)

    vis_step = get_vis_step(len(dataloader_train))
    print("Visualization step: {}".format(vis_step))

    for epoch in range(last_epoch, cfg.EPOCHS):
        print(time.strftime("%Y-%m-%d:%H-%M-%S",time.localtime()))
        print('Epoch:', epoch)
        if scheduler:
            print('LR:', scheduler.get_last_lr())
        model.train()
        for iteration, batch in enumerate(dataloader_train):
            index     = batch[0]

            video_info = {}
            for key, value in batch[1].items():
                video_info[key] = value.cuda()

            query_info = {}
            for key, value in batch[2].items():
                query_info[key] = value.cuda()

            start    = batch[3].cuda()
            end      = batch[4].cuda()

            localiz  = batch[5].cuda()
            localiz_lengths = batch[6]
            time_starts = batch[7]
            time_ends = batch[8]
            factors = batch[9]
            fps = batch[10]

            loss, individual_loss, pred_start, pred_end, attention, atten_loss = model(video_info, query_info, start, end, localiz)
            
            if iteration%vis_step==0:
                print(time.strftime("%Y-%m-%d:%H-%M-%S",time.localtime()))
                print("{}/{} Loss :{}".format(iteration, len(dataloader_train), loss))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if interval == "step":
                scheduler.step()    

            vis_train.run(index, pred_start, pred_end, start, end, video_info["videoFeat_lengths"], epoch, loss.detach(), individual_loss, attention, atten_loss, time_starts, time_ends, factors, fps)

            writer.add_scalar(
                f'mlnlp/Progress_Loss',
                loss.item(),
                total_iterations)

            writer.add_scalar(
                f'mlnlp/Progress_Attention_Loss',
                atten_loss.item(),
                total_iterations)

            writer.add_scalar(
                f'mlnlp/Progress_Mean_IoU',
                vis_train.mIoU[-1],
                total_iterations)

            total_iterations += 1.


        writer.add_scalar(
            f'mlnlp/Train_Loss',
            np.mean(vis_train.loss),
            epoch)

        writer.add_scalar(
            f'mlnlp/Train_Mean_IoU',
            #np.mean(vis_train.mIoU),
            np.mean(np.concatenate(vis_train.IoU)),
            epoch)

        vis_train.plot(epoch)

        # if scheduler:
        #     torch.save({
        #         'epoch': epoch,
        #         'dataloader_train': dataloader_train,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'loss': loss,
        #         'total_iterations': total_iterations,
        #         'total_iterations_val': total_iterations_val,
        #         }, "./checkpoints/{}/model_epoch_{}".format(cfg.EXPERIMENT_NAME,epoch))
        # else:
        #     torch.save({
        #         'epoch': epoch,
        #         'dataloader_train': dataloader_train,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss,
        #         'total_iterations': total_iterations,
        #         'total_iterations_val': total_iterations_val,
        #         }, "./checkpoints/{}/model_epoch_{}".format(cfg.EXPERIMENT_NAME,epoch))

        if interval == "epoch":
            scheduler.step()
        
        checkpoint = {}

        checkpoint['epoch'] = epoch
        checkpoint['dataloader_train'] = dataloader_train
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['loss'] = loss
        checkpoint['total_iterations'] = total_iterations
        checkpoint['total_iterations_val'] = total_iterations_val

        if cfg.SOLVER.SCHEDULER:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if cfg.LANGUAGE.TRAINING == "no-finetuning" or cfg.LANGUAGE.TRAINING == "adapter":

            # Save only the non-BERT weights
            model_state_dict = {}
            for key, item in model.state_dict().items():
                if "language_model" not in key:
                    model_state_dict[key] = item
            
            checkpoint["model_state_dict"] = model_state_dict
            
            if cfg.LANGUAGE.TRAINING == "adapter":
                adapter_file = "./checkpoints/{}/adapter_epoch_{}".format(cfg.EXPERIMENT_NAME,epoch)
                model.language_model.save_adapter(adapter_file, adapter_name="tvg_adapter")

                checkpoint["adapter"] = adapter_file

        elif cfg.LANGUAGE.TRAINING == "finetuning":
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            checkpoint["model_state_dict"] = model.state_dict()

        torch.save(checkpoint, "./checkpoints/{}/model_epoch_{}".format(cfg.EXPERIMENT_NAME,epoch))

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for iteration, batch in enumerate(dataloader_test):
                index     = batch[0]

                video_info = {}
                for key, value in batch[1].items():
                    video_info[key] = value.cuda()

                query_info = {}
                for key, value in batch[2].items():
                    query_info[key] = value.cuda()

                start    = batch[3].cuda()
                end      = batch[4].cuda()

                localiz  = batch[5].cuda()
                localiz_lengths = batch[6]
                time_starts = batch[7]
                time_ends = batch[8]
                factors = batch[9]
                fps = batch[10]

                loss, individual_loss, pred_start, pred_end, attention, atten_loss = model(video_info, query_info, start, end, localiz)
                vis_test.run(index, pred_start, pred_end, start, end, video_info["videoFeat_lengths"], epoch, loss.detach(), individual_loss, attention,atten_loss, time_starts, time_ends, factors, fps)
                #print(loss)
                writer.add_scalar(
                    f'mlnlp/Progress_Valid_Loss',
                    loss.item(),
                    total_iterations_val)

                writer.add_scalar(
                    f'mlnlp/Progress_Valid_Atten_Loss',
                    atten_loss.item(),
                    total_iterations_val)

                writer.add_scalar(
                    f'mlnlp/Progress_Valid_Mean_IoU',
                    vis_test.mIoU[-1],
                    total_iterations_val)

                total_iterations_val += 1
                val_loss += loss.item()

            writer.add_scalar(
                f'mlnlp/Valid_Loss',
                np.mean(vis_test.loss),
                epoch)

            writer.add_scalar(
                f'mlnlp/Valid_Mean_IoU',
                #np.mean(vis_test.mIoU),
                np.mean(np.concatenate(vis_test.IoU)),
                epoch)

        tIoU, mIoU = vis_test.plot(epoch)
        writer.add_scalars(f'mlnlp/Valid_tIoU_th', tIoU, epoch)

        # if mIoU > best_metrics['best_mIoU']:
        #     best_metrics["best_tIoU"] = tIoU
        #     best_metrics["best_mIoU"] = mIoU
        #     best_metrics["best_epoch"] = epoch
            
        #     with open("./checkpoints/{}/best_metrics.pickle".format(cfg.EXPERIMENT_NAME), "wb") as f:
        #         pickle.dump(best_metrics, f)

        # print("Best Epoch: {}".format(best_metrics["best_epoch"]))

        # models_to_keep = [epoch, best_metrics["best_epoch"]]
        # clean_models(models_to_keep, "./checkpoints/{}".format(cfg.EXPERIMENT_NAME))

        # With Early stopping
        best_metrics["tIoU"] = tIoU
        best_metrics["mIoU"] = mIoU
        best_metrics["epoch"] = epoch
        best_metrics["val_loss"] = np.mean(vis_test.loss)
        print("val loss: {}".format(np.mean(vis_test.loss)))

        if early_stopper.early_stop(best_metrics):  
            print("Early stopping.")    
            break

        


def tester(cfg):
    print('testing')
    dataloader_test, dataset_size_test   = data.make_dataloader(cfg, is_train=False)

    model = modeling.build(cfg)

    if cfg.TEST.CHECKPOINT.startswith('.'):
        load_path = cfg.TEST.CHECKPOINT.replace(".", os.path.realpath("."))
    else:
        load_path = cfg.TEST.CHECKPOINT
    
    checkpoint = torch.load(load_path)

    if type(checkpoint) is dict:
        with open("./checkpoints/{}/best_metrics.pickle".format(cfg.EXPERIMENT_NAME), "rb") as f:
            best_metrics = pickle.load(f)

        if cfg.LANGUAGE.TRAINING == "finetuning":
            print("Language model training: Finetuning")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Language model training: No finetuning")
            lm_dict =  {}
            for key, item in model.state_dict().items():
                if "language_model" in key:
                    lm_dict[key] = item
            
            # Merge with the checkpoint state dictionary:
            full_state_dict = {**lm_dict, **checkpoint["model_state_dict"]}

            # Load the full state dictionary
            model.load_state_dict(full_state_dict)

            if cfg.LANGUAGE.TRAINING == "adapter":
                print("with adapters")
                if "adapter" in checkpoint.keys():
                    model.language_model.load_adapter(checkpoint["tvg_adapter"])
                else:
                    print("Warning: No adapter path provided in checkpoint. Proceed with randomly initialized adapter.")
    else:
        model = checkpoint
    model.cuda()

    vis_test  = Visualization(cfg, dataset_size_test, is_train=False)

    writer_path = os.path.join(cfg.VISUALIZATION_DIRECTORY, cfg.EXPERIMENT_NAME)
    writer = SummaryWriter(writer_path)

    model_name = cfg.TRAIN.CHECKPOINT.split("/")[-2] + "_" +cfg.TRAIN.CHECKPOINT.split("/")[-1]
    attention_file = "attention_weights_analysis/" + model_name + '_anno_with_attn.json'

    total_iterations = 0
    total_iterations_val = 0

    model.eval()
    epoch = 1
    results_data = {}
    with torch.no_grad():
        for iteration, batch in enumerate(dataloader_test):
            index     = batch[0]

            video_info = {}
            for key, value in batch[1].items():
                video_info[key] = value.cuda()

            query_info = {}
            for key, value in batch[2].items():
                query_info[key] = value.cuda()

            start    = batch[3].cuda()
            end      = batch[4].cuda()

            localiz  = batch[5].cuda()
            localiz_lengths = batch[6]
            time_starts = batch[7]
            time_ends = batch[8]
            factors = batch[9]
            fps = batch[10]

            loss, individual_loss, pred_start, pred_end, attention, atten_loss = model(video_info, query_info, start, end, localiz)
            vis_test.run(index, pred_start, pred_end, start, end, video_info["videoFeat_lengths"], epoch, loss.detach(), individual_loss, attention,atten_loss, time_starts, time_ends, factors, fps)
            total_iterations_val += 1

            for k,v in aux.items():
                results_data[k] = v
                # Erica ------------------------------------------------------------------------
                keys2retrieve = ['video', 'tokens', 'time_start', 'time_end', 'description']
                for key in keys2retrieve:
                    results_data[k][key] = dataloader_test.dataset.anns[k][key]
            
            if iteration%100 == 0:
                print(time.strftime("%Y-%m-%d:%H-%M-%S",time.localtime()))
                print("{}/{} Loss :{}".format(iteration, len(dataloader_test), loss))
                with open(attention_file, 'w') as outfile:
                    json.dump(results_data, outfile)

        a, mIoU = vis_test.plot(epoch)
        print("Final file:")
        with open(attention_file, 'w') as outfile:
            json.dump(results_data, outfile)
