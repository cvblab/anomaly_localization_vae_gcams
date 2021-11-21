from datasets import *
from trainers import *
import numpy as np
import random
import argparse
torch.autograd.set_detect_anomaly(True)


def main(args):

    exp = {"dir_datasets": args.dir_datasets,
           "dir_out": args.dir_out,
           "load_weigths": args.load_weigths, "epochs": args.epochs, "item": args.item, "method": args.method,
           "input_shape": args.input_shape,
           "batch_size": 32, "lr": 1e-05, "zdim": 32, "images_on_ram": True, "wkl": 1, "wr": 1, "wadv": 0, "wc": 0,
           "wae": args.wae, "epochs_to_test": 50, "dense": True, "expansion_loss": True, "log_barrier": True,
           "channel_first": True, "normalization_cam": args.normalization_cam, "avg_grads": True, "t": 20, "context": False,
           "level_cams": -4, "p_activation_cam": args.p_activation_cam, "bayesian": False, "loss_reconstruction": "bce",
           "hist_match": True,
           "expansion_loss_penalty": args.expansion_loss_penalty, "restoration": False, "threshold_on_normal": False, "n_blocks": 5}

    metrics = []
    for iteration in [0, 1, 2]:

        # Set dataset and data loaders
        test_dataset = TestDataset(exp['dir_datasets'], item=exp['item'], partition='val', input_shape=(1, 224, 224),
                                   channel_first=True, norm='max', histogram_matching=True)

        dataset = MultiModalityDataset(exp['dir_datasets'], exp['item'], input_shape=exp['input_shape'],
                                       channel_first=exp['channel_first'], norm='max', hist_match=exp['hist_match'])

        train_generator = WSALDataGenerator(dataset, partition='train', batch_size=exp['batch_size'], shuffle=True)

        # Set trainer and train model
        trainer = WSALTrainer(exp["dir_out"], item=exp["item"], method=exp["method"], zdim=exp["zdim"], lr=exp["lr"],
                              input_shape=exp["input_shape"], expansion_loss=exp["expansion_loss"], wkl=exp["wkl"],
                              wr=exp["wr"], wadv=exp["wadv"], wae=exp["wae"], epochs_to_test=exp["epochs_to_test"],
                              load_weigths=exp["load_weigths"],
                              n_blocks=exp["n_blocks"], dense=exp["dense"], log_barrier=exp["log_barrier"],
                              normalization_cam=exp["normalization_cam"], avg_grads=exp["avg_grads"], t=exp["t"],
                              context=exp["context"], level_cams=exp["level_cams"], iteration=iteration,
                              p_activation_cam=exp["p_activation_cam"], bayesian=exp["bayesian"],
                              loss_reconstruction=exp["loss_reconstruction"],
                              expansion_loss_penalty=exp["expansion_loss_penalty"], restoration=exp["restoration"],
                              threshold_on_normal=exp["threshold_on_normal"])

        # Save experiment setup
        with open(exp['dir_out'] + 'setup.json', 'w') as fp:
            json.dump(exp, fp)

        # Train
        trainer.train(train_generator, test_dataset, exp['epochs'])

        # Save overall metrics
        metrics.append(list(trainer.metrics.values()))

    # Compute average performance and save performance in dictionary

    metrics = np.array(metrics)
    metrics_mu = np.mean(metrics, 0)
    metrics_std = np.std(metrics, 0)

    labels = list(trainer.metrics.keys())
    metrics_mu = {labels[i]: metrics_mu[i] for i in range(0, len(labels))}
    metrics_std = {labels[i]: metrics_std[i] for i in range(0, len(labels))}

    with open(exp['dir_out'] + exp['item'][0] + '/' + 'metrics_avg_val.json', 'w') as fp:
        json.dump(metrics_mu, fp)
    with open(exp['dir_out'] + exp['item'][0] + '/' + 'metrics_std_val.json', 'w') as fp:
        json.dump(metrics_std, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_datasets", default="../data/BRATS_5slices/", type=str)
    parser.add_argument("--dir_out", default="../data/results/proposed/", type=str)
    parser.add_argument("--method", default="proposed", type=str)
    parser.add_argument("--expansion_loss_penalty", default="log_barrier", type=str)
    parser.add_argument("--normalization_cam", default="sigm", type=str)
    parser.add_argument("--item", default=["flair"], type=list)
    parser.add_argument("--load_weigths", default=False, type=bool)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--t", default=20, type=int)
    parser.add_argument("--wae", default=10, type=int)
    parser.add_argument("--p_activation_cam", default=0.2, type=float)
    parser.add_argument("--input_shape", default=[1, 224, 224], type=list)

    args = parser.parse_args()
    main(args)


