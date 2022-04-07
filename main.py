import argparse

from configparser import ConfigParser
from nerf import NeRFRunner

CONF_DIR = "./conf/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "NeRF argument parser.")
    parser.add_argument("--conf", type = str, default = "fern")
    args = parser.parse_args()

    conf = ConfigParser()
    conf.read(CONF_DIR + args.conf + ".ini")

    gpu = int(conf.get(args.conf, "GPU"))
    img_dir = conf.get(args.conf, "IMG_DIR")
    ckpt_path = conf.get(args.conf, "CKPT_PATH")
    low_res = int(conf.get(args.conf, "LOW_RES"))
    epoch = int(conf.get(args.conf, "EPOCH"))
    batch_ray = int(conf.get(args.conf, "BATCH_RAY"))
    learning = float(conf.get(args.conf, "LEARNING"))
    lr_gamma = float(conf.get(args.conf, "LR_GAMMA"))
    lr_milestone = list(conf.get(args.conf, "LR_MILESTONE"))
    n_coarse = int(conf.get(args.conf, "N_COARSE"))
    n_fine = int(conf.get(args.conf, "N_FINE"))
    data_type = conf.get(args.conf, "DATA_TYPE")
    step = int(conf.get(args.conf, "STEP"))
    decay_end = float(conf.get(args.conf, "DECAY_END"))
    #sched = conf.get(args.conf, "SCHED")

    run_nerf = NeRFRunner(
        gpu = gpu,
        img_dir = img_dir,
        ckpt_path = ckpt_path,
        low_res = low_res,
        epoch = epoch,
        batch_ray = batch_ray,
        learning = learning,
        lr_gamma = lr_gamma,
        lr_milestone = lr_milestone,
        n_coarse = n_coarse,
        n_fine = n_fine,
        data_type = data_type,
        step = step,
        decay_end = decay_end,
        sched = "EXP")

    run_nerf.trainer()
