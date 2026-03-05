# main.py
# Entry point for joint IJEPA + diffusion training.
#
# - Loads YAML config (IJEPA + diffusion settings)
# - Optionally injects diffusion checkpoint dir from CLI
# - Spawns one process per GPU and calls train.main(args)

import argparse
import multiprocessing as mp
import os
import pprint
import sys

import yaml

from utils.distributed import init_distributed  # imported but not used here; train.py handles init
from train import main as app_main


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fname",
        type=str,
        help="Path to YAML config file",
        default="configs.yaml",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cuda:0"],
        help="Which devices to use on local machine (one process per device)",
    )
    parser.add_argument(
        "--sd_checkpoint_dir",
        type=str,
        default=None,
        help="Optional path to diffusion model checkpoint directory passed to load_sd_model",
    )

    return parser.parse_args()


def process_main(
    rank: int,
    fname: str,
    world_size: int,
    devices,
    sd_checkpoint_dir: str | None,
):
    # Restrict this process to a single visible CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()

    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # --------------------------------------------------------------------- #
    # Load config YAML
    # --------------------------------------------------------------------- #
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # Inject diffusion checkpoint dir (if provided on CLI)
    if sd_checkpoint_dir is not None:
        if "meta" not in params:
            params["meta"] = {}
        params["meta"]["sd_checkpoint_dir"] = sd_checkpoint_dir
        logger.info(f"Using sd_checkpoint_dir from CLI: {sd_checkpoint_dir}")

    # --------------------------------------------------------------------- #
    # Log rank/world_size; actual distributed init happens in train.main
    # --------------------------------------------------------------------- #
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # --------------------------------------------------------------------- #
    # Call training main
    # --------------------------------------------------------------------- #
    app_main(args=params)


if __name__ == "__main__":
    args = parse_args()

    num_gpus = len(args.devices)

    # Use spawn to be safe with CUDA + multiprocessing
    mp.set_start_method("spawn", force=True)

    # One process per GPU
    processes = []
    for rank in range(num_gpus):
        p = mp.Process(
            target=process_main,
            args=(
                rank,
                args.fname,
                num_gpus,
                args.devices,
                args.sd_checkpoint_dir,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
