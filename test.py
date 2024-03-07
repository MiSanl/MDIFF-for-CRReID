import torch
import os
import argparse
from datasets import make_dataloader
from model.MDIFF import Res50_D_MDIFF
from processor import do_inference_separate
from utils.logger import setup_logger
"""
test code for MDIFF-CRReID
suporrted datasets: mlr_viper, mlr_caviar, mlr_market1501, mlr_dukemtmc-reid
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MDIFF-CRReID Testing")
    parser.add_argument(
        "--output", default="./log", help="path to save output log", type=str)
    parser.add_argument(
        "--gpu", default="0", help="gpu id for run testing", type=str)
    parser.add_argument(
        "--data", default="../data", help="dataset root dir", type=str)
    parser.add_argument(
        "--dataset", default="mlr_viper", help="dataset name", type=str)
    parser.add_argument(
        "--model_path", default="./check/check_resnet_mdiff_viper_best.pth", help="path for pretrained weight", type=str)
    parser.add_argument('--batch', type=int, default=32)

    args = parser.parse_args()

    output_dir = args.output
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("MDIFF_CRReID", output_dir, if_train=False)
    logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    (query_loader, test_loader,
     num_query, num_classes, camera_num, view_num) = make_dataloader(name=args.dataset, bs=args.batch, root_dir=args.data)

    if_FDR = False if args.dataset in ['mlr_market1501', 'mlr_dukemtmc-reid'] else True
    model = Res50_D_MDIFF(num_class=num_classes, FDR=if_FDR)
    if args.model_path != '':
        param_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(param_dict)
        logger.info('Loading pretrained model from {} for inference'.format(args.model_path))

    do_inference_separate(model,
                          [query_loader, test_loader],
                          num_query)
