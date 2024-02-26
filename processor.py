import logging
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def do_inference_separate(model,
                          val_loader,
                          num_query):
    device = "cuda"
    logger = logging.getLogger("MDIFF_CRReID.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm='yes', reranking=False)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for idx, seprate_test_loader in enumerate(val_loader):
        img_type_l = ['lr', 'hr']  # val_loader: query_loader gallery_loader
        with torch.no_grad():
            for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(seprate_test_loader):
                img = img.to(device)
                feat, _ = model(img, img_type=img_type_l[idx])
                evaluator.update((feat, pid, camid))
                img_path_list.extend(imgpath)

    # cmc, mAP, _, _, _, _, _ = evaluator.compute(dist_type='cosine')  # cosine
    cmc, mAP, _, _, _, _, _ = evaluator.compute()  # euclidean
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

