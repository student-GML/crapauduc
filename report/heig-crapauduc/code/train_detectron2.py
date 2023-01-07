from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
cfg = get_cfg()
# On charge une configuration de base
cf_file="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(cf_file))
cfg.DATASETS.TRAIN = ("triton_train",)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cf_file)  
# cfg.params... voir dans la documentation
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
predictor = DefaultPredictor(cfg)