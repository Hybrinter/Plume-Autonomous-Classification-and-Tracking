"""
pact.model — U-Net/ResNet-34 segmentation model for PACT plume detection.

Satisfies: REQ-AIML-HIGH-001, REQ-AIML-HIGH-002, REQ-AIML-IMAG-001,
           REQ-AIML-COMP-001, REQ-AIML-COMP-002

Submodules:
    architecture  — build_model() factory for smp.Unet with ResNet-34 encoder
    dataset       — HsgAimlDataset and download_dataset() for HSG-AIML (Zenodo)
    train         — TrainConfig, train_epoch(), validate_epoch()
    evaluate      — iou_score(), dice_score(), precision_recall()
    augmentation  — build_train_transforms(), build_val_transforms()
    inference     — InferenceEngine frozen dataclass
    quantize      — quantize_to_int8() stub (TensorRT INT8 pending)
"""
