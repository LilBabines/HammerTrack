from ultralytics import YOLO
import argparse
import albumentations as A

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO26 model")

    # --- Model ---
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26m-obb.pt",
        help="Model variant to use: yolo26n.pt, yolo26s.pt, yolo26m.pt, yolo26l.pt, yolo26x.pt (default: yolo26n.pt)",
    )

    # --- Dataset ---
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset config file (e.g. data.yaml)",
    )

    # --- Training hyperparameters ---
    parser.add_argument("--epochs",   type=int,   default=200,  help="Number of training epochs (default: 100)")
    parser.add_argument("--imgsz",    type=int,   default=1024,  help="Input image size (default: 640)")
    parser.add_argument("--batch",    type=int,   default=8,   help="Batch size, -1 for auto (default: 16)")
    parser.add_argument("--workers",  type=int,   default=8,    help="Number of dataloader workers (default: 8)")

    # --- Device ---
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: '0', '0,1', 'cpu' (default: '0')",
    )

    # --- Output ---
    parser.add_argument("--project", type=str, default="fintune_runs/", help="Save results to project/name (default: runs/train)")
    parser.add_argument("--name",    type=str, default="train_main",        help="Experiment name (default: exp)")

    # --- Flags ---

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model : {args.model}")
    model = YOLO(args.model)

    print(f"Starting training on : {args.data}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        flipud= True,
        # cutmix = 0.2,
        # augmentations= a
        
    )

    


if __name__ == "__main__":
    main()