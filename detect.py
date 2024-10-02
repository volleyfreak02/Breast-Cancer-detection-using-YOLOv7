import torch
import argparse
from pathlib import Path

def detect(weights, source, img_size, conf_thres):
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path=weights, source='local')
    model.eval()

    # Load image
    img_path = Path(source)
    imgs = [img_path]

    # Perform detection
    results = model(imgs, size=img_size, conf_thres=conf_thres)
    results.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/yolov7-breast-cancer/weights/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='dataset/images/test', help='source directory or file for inference')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    opt = parser.parse_args()

    detect(opt.weights, opt.source, opt.img_size, opt.conf_thres)
