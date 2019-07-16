import cv2
import torch
import numpy as np

from torch.nn import functional as F
from torchvision.transforms import functional as T

from models import HRFPN34 as Model

sh = 480
sw = 640

norm_mean = [0.485, 0.456, 0.406]
norm_std  = [0.229, 0.224, 0.225]

vid = cv2.VideoCapture(0)

model   = Model(68).cuda()
weights = torch.load("./hrfpn34.weights", map_location="cpu")['state_dict']
model.load_state_dict(weights)
model.eval()
#model   = model.half()

while True:

    ret, frame = vid.read()
    if not ret: break

    ih, iw = frame.shape[:2]
    disp_frame = np.array(frame, copy=False)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame_rgb, (sw, sh))
    img = torch.Tensor(img).float().permute(2, 0, 1).contiguous() / 255
    img = T.normalize(img, norm_mean, norm_std)
    img = img.unsqueeze(0)#.half()
    img = img.cuda(non_blocking=True)

    preds = model(img)[-1]
    preds = F.interpolate(preds, size=(ih, iw))
    preds = preds[0].max(0)[0].float().cpu().detach()
    preds = preds * 255
    preds = preds.numpy().astype(np.uint8)

    heatmap = cv2.applyColorMap(preds, cv2.COLORMAP_JET)
    disp_frame = cv2.addWeighted(heatmap, 0.2, disp_frame, 0.8, 0)

    cv2.imshow("Face Keypoints Webcam Demo", disp_frame)

    if cv2.waitKey(1) == ord("q"):
        break
