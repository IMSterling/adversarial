from yolov3.models import *
from yolov3.utils.simple_dataset import *
from yolov3.utils.utils import *
import torch

def detect(image_tensor,model,device):
    """
       This is the yoloV3 detector implementation, borrowed directly from the yolov3 github repo with modifications then done by Ian .
       This function calculates the detections produced by the yoloV3 Nnet and returns them as a torch.tensor
       Params:
          image_tensor --> The torch.tensor of a batch of images
          model --> The darknet model to use to calculate detections (darknet model)
          device --> The device to run on (either 'cpu' or 'cuda:0')
       Returns:
         predictions --> A list of torch.tensors of predictions for each image in a batch

        """

    # Set variables for the image_size desired by the model and the model to use
    imgsz = (320, 192) if ONNX_EXPORT else 512
    out, image_tensor, weights, half, view_img, save_txt = 'output', image_tensor, 'yolov3-tiny.pt', 'False', 'False', 'False'

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Create a dataset object of desired size
    dataset = LoadImages(image_tensor, img_size=imgsz)


    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once

    # Initialize predictions
    predictions = []

    # For every image in the dataset
    for img, im0s, vid_cap in dataset:
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img/255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()

        # Get prediction
        pred = model(img, augment='False')[0]

        # Inference
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.6, multi_label=False, classes=None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):
            if det is not None and len(det):

                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_tensor[0,0,:,:].shape).round()

                # Append the detection to the list of predicitons
                predictions.append(det)
    # Return the list of predictions
    return(predictions)


