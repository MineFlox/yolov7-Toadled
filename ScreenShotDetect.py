from pathlib import Path

import torch

from models.yolo import Model
from utils.general import check_requirements, set_logging
from utils.torch_utils import select_device
import numpy as np
import mss
import cv2
import time
import keyboard
import pyautogui

pyautogui.FAILSAFE = False

global hungry_meter
hungry_meter = 0

dependencies = ['torch', 'yaml']
check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('pycocotools', 'thop'))
set_logging()


def plot_boxes(img, results, move=True):
    global hungry_meter
    hungry_meter += 1
    lockSpeed = .5
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    lockDistance = 1
    x_shape, y_shape = img.shape[1], img.shape[0]
    cWidth = x_shape / 2
    cHeight = y_shape / 2
    if hungry_meter > 25 and move:
        location_continue = (1426, 676)
        location_continue2 = (1372, 936)
        location_continue0 = (831, 515)
        print(location_continue0)
        pyautogui.moveTo(location_continue0[0], location_continue0[1])
        pyautogui.click()
        print(location_continue)
        pyautogui.moveTo(location_continue[0], location_continue[1])
        pyautogui.click()
        time.sleep(.1)
        print(location_continue2)
        pyautogui.moveTo(location_continue2[0], location_continue2[1])
        pyautogui.click()
        hungry_meter = 0
        print('Continue?')

    # print(img.shape)
    # print(cWidth, cHeight, "Center")

    best_detection = None
    closest_mouse_dist = float('inf')
    bestx = 0
    besty = 0
    bestThresh = 0
    colorBomb = (0, 0, 255)

    for i in range(n):
        row = cord[i]
        Detection = labels[i]
        if row[4] >= .79:  ### threshold value for detection. We are discarding everything below this value
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)  ## BBOx coordniates
            if Detection > 0:
                cv2.putText(img, f'FOOD - {row[4]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
                colorSquare = colorBomb
            else:
                # Bomb
                cv2.putText(img, f'BOMB - {row[4]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
                colorSquare = (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), colorSquare, 2)
            centerx = (x1 + x2) / 2
            centery = (y1 + y2) / 2

            dist = abs(960 - centerx) + abs(540 - centery)

            if dist < closest_mouse_dist and Detection > 0:
                closest_mouse_dist = dist
                best_detection = row
                closest_mouse_dist = dist
                bestx = centerx
                besty = centery
                bestThresh = row[4]

    if best_detection is not None and move:
        # print(f"MOVING To Target - {bestx} - {besty} - Type: {Detection} - Thresh: {bestThresh}")

        # Try and lead them
        if bestx > 953:
            bestx = bestx - 45
        else:
            bestx = bestx + 45

        if besty > 670:
            besty = besty - 25
        else:
            besty = besty + 25

        pyautogui.moveTo(bestx, besty)
        # print(pyautogui.position())
        pyautogui.click()
        hungry_meter = 0
        # time.sleep(.1)
    # else:
    #     # Move randomly to get coins?
    #     randomx = random.randint(40, 1880)
    #     randomy = random.randint(40, 1040)
    #     pyautogui.moveTo(randomx, randomy) 

    return img


def custom(path_or_model='path/to/model.pt', autoshape=True):
    """custom mode
    Arguments (3 options):
        path_or_model (str): 'path/to/model.pt'
        path_or_model (dict): torch.load('path/to/model.pt')
        path_or_model (nn.Module): torch.load('path/to/model.pt')['model']
    Returns:
        pytorch model
    """
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model,
                                                                                      str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    print(torch.cuda.is_available())
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)


if __name__ == '__main__':
    model = custom(path_or_model='./Toadled-EndGame.pt')  # custom example

    sct = mss.mss()
    box = {
        'top': 0,
        'left': 0,
        'width': 1920,
        'height': 1080,
    }
    img = sct.grab(box)
    img = np.array(img)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    print("-------- Press 's' to begin program! --------")
    while not keyboard.is_pressed("s"):
        time.sleep(.5)

    print("Starting program in 5 seconds.")
    time.sleep(5)

    while not keyboard.is_pressed("q"):
        start_time = time.time()
        img = sct.grab(box)
        img = np.array(img)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = model([img], size=640)  # batched inference

        # Make final argument False to prevent moving and clicking.
        img = plot_boxes(img, results, True)

        # Make True if you would like to see output of drawn bounding boxes
        if False:
            cv2.namedWindow("test", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            small = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
            cv2.imshow("test", small)

        if cv2.waitKey(1) == ord('q'):
            break
