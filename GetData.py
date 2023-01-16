import mss
import keyboard
import time

# Wait until space is pressed 

keyboard.wait('space')
image_counter = 0

while not keyboard.is_pressed("q"):
    with mss.mss() as sct:
        filename = sct.shot(mon=2, output=f".\ImageData\Toadled-{image_counter}.png")
        print(filename)
        image_counter += 1
        time.sleep(.75)