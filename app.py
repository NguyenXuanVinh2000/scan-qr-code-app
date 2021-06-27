import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyzbar import pyzbar
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
)


def obj_detection(my_img):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    column1, column2 = st.beta_columns(2)

    column1.subheader("Input image")
    st.text("")
    plt.figure(figsize=(16, 16))
    plt.imshow(my_img)
    column1.pyplot(use_column_width=True)
    newImage = np.array(my_img.convert('RGB'))
    img, link = decode(newImage)
    st.text("")
    column2.subheader("Output image")
    st.text("")
    plt.figure(figsize=(15, 15))

    plt.imshow(img)
    column2.pyplot(use_column_width=True)
    if link == []:
        st.success("Unable to decode QR code")
    for i in link:
        st.success(i)
    st.markdown(
        "SCAN QRCODE BY NGUYEN XUAN VINH"
    )


def decode(newImage):
    net = cv2.dnn.readNet("yolov4.weights",
                          "yolov4.cfg")

    labels = []
    with open("yolo.names", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    names_of_layer = net.getLayerNames()
    output_layers = [names_of_layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Image loading
    img = cv2.cvtColor(newImage, 1)
    height, width, channels = img.shape

    # Objects detection (Converting into blobs)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True,
                                 crop=False)  # (image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    classID = []
    confidences = []
    boxes = []

    # SHOWING INFORMATION CONTAINED IN 'outputs' VARIABLE ON THE SCREEN
    for op in outputs:
        for detection in op:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # OBJECT DETECTED
                # Get the coordinates of object: center,width,height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  # width is the original width of image
                h = int(detection[3] * height)  # height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w / 2)  # Top-Left x
                y = int(center_y - h / 2)  # Top-left y

                # To organize the objects in array so that we can extract them later
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classID.append(class_id)

    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00, 1.00, 0.5, 0.01)
    nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    print(indexes)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    text1 = ""
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # To get the name of object
            label = str.upper((labels[classID[i]]))
            cv2.rectangle(img, (x, y), (x + w, y + h), 3)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            barcodes = pyzbar.decode(img)
            for barcode in barcodes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type
                text1 = "{} ({})".format(barcodeData, barcodeType)
                # cv2.putText(img, text1, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if text1 != "":
                    def KT():
                        for j in items:
                            if j == text1:
                                return True

                    if not KT():
                        items.append(text1)
    return img, items


def decode_camera(newImage):
    net = cv2.dnn.readNet("yolov4.weights",
                          "yolov4.cfg")

    labels = []
    with open("yolo.names", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    names_of_layer = net.getLayerNames()
    output_layers = [names_of_layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Image loading
    img = cv2.cvtColor(newImage, 1)
    height, width, channels = img.shape

    # Objects detection (Converting into blobs)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True,
                                 crop=False)  # (image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    classID = []
    confidences = []
    boxes = []

    # SHOWING INFORMATION CONTAINED IN 'outputs' VARIABLE ON THE SCREEN
    for op in outputs:
        for detection in op:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # OBJECT DETECTED
                # Get the coordinates of object: center,width,height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  # width is the original width of image
                h = int(detection[3] * height)  # height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w / 2)  # Top-Left x
                y = int(center_y - h / 2)  # Top-left y

                # To organize the objects in array so that we can extract them later
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classID.append(class_id)

    score_threshold = st.sidebar.slider("Confidence_threshold", 0.00, 1.00, 0.5, 0.01)
    nms_threshold = st.sidebar.slider("NMS_threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    print(indexes)

    # font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    text1 = ""
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # To get the name of object
            label = str.upper((labels[classID[i]]))
            cv2.rectangle(img, (x, y), (x + w, y + h), 3)
            #            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            barcodes = pyzbar.decode(img)
            for barcode in barcodes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type
                text1 = "{} ({})".format(barcodeData, barcodeType)
                cv2.putText(img, text1, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if text1 != "":
                    def KT():
                        for j in items:
                            if j == text1:
                                return True

                    if not KT():
                        items.append(text1)
    return img, items


def app_object_detection():
    class yolov4(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[[]]"

        def __init__(self) -> None:
            self.result_queue = queue.Queue()


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            annotated_image, result = decode_camera(image)
            link = []
            def KT(j):
                for i in link:
                    if i == j:
                        return True
            for j in result:
                if not KT(j):
                    link.append(j)
            self.result_queue.put(link)
            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=yolov4,
        async_processing=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.05
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
    if st.checkbox("Show the result scan QR code", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break


    st.markdown(
        "SCAN QRCODE BY NGUYEN XUAN VINH"
    )


def main():
    st.title("WELCOME TO SCAN QR CODE APP - UIT")
    st.write(
        "Yon can scan QR code. Click option:")

    choice = st.radio("", ("See an illustration", "Scan QR code with Image", "Scan QR code with Camera"))
    # st.write()

    if choice == "Scan QR code with Image":
        image_file = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            my_img = Image.open(image_file)
            obj_detection(my_img)
    elif choice == "Scan QR code with Camera":
        app_object_detection()
    elif choice == "See an illustration":
        my_img = Image.open("qr.jpg")
        obj_detection(my_img)


def app_media_constraints():
    """ A sample to configure MediaStreamConstraints object """
    frame_rate = 5
    WEBRTC_CLIENT_SETTINGS.update(
        ClientSettings(
            media_stream_constraints={
                "video": {"frameRate": {"ideal": frame_rate}},
            },
        )
    )
    webrtc_streamer(
        key="media-constraints",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )
    st.write(f"The frame rate is set as {frame_rate}")


if __name__ == '__main__':
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
               "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
