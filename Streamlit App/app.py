import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.title('Virtual Draw')
class VideoTransformer(VideoTransformerBase):

    def __init__(self):
        self.canvas = None
        self.x1, self.y1 = 0, 0


    def transform(self, frame):
        noiseth = 800
        img = frame.to_ndarray(format="bgr24")
        if self.canvas is None:
            self.canvas = np.zeros_like(img)
        # Converts images from BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([55, 130, 92], np.uint8)
        upper_green = np.array([90, 255, 255], np.uint8)

        lower_yellow = np.array([21, 130, 200], np.uint8)
        upper_yellow = np.array([40, 255, 255], np.uint8)

        # creates a mask of green coloured objects found in the frame.
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # To clear the canvas
        if np.any(mask2):
            res = cv2.bitwise_and(img, img, mask=mask2)
            edges = cv2.Canny(res, 50, 200)
            gauss = cv2.blur(edges, (7, 7), 0)  # reduce noise
            contours, hierarchy = cv2.findContours(gauss.copy(), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)
            if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noiseth:
                self.canvas = np.zeros_like(img)
        # the green coloured objects are highlighted
        res = cv2.bitwise_and(img, img, mask=mask)
        edges = cv2.Canny(res, 50, 200)
        gauss = cv2.blur(edges, (7, 7), 0)  # reduce noise
        contours, hierarchy = cv2.findContours(gauss.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(frame, contours, -1, (0, 0, 255), thickness=2)
        if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noiseth:
            c = max(contours, key=cv2.contourArea)
            # boundingReact- highlight the region of contour
            x2, y2, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if self.x1 == 0 and self.y1 == 0:
                self.x1, self.y1 = x2, y2
            else:
                self.canvas = cv2.line(self.canvas, (self.x1, self.y1), (x2, y2), [
                                       255, 0, 255], thickness=4)
            self.x1, self.y1 = x2, y2
        else:
            self.x1, self.y1 = 0, 0
        final = cv2.add(img, self.canvas)
        # cv2.imshow('mask',mask2)
        # cv2.imshow('res',self.canvas)
        # cv2.imshow('frame',cv2.flip(final,1))
        test = cv2.flip(self.canvas, 1)
        return test


webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

components.html(
        f"""
        <a href="https://www.linkedin.com/in/paul-elias-sojan/" target="_blank">
      <h4>Developed by: Paul Elias Sojan</h4>
      </a>
      <script>
    navigator.mediaDevices.getUserMedia(constraints)
    .then(function(stream) 
        audio:false
    
          </script>

	""",
    )
