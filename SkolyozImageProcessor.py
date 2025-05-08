import cv2
import numpy as np
import mediapipe as mp
import math
from SkolyozFramework import BaseImageProcessor

class SkolyozImageProcessor(BaseImageProcessor):
    def __init__(self, img_size=(224, 224)):
        super().__init__(img_size)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    def preprocess(self, image):
        return np.expand_dims(self.normalize_image(self.resize_image(image)), axis=0)

    def detect_spine_in_xray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cv2.GaussianBlur(gray, (5, 5), 0))
        edges = cv2.Canny(enhanced, 50, 150)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours = sorted(cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea, reverse=True)[:5]
        spine = [c for c in contours if cv2.boundingRect(c)[3]/cv2.boundingRect(c)[2] > 2]
        
        result = image.copy()
        cv2.drawContours(result, spine, -1, (0, 255, 0), 2)
        
        if not spine:
            return result, []
        
        points = np.vstack([c.reshape(-1, 2) for c in spine])
        points = points[np.argsort(points[:, 1])]
        curve = [[np.mean(points[points[:,1]==y][:,0]), y] for y in np.unique(points[:,1])]
        for i in range(1, len(curve)):
            cv2.line(result, tuple(np.int32(curve[i-1])), tuple(np.int32(curve[i])), (255, 0, 0), 2)

        return result, np.array(curve)

    def detect_spine_with_mediapipe(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        with self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
            result = pose.process(img_rgb)
            output = image.copy()
            points = []

            if result.pose_landmarks:
                self.mp_drawing.draw_landmarks(output, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                h, w = output.shape[:2]
                for i in [0, 11, 12, 23, 24]:
                    lm = result.pose_landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append([x, y])
                    cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
                for i in range(1, len(points)):
                    cv2.line(output, tuple(points[i-1]), tuple(points[i]), (255, 0, 0), 2)
            return output, np.array(points)

    def extract_spine_contour(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea, default=None)
        result = image.copy()
        if max_contour is not None:
            cv2.drawContours(result, [max_contour], -1, (0, 255, 0), 2)
        return result

    def calculate_cobb_angle(self, curve):
        if curve is None or len(curve) < 10:
            return None
        t, b = curve[:len(curve)//3], curve[2*len(curve)//3:]
        if len(t) > 1 and len(b) > 1:
            top_angle = math.atan(np.polyfit(t[:,1], t[:,0], 1)[0]) * 180 / math.pi
            bot_angle = math.atan(np.polyfit(b[:,1], b[:,0], 1)[0]) * 180 / math.pi
            return abs(top_angle - bot_angle)
        return None

    def enhance_xray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
