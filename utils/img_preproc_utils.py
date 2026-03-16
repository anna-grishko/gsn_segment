import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology, filters, measure


# Image subprocessor class
class ImgProcessor(object):
    def __init__(self, img):
        if img is None:
            raise ValueError("Input image cannot be None")
        self.img = img

    @property
    def shape(self):
        return self.img.shape

    @classmethod
    def read(cls, folder, file_name):
        path = os.path.join(folder, file_name)
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not read image at {path}")
            return None
        return cls(img)

    def resize(self, target_height=500):
        # Select ROI
        roi = cv2.selectROI("select the area and press enter", self.img)
        # close the window
        cv2.destroyAllWindows()
        # Crop image
        image_to_process = self.img

        if roi != (0, 0, 0, 0):
            x, y, w, h = [int(v) for v in roi]
            image_to_process = self.img[y:y + h, x:x + w]

        # Calculate aspect ratio
        h_orig, w_orig = image_to_process.shape[:2]
        scale = target_height / h_orig
        target_width = int(w_orig * scale)

        self.img = cv2.resize(image_to_process, (target_width, target_height),
                              interpolation=cv2.INTER_CUBIC)
        return self

    def mask(self, blur_func=cv2.bilateralFilter, **blur_params):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply blur with dynamic parameters
        if blur_func:
            # Default params for bilateral if none provided
            if blur_func == cv2.bilateralFilter and not blur_params:
                blur_params = {'d': 15, 'sigmaColor': 100, 'sigmaSpace': 5}
            img_blur = blur_func(img_gray, **blur_params)
        else:
            img_blur = img_gray

        # Simple Otsu threshold
        img_open = morphology.area_opening(img_blur)
        thresh = filters.threshold_otsu(img_open)
        mask = (img_open <= thresh).astype(np.uint8)

        return {
            "gray": img_gray,
            "blur": img_blur,
            "open": img_open,
            "mask": mask
        }


class ContourGSN:
    def __init__(self, processor_obj, mask):
        # We take the whole processor object to access .img and .shape easily
        self.processor = processor_obj
        self.mask = mask

        # Core data (initialized by find_contour)
        self.contours = []
        self.gsn_contour = None
        self.gsn_contour_idx = None

        # Derived data (initialized by segment_gsn)
        self.gsn_curve = None
        self.points = []
        self.flipped_data = {
            "img": None,
            "curve": None,
            "contour": None,
            "points": []
        }

    @classmethod
    def find_contour(cls, processor_obj, mask):
        instance = cls(processor_obj, mask)
        instance.contours = measure.find_contours(mask, level=0.5)

        if not instance.contours:
            raise ValueError("No contours found in the provided mask.")

        # GSN is never a closed contour
        # -> find the longer unclosed contour
        # for now substitute by the longest contour
        instance.gsn_contour_idx = np.argmax([contour.shape[0] for contour in instance.contours])  # find index of the biggest contour
        instance.gsn_contour = instance.contours[instance.gsn_contour_idx]

        return instance

    def show_contours(self, all=True):
        fig, ax = plt.subplots()
        # img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ax.imshow(cv2.cvtColor(self.processor.img, cv2.COLOR_BGR2RGB), cmap=plt.cm.Spectral)
        # ax.imshow(self.img) #, cmap=plt.cm.gray)
        if all:
            for contour in self.contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        else:
            ax.plot(self.gsn_contour[:, 1], self.gsn_contour[:, 0], linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def segment_gsn(self):
        """Segments the Great Sciatic Notch based on landmarks."""
        point_1, point_2 = get_landmarks(self.processor.img, self.gsn_contour)
        # find the closest point to the first point
        distances = np.linalg.norm(self.gsn_contour - point_1, axis=1)
        idx1 = np.argmin(distances)
        # find the closest point to the second point
        distances = np.linalg.norm(self.gsn_contour - point_2, axis=1)
        idx2 = np.argmin(distances)

        # the final curve segment contains points between those indexes
        start, end = min(idx1, idx2), max(idx1, idx2)
        self.gsn_curve = self.gsn_contour[start:end]
        self.points = [self.gsn_contour[idx1], self.gsn_contour[idx2]]

        # Handle Flipping (Normalization)
        self._apply_normalization()

    def _apply_normalization(self):
        """Internal helper to orient the notch consistently."""
        ### step 1 completed: curve segmented
        # origin [0,0]
        # detect if the arc is facing up or down and automatically flip
        # point 1 = piriformis tubercle, point 2 = tip of the ischial spine
        rows, cols = self.processor.shape[:2]
        img_f = self.processor.img.copy()
        curve_f = self.gsn_curve.copy()
        contour_f = self.gsn_contour.copy()
        pts_f = [p.copy() for p in self.points]

        if pts_f[0][1] > pts_f[1][1]:
            # 1. flip around y-axis: cv2.flip(img_blur1, 1)
            img_f = cv2.flip(img_f, 1)
            curve_f[:, 1] = cols - 1 - curve_f[:, 1]
            contour_f[:, 1] = cols - 1 - contour_f[:, 1]
            for p in pts_f: p[1] = cols - 1 - p[1]

        # IF curve facing upwards <=> curve midpoint y coord < curve y endpoints
        mid_idx = len(curve_f) // 2
        if curve_f[mid_idx][0] < pts_f[0][0]:
            img_f = cv2.flip(img_f, 0)
            curve_f[:, 0] = rows - 1 - curve_f[:, 0]
            contour_f[:, 0] = rows - 1 - contour_f[:, 0]
            for p in pts_f: p[0] = rows - 1 - p[0]
        # IF curve facing downwards - OK

        self.flipped_data = {
            "img": img_f, "curve": curve_f,
            "contour": contour_f, "points": pts_f
        }



def transform_gsn(point_piriform, point_ischial, gsn_curve):
    ### Affine Transformation (translation and rotation)
    # origin (pocatek souradnic) [row,column]=[y,x]
    point_origin = [0, 0]
    # calculate translation according to the point_1
    shift = point_origin - point_piriform

    point_ischial_shift = point_ischial + shift

    theta = np.arctan((point_ischial_shift[0]) / (point_ischial_shift[1]))
    M = cv2.getRotationMatrix2D((point_origin), theta * 180 / np.pi, 1)

    # translate points
    shifted_points = gsn_curve + shift
    # rotate points
    M_new = np.array([[M[0][0], -M[0][1]],
                      [M[0][1], M[0][0]]])
    transformed_points = M_new.dot(shifted_points.T).T
    point_ischial = M_new.dot(point_ischial_shift.T).T

    # Scale
    transformed_points = transformed_points / point_ischial[1]

    return transformed_points


class CoordinateStore:
    def __init__(self, img):
        self.points = []
        self.img = img

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            # img = self.img.copy()

            # displaying selected points on the image window
            cv2.circle(self.img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            self.points.append((x, y))
            cv2.imshow('Select two landmarks', self.img)


def get_landmarks(image, contour):
    ### restrict contour
    image_temp = image.copy()
    coordinate_store = CoordinateStore(image_temp)
    # displaying the image
    # gsn_contour = np.array([np.array([x, y]) for y, x in self.gsn_contour.astype(int)])
    # ctr = np.array(self.gsn_contour.astype(int)).reshape((-1, 1, 2)).astype(np.int32)
    # ctr = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
    cv_contour = []
    for point in contour:
        intify = [int(point[1]), int(point[0])]
        cv_contour.append([intify])  # extra pair of brackets because ¯\_(ツ)_/¯ it's OpenCV
        # convert to numpy and draw
    cv_contour = np.array(cv_contour)
    # cv2.drawContours(image_temp, [cv_contour.astype(int)], 0, (0, 0, 255), 2)
    cv2.polylines(image_temp, [cv_contour.astype(int)], isClosed=False, color=(255, 0, 0), thickness=2)
    cv2.imshow('Select two landmarks', image_temp)
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('Select two landmarks', coordinate_store.click_event)
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    # close the window
    cv2.destroyAllWindows()

    # if coordinate_store1.points[0][0] < coordinate_store1.points[1][0]:
    point_1 = [coordinate_store.points[0][1], coordinate_store.points[0][0]]
    point_2 = [coordinate_store.points[1][1], coordinate_store.points[1][0]]

    return [point_1, point_2]
