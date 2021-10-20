import math

import cv2
import numpy as np


def toRad(angle):
    return angle * math.pi / 180


class Warper:
    """
    Warp the image in left/right or up/down direction.
    """
    def __init__(self, yaw_angle, pitch_angle, imz_sz=224):
        self._sz = imz_sz
        roll_angle = 0
        rotation_vector = np.array([toRad(-1 * pitch_angle), toRad(yaw_angle), toRad(roll_angle)], dtype=np.float32)
        self._R = cv2.Rodrigues(rotation_vector)[0]

        T1 = np.matrix([[1., 0., 0. - self._sz // 2], [0., 1., 0. - self._sz // 2], [0., 0., 1.]])
        T2 = np.matrix([[1., 0., self._sz // 2], [0., 1., self._sz // 2], [0., 0., 1.]])
        self._W = T2 * self._R * T1
        self.boundaries = self.get_boundaries()
        print(f'[{self.__class__.__name__}] Yaw:{yaw_angle} Pitch:{pitch_angle} Boundaries:{self.boundaries}')
        # remove the zero component segment.

    def get_boundaries(self):
        width = height = self._sz
        pnts = np.asarray([
            [0, 0],
            [0, height],
            [width, 0],
            [width, height],
        ], dtype=np.float32)

        # check: http://answers.opencv.org/question/252/cv2perspectivetransform-with-python/
        pnts = np.array([pnts])

        dst_pnts = cv2.perspectiveTransform(pnts, self._W)[0]
        final_boundaries = np.asarray(dst_pnts, dtype=np.float32)
        # print(final_boundaries)
        sx = max(0, max(final_boundaries[[0, 1], 0]))
        sy = max(0, max(final_boundaries[[0, 2], 1]))
        ex = min(width, min(final_boundaries[[2, 3], 0]))
        ey = min(height, min(final_boundaries[[1, 3], 1]))
        return np.array([sx, sy, ex, ey]).astype(np.int32)

    def transform(self, img):
        """
        < 1 ms time
        """
        transf_img = cv2.warpPerspective(img, self._W, (self._sz, self._sz))
        return transf_img


class WarperWithCameraMatrixFull:
    """
    Warp the image in left/right or up/down direction.
    """
    def __init__(self, camera_matrix, camera_face_distance, yaw_angle, pitch_angle, imz_sz=224):
        self._sz = imz_sz
        self._C = camera_matrix
        self._dis = camera_face_distance
        roll_angle = 0
        D = np.diagflat([self._dis] * 3)

        rotation_vector = np.array([toRad(-1 * pitch_angle), toRad(yaw_angle), toRad(roll_angle)], dtype=np.float32)
        self._R = cv2.Rodrigues(rotation_vector)[0]

        self._W = self._C @ self._R @ np.linalg.inv(self._C) @ D
        self._W = self.cancel_lateral_shift() * self._W
        self.boundaries = self.get_boundaries()
        print(f'[{self.__class__.__name__}] Yaw:{yaw_angle} Pitch:{pitch_angle} Boundaries:{self.boundaries}')
        # remove the zero component segment.

    def cancel_lateral_shift(self):
        pnts = np.asarray([
            [112, 112],
        ], dtype=np.float32)

        pnts = np.array([pnts])

        dst_pnts = cv2.perspectiveTransform(pnts, self._W)[0]
        # print(dst_pnts.shape, pnts.shape)
        final_boundaries = np.asarray(dst_pnts - pnts[0], dtype=np.float32)
        bbox = final_boundaries[0]
        T2 = np.matrix([[1., 0., 0 - bbox[0]], [0., 1., 0 - bbox[1]], [0., 0., 1.]])
        return T2

    def get_boundaries(self):
        width = height = self._sz
        pnts = np.asarray([
            [0, 0],
            [0, height],
            [width, 0],
            [width, height],
        ], dtype=np.float32)

        # check: http://answers.opencv.org/question/252/cv2perspectivetransform-with-python/
        pnts = np.array([pnts])

        dst_pnts = cv2.perspectiveTransform(pnts, self._W)[0]
        final_boundaries = np.asarray(dst_pnts, dtype=np.float32)
        # print(final_boundaries.astype(np.int))
        sx = max(0, max(final_boundaries[[0, 1], 0]))
        sy = max(0, max(final_boundaries[[0, 2], 1]))
        ex = min(width, min(final_boundaries[[2, 3], 0]))
        ey = min(height, min(final_boundaries[[1, 3], 1]))
        return np.array([sx, sy, ex, ey]).astype(np.int32)

    def transform(self, img):
        """
        < 1 ms time
        """
        transf_img = cv2.warpPerspective(img, self._W, (self._sz, self._sz))
        return transf_img
