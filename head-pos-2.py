import cv2
import dlib
import numpy as np
from imutils import face_utils


class HeadPose():
    """Estimation of the angles of Euler"""
    
    def __init__(self):
        self.face_landmark_path = 'shape_predictor_68_face_landmarks.dat'
        
        
    def get_point(self):
        """Get 2D coordonates, 3D correspondancy and camera parameters"""
        
        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
             0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
             0.0, 0.0, 1.0]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        self.cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                 [1.330353, 7.122144, 6.903745],
                                 [-1.330353, 7.122144, 6.903745],
                                 [-6.825897, 6.760612, 4.402142],
                                 [5.311432, 5.485328, 3.987654],
                                 [1.789930, 5.393625, 4.413414],
                                 [-1.789930, 5.393625, 4.413414],
                                 [-5.311432, 5.485328, 3.987654],
                                 [2.005628, 1.409845, 6.165652],
                                 [-2.005628, 1.409845, 6.165652],
                                 [2.774015, -2.080775, 5.048531],
                                 [-2.774015, -2.080775, 5.048531],
                                 [0.000000, -3.116408, 6.097667],
                                 [0.000000, -7.415691, 4.070434]])

        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                   [10.0, 10.0, -10.0],
                                   [10.0, -10.0, -10.0],
                                   [10.0, -10.0, 10.0],
                                   [-10.0, 10.0, 10.0],
                                   [-10.0, 10.0, -10.0],
                                   [-10.0, -10.0, -10.0],
                                   [-10.0, -10.0, 10.0]])

        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        
    def get_pose(self,shape):
        """Get pose estimation from 2D image"""
        self.image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        
            #function for pose estimation.
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, self.image_pts,
                                                        self.cam_matrix,self.dist_coeffs)
        
        #Camera Calibration and 3D Reconstruction
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec,
                                            self.cam_matrix,self.dist_coeffs)
        
        

        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

        #converts rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        
        #hconcat takes in a list of images, and concatenates them horizontally
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        
        return reprojectdst, euler_angle
        
    def run_pose_estimation(self):
        """Run camera capture"""
        
        self.get_point()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to connect to camera.")
            return
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.face_landmark_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                face_rects = detector(frame, 0)

                if len(face_rects) > 0:
                    shape = predictor(frame, face_rects[0])
                    shape = face_utils.shape_to_np(shape)

                    reprojectdst, euler_angle = self.get_pose(shape)

                    for (x, y) in shape:
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                    for start, end in self.line_pairs:
                        

                        cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)
                        cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
                        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), thickness=2)

                cv2.imshow("demo", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                
                
                
                
#driver-func
head_pose_estimator = HeadPose()
head_pose_estimator.run_pose_estimation()