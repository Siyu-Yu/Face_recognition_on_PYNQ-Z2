import cv2
import os
import numpy as np
import mtcnn
import utils as utils
import InceptionResNetV1

class face_rec():
    def __init__(self):
        # create mtcnn object
        # detect face
        self.mtcnn_model = mtcnn()
        # threshold
        self.threshold = [0.5,0.8,0.9]

        # load facenet
        # detected face transfer to 128-dimensional vector
        self.facenet_model = InceptionResNetV1()
        # model.summary()
        model_path = './model_data/facenet_keras.h5'
        self.facenet_model.load_weights(model_path)

        #-----------------------------------------------#
        #   encode the face in dataset
        #   known_face_encodings: face after encoding
        #   known_face_names: name of face
        #-----------------------------------------------#
        face_list = os.listdir("face_dataset")

        self.known_face_encodings=[]

        self.known_face_names=[]

        for face in face_list:
            name = face.split(".")[0]

            img = cv2.imread("./face_dataset/"+face)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            # detect face
            rectangles = self.mtcnn_model.detectFace(img, self.threshold)

            # convert to rectangle
            rectangles = utils.rect2square(np.array(rectangles))
            # facenet input a 160*160 picture
            rectangle = rectangles[0]
            # mark the landmark
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

            crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))

            new_img,_ = utils.Alignment_1(crop_img,landmark)

            new_img = np.expand_dims(new_img,0)
            # transfer the detected face into facenet, realize the 128-dimensional feature vector extraction
            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)

    def recognize(self,draw):
        #-----------------------------------------------#
        #   Face recognition
        #   Positioning first, then database matching
        #-----------------------------------------------#
        height,width,_ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

        # detect human face
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)

        if len(rectangles)==0:
            return

        # convert to rectangle
        rectangles = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        rectangles[:,3] = np.clip(rectangles[:,3],0,height)
        #-----------------------------------------------#
        #   Encode the detected face
        #-----------------------------------------------#
        face_encodings = []
        for rectangle in rectangles:
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160

            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv2.resize(crop_img,(160,160))

            new_img,_ = utils.Alignment_1(crop_img,landmark)
            new_img = np.expand_dims(new_img,0)

            face_encoding = utils.calc_128_vec(self.facenet_model,new_img)
            face_encodings.append(face_encoding)

        face_names = []
        for face_encoding in face_encodings:
            # Take out a face and compare it with all faces in the database to calculate the score
            matches = utils.compare_faces(self.known_face_encodings, face_encoding, tolerance = 0.9)
            name = "Unknown"
            # Find the closest face
            face_distances = utils.face_distance(self.known_face_encodings, face_encoding)
            # Take out the score of this recent face
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        rectangles = rectangles[:,0:4]
        #-----------------------------------------------#
        #   
        #-----------------------------------------------#
        for (left, top, right, bottom), name in zip(rectangles, face_names):
            cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw, name, (left , bottom - 15), font, 0.75, (255, 255, 255), 2) 
        return draw

if __name__ == "__main__":

    dududu = face_rec()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, draw = video_capture.read()
        dududu.recognize(draw) 
        cv2.imshow('Video', draw)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()