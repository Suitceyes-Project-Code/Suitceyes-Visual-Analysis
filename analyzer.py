import numpy as np
import os
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import facenet
from data import *
import json
import pickle

sess = tf.Session()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# You can download it here:  http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.tar.gz
_OD_PATH_TO_CKPT = './models/oid_objects/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12.pb'
# List of the strings that is used to add correct label for each box.
_OD_PATH_TO_LABELS = './models/oid_objects/oid_v4_label_map.pbtxt.txt'
#_OD_EVERYTHING_CLASS_POOL = [each for each in range(_OD_NUM_CLASSES)] #Show every class
#_OD_SE_DEMO_CLASS_POOL = list(se_demo_ci.keys())
_OD_NUM_CLASSES = 604 #for category index range of colors

_OD_CATEGORY_INDEX = None

with open('./models/oid_objects/se_oid_ci.pkl', mode='rb') as file:
    se_oid_ci = pickle.load(file)

_FD_CLASS_LIST = svm_face_list

print("Importing ObD module...")
with tf.variable_scope('rcnn_graph'):
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(_OD_PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print("Done")

###############################################################################
#####|___________---FD---___________---FD---___________---FD---___________|####
###############################################################################

print("Importing FD module...")

# Load the facenet model
#You can download it here: https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view
PATH_TO_PRETRAINED_MODEL = ''
with tf.variable_scope('facenet_graph'):
    facenet.load_model(sess,PATH_TO_PRETRAINED_MODEL)

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("facenet_graph/input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("facenet_graph/embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("facenet_graph/phase_train:0")
embedding_size = embeddings.get_shape()[1]

classifier_filename_exp = './models/lfw_classifier_demo.pkl'

with open(classifier_filename_exp, 'rb') as infile:
    (model, _) = pickle.load(infile)

print("Done")


g = tf.get_default_graph()

# delete every previous file in the log folder
for log_f in os.listdir('./logs/tf'):
    os.remove(os.path.join('./logs/tf', log_f))

graph_logger = tf.summary.FileWriter('./logs/tf')
graph_logger.add_graph(sess.graph)

print('Analyzer imported')

def box_trans_1(box, width, height): #transform from (ymin, xmin, ymax, xmax)[norm] to (left, top, width, height)[non-norm]
    box = [box[0] * height, box[1] * width, box[2] * height, box[3] * width]
    box = list(map(int, box))
    box = [box[1], box[0], abs(box[3]-box[1]), abs(box[2]-box[0])]
    return box

def box_trans_2(box, height, width): #transform from (left, top, right, bottom)[non-norm] to (ymin, xmin, ymax, xmax)[norm]
    xmin, ymin, xmax, ymax = box
    new_box = [ymin/height, xmin/width, ymax/height, xmax/width]
    return new_box

def rev_box_trans(box_list): #transform from (left, top, width, height)[non-norm] to (ymin, xmin, ymax, xmax)[non-norm]
    res = []
    for box in box_list:
        temp = [box[1], box[0], box[3]+box[1], box[2]+box[0]]
        res = res + [temp]
    return res

def to_norm(box_list, height, width): #transform from (left, top, width, height)[non-norm] to (ymin, xmin, ymax, xmax)[norm]
    res = []
    for box in box_list:
        temp = [box[1]/height, box[0]/width, (box[3]+box[1])/height, (box[2]+box[0])/width]
        res = res + [temp]
    return res

def to_non_norm(box_list, height, width): #transform from (ymin, xmin, ymax, xmax)[norm] to (ymin, xmin, ymax, xmax)[non-norm]
    res = []
    for box in box_list:
        temp = [box[0]*height, box[1]*width, box[2]*height, box[3]*width]
        temp = list(map(int, temp))
        res += [temp]
    return res

def reject_outliers(data, m = 2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def calculate_object_distance(depth_box):
    non_zero_data = depth_box[depth_box.nonzero()]
    non_zero_fraction = non_zero_data.shape[0]/(depth_box.shape[0]*depth_box.shape[1])
    if non_zero_fraction > 0.01:
        #clean_data = reject_outliers(non_zero_data)
        clean_data = non_zero_data
        distance = clean_data.mean() #can also go with median
    else:
        distance = -1
    return distance

def process_image(image_np,
                  depth_reg,
                  file_name, 
                  timestamp, 
                  confidence_drop = 0.35, 
                  upper_size_thres = 1, 
                  face_confidence = 0.2,
                  class_pool = 'se_demo',
                  flags = []):
    
    global _OD_CATEGORY_INDEX #modify the global parameter to be 'seen' by visualize_results function
    
    #Select class pool    
    if class_pool == 'se_oid':
        _OD_CATEGORY_INDEX = se_oid_ci
    else:
        _OD_CATEGORY_INDEX = se_demo_ci
        
    class_pool = list(_OD_CATEGORY_INDEX.keys())
        
    inverse_category_index = {}
    for key in _OD_CATEGORY_INDEX.keys():
        inverse_category_index[_OD_CATEGORY_INDEX[key]['name']] = key

    height = image_np.shape[0]
    width = image_np.shape[1]
    
    
    #OBD INFERENCE    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    frame_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = g.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = g.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = g.get_tensor_by_name('detection_scores:0')
    classes = g.get_tensor_by_name('detection_classes:0')
    num_detections = g.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_np_expanded})
    to_del=()
    #delete unwanted boxes (too large, unwanted labels, etc)
    for i in range(boxes.shape[1]):
        if((np.abs(boxes[0][i][0]-boxes[0][i][2])*np.abs(boxes[0][i][1]-boxes[0][i][3]) >= upper_size_thres) or 
            (scores[0][i] < confidence_drop) or
            (classes[0][i] not in class_pool)):
            to_del = to_del + (i,)
            num_detections[0] = num_detections[0] - 1
    boxes = np.delete(boxes, to_del, axis=1)
    scores = np.delete(scores, to_del, axis=1)
    classes = np.delete(classes, to_del, axis=1)
    
    boxes = np.squeeze(boxes, axis=0)
    classes = np.squeeze(classes, axis=0).astype(np.int32)
    scores = np.squeeze(scores, axis=0)
    
    boxes_nonnorm = np.array(to_non_norm(boxes, height, width)) # (ymin, xmin, ymax, xmax)[non-norm]
    
    #Face recognition
    face_idx = [each_idx for each_idx, each in enumerate(classes) if each==inverse_category_index['face']]
    nrof_faces = len(face_idx)
    if nrof_faces > 0:
        face_boxes = boxes_nonnorm[face_idx]
        face_scores = scores[face_idx]
        face_classes = classes[face_idx]
        aligned_array = np.empty((nrof_faces, 160, 160, 3))
        img_size = (height, width)
        for i, det in enumerate(face_boxes):
            bb = np.zeros(4, dtype=np.int32)
            bb[1] = np.maximum(det[1]-32/2, 0) # xmin / left
            bb[0] = np.maximum(det[0]-32/2, 0) # ymin / top
            bb[3] = np.minimum(det[3]+32/2, img_size[1]) #xmax / right
            bb[2] = np.minimum(det[2]+32/2, img_size[0]) #ymax / bottom
            cropped = image_np[bb[0]:bb[2],bb[1]:bb[3],:]
            scaled = cv2.resize(cropped, (160,160))
            aligned_array[i,:,:,:] = scaled
            
            # Run forward pass to calculate embeddings
            emb_array = np.zeros((nrof_faces, embedding_size))
            whit_images = np.zeros_like(aligned_array)
            for idx, img in enumerate(aligned_array):
                whit_images[idx] = facenet.prewhiten(img)
            feed_dict = {images_placeholder:whit_images, phase_train_placeholder:False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)

        # Classify faces
        face_predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(face_predictions, axis=1)
        best_class_probabilities = face_predictions[np.arange(len(best_class_indices)), best_class_indices]
        
        for i in range(len(best_class_indices)):
            if best_class_probabilities[i] >= face_confidence:
                known_face_name = _FD_CLASS_LIST[best_class_indices[i]]
                try:
                    face_classes[i] = inverse_category_index[known_face_name]
                    face_scores[i] = best_class_probabilities[i]
                except:
                    #in case a high score face is detected that is not in the general category index this exception is thrown
                    face_classes[i] = 0 #this is fixed for person_unkown
                
            else:
                face_classes[i] = 0 #this is fixed for person_unkown
                #face_scores[i] doesn't change, it takes the score of the object detector
        
        #merge results            
        scores[face_idx] = face_scores
        classes[face_idx] = face_classes
        print(classes)
        
    #calculate object distances from depth_reg
    distances = [-1 for i in range(len(boxes_nonnorm))]
    if depth_reg is not None:
        for idb, box in enumerate(boxes_nonnorm): # (ymin, xmin, ymax, xmax)[non-norm]
            depth_crop = depth_reg[box[0]:box[2], box[1]:box[3]]
            distances[idb] = calculate_object_distance(depth_crop)
        
    #output results
    target_dict = []
    for box, conf, cl_idx, dist in zip(boxes, scores, classes, distances):
        box_t = box_trans_1(box, width, height)
        target_dict += [{"left":box_t[0], 
                        "top":box_t[1], 
                        "width":box_t[2], 
                        "height":box_t[3],
                        "type":_OD_CATEGORY_INDEX[cl_idx]['name'],
                        "distance": dist,
                        "confidence":float("{0:.3f}".format(conf))}]
    son = {'image':{"name":file_name,
                    "width":width, 
                    "height":height, 
                    "timestamp":timestamp.strftime('%Y-%m-%dT%H:%M:%S.%f')[0:-3],
                    "scene_type":"scene_unknown", 
                    "scene_score":float("{0:.3f}".format(0)),
                    "target":target_dict}}
    
    if "save_data_json_locally" in flags:
        with open(os.path.join('.', 'output', 'output_'+file_name+'.json'), 'w') as outfile:
            json.dump(son, outfile, sort_keys=False, indent=1)
        
    result_dict = {}
    result_dict['boxes'] = boxes
    result_dict['scores'] = scores
    result_dict['classes'] = classes
    result_dict['distances'] = distances
        
    return result_dict, son

def visualize_results(write_on, file_name, result_dict):
    global _OD_CATEGORY_INDEX
    
    boxes = result_dict['boxes']
    classes = result_dict['classes']
    scores = result_dict['scores']
    distances = result_dict['distances']
    
    height = write_on.shape[0]
    width = write_on.shape[1]
    
    boxes_nonnorm = np.array(to_non_norm(boxes, height, width)) # (ymin, xmin, ymax, xmax)[non-norm]
    
    #Box Blurring
    #for box, cl in zip(boxes, classes): # blur boxes
        #if cl==1: #check if person
        #box = box_trans_1(box, height, width)
        #cropped_img = write_on[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        #blurred_box = cv2.GaussianBlur(cropped_img, (0, 0), 10)
        #write_on[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = blurred_box

    #Visualize object detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        write_on,
        boxes,
        classes,
        scores,
        _OD_CATEGORY_INDEX,
        use_normalized_coordinates=True,
        max_boxes_to_draw=500,
        line_thickness=4,
        min_score_thresh=0.01)
    
    #Print distances
    for box, d in zip(boxes_nonnorm, distances):

        text = "distance:{0:.3f}".format(d/1000)
        font  = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.45
        rectangle_bgr = (0, 0, 0)

        # get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

        # set the text start position, use left corner, if left is beyond half width then use right corner
        if box[1] < width/2:
            text_offset_x = box[1]
            text_offset_y = box[2]
        else:
            text_offset_x = box[3] - text_width
            text_offset_y = box[2]

        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
        cv2.rectangle(write_on, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(write_on, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
    
    cv2.imwrite(os.path.join('.', 'output', 'output_'+file_name+'.jpg'), write_on)
    
