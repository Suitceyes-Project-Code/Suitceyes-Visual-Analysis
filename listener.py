from realtime import ortc
import json
import numpy as np
import requests
import time
import uuid
import os
from threading import Lock, Thread
import datetime
import traceback
import cv2
import analyzer
from tqdm import tqdm

# The url of the location that the images are uploaded and stored
UPLOAD_URL = ''
STORAGE_URL = ''

storage_path = ''

frame_count = 0
mean_analysis_interval = 0.7
analysis_interval_sum = 0
threads_on = 0

T_queue = []

application_key = 'c2dNjc'
ortc_client = ortc.OrtcClient()
ortc_client.cluster_url = "https://ortc-developers.realtime.co/server/ssl/2.1"

#get lock object
lock = Lock()

def rt_on_exception(sender, exception):
    print ('ORTC Exception: '+exception)

def rt_on_connected(sender):
    print ('ORTC Connected')

def rt_on_disconnected(sender):
    print ('ORTC Disconnected')
    import _thread
    _thread.interrupt_main()
        
def rt_on_subscribed(sender, channel):
    print ('ORTC Subscribed to: '+channel)
    
def rt_on_unsubscribed(sender, channel):
    print ('ORTC Unsubscribed from: '+channel)

def rt_on_reconnecting(sender):
    print ('ORTC Reconnecting')

def rt_on_reconnected(sender):
    print ('ORTC Reconnected')
    
def upload_to_storage(file):
    file_name = file.split(sep='/')[-1]
    try:
        data = {'file': open(file, 'rb')}
    except Exception as err:
        print('Error in opening file.')
        return -1
    try:
        r = requests.post(UPLOAD_URL, files=data)
        if r.text == 'UPLOAD OK':
            return STORAGE_URL+file_name
        else:
            print('Upload Error: "{}"'.format(r.text))
            return -1
    except requests.exceptions.ConnectionError as err:
        print('Connection error')
        return -1

def read_local_image(image_path):
    img_np_bgr = cv2.imread(image_path)
    img_np_rgb = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
    return img_np_bgr, img_np_rgb

def read_depth_reg(image_path):
    depth_reg = cv2.imread(image_path, -1)
    return depth_reg

def rotate_image_90(image_np):
    return np.rot90(image_np)

def rotate_image_180(image_np):
    return rotate_image_90(rotate_image_90(image_np))

def send_results_to_KB(json_data):
    message_id = "VA_"+uuid.uuid4().hex
    data_json_filename = message_id+'.json'
    with open(os.path.join(storage_path, data_json_filename), mode='w') as file:
        json.dump(json_data, file)
    link = STORAGE_URL+data_json_filename

    topic_va_kb = {'header':{'sender':'VA', 'recipients':['KBS'], 'timestamp':'', 'message_id':''}, 'body':{'data':''}}
    topic_va_kb['header']['timestamp'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[0:-3]
    topic_va_kb['header']['message_id'] = message_id
    topic_va_kb['body']['data'] = link

    ortc_client.send(channel='VA_KBS_channel', message=json.dumps(topic_va_kb))
    print('\tSending message with id:{} to VA_KBS_channel'.format(message_id))

def rt_on_message(sender, channel, message):
    global T_queue
    # make new thread and push to FIFO queue.
    T_queue += [Thread(target=message_handler, args=(message, len([each for each in T_queue if each.is_alive()])))]
    T_queue[-1].start()
    print('\nNew message is being processed by thread {}.'.format(T_queue[-1].ident))
    
def message_handler(message, num_of_waiting_threads):
    global mean_analysis_interval
    global analysis_interval_sum
    global frame_count
    global threads_on
    
    threads_on += 1
    
    capture_timestamp_str = message.split(sep='-')[0]
    try:
        capture_timestamp = datetime.datetime.strptime(capture_timestamp_str, "%d_%m_%y_%H_%M_%S_%f")
    except:
        print("Invalid request, skipping.")
        threads_on -= 1
        return -1
    
    upload_interval = datetime.datetime.utcnow() - capture_timestamp
    upload_interval = upload_interval.total_seconds()
    processing_time_estimator = upload_interval+threads_on*mean_analysis_interval
    
    
    #lock here
    lock.acquire()
    start = time.time()
    
    message_flags = message.split(sep='-')[1:]
    rgb_filename = 'rgb_'+capture_timestamp_str+'.jpg'
    depth_reg_filename = 'depth_reg_'+capture_timestamp_str+'.png'
    
    try:
        image_np_bgr, image_np_rgb = read_local_image(os.path.join(storage_path, rgb_filename))
    except:
        print("Could not read images, skipping.")
        lock.release()
        threads_on -= 1
        return -1
    
    # possible flags:
        # -nodepth: there is no depth provided
        # -conf=x: confidence drop is x
        # -cl=se_demo, or -cl=se_oid : choose class pool
    
    if 'nodepth' in message_flags:
        depth_reg = None
    else:
        try:
            #rotate rgb
            #image_np_bgr = rotate_image_180(image_np_bgr)
            #image_np_rgb = rotate_image_180(image_np_rgb)
            
            #read depth_reg
            depth_reg = read_depth_reg(os.path.join(storage_path, depth_reg_filename))
            
            #rotate depth reg
            #depth_reg = rotate_image_180(depth_reg)
            
            #check the shapes
            print(image_np_rgb.shape, depth_reg.shape)
        except:
            print("Could not read depth image, skipping that.")
            #lock.release()
            #return -1
            depth_reg = None
        
    try:
        conf_t = float([each for each in message_flags if 'conf' in each][0].split(sep='=')[1])
    except:
        conf_t = 0.5
        
    try:
        cl_p = [each for each in message_flags if 'cl' in each][0].split(sep='=')[1]
    except:
        cl_p = 'se_demo'

    #process here and send the results
    #print('image size is:{}x{}'.format(image_np_rgb.shape[1], image_np_rgb.shape[0]))
    analysis_flags = ["save_data_json_locally"]
    result_dict, json_data = analyzer.process_image(image_np_rgb, depth_reg,
                                                    confidence_drop = conf_t,
                                                    class_pool = cl_p,
                                                    file_name=capture_timestamp_str,
                                                    timestamp=capture_timestamp,
                                                    flags=analysis_flags)
    end = time.time()
    frame_count += 1
    analysis_interval = end-start
    analysis_interval_sum += analysis_interval
    mean_analysis_interval = analysis_interval_sum/frame_count
    full_process_interval = datetime.datetime.utcnow() - capture_timestamp
    full_process_interval = full_process_interval.total_seconds()
    print('\timage upload and processing estimated {} seconds'.format(processing_time_estimator))
    print('\timage upload and processing took {} seconds'.format(full_process_interval))
    print('\timage upload took {} seconds'.format(upload_interval))
    print('\timage processing took {} seconds'.format(analysis_interval))
    print('\tsum of upload and processing is {} seconds'.format(upload_interval+analysis_interval))
    print('\tactive threads are {}. mean analysis time is {}'.format(threads_on, mean_analysis_interval))
    fps = frame_count/analysis_interval_sum
    print('VA runs at {} fps'.format(fps))
    lock.release()
    
    send_results_to_KB(json_data)
    analyzer.visualize_results(image_np_bgr, capture_timestamp_str, result_dict)
    
    threads_on -= 1
    return 0

ortc_client.set_on_exception_callback(rt_on_exception)
ortc_client.set_on_connected_callback(rt_on_connected)
ortc_client.set_on_disconnected_callback(rt_on_disconnected)
ortc_client.set_on_subscribed_callback(rt_on_subscribed)
ortc_client.set_on_unsubscribed_callback(rt_on_unsubscribed)
ortc_client.set_on_reconnecting_callback(rt_on_reconnecting)
ortc_client.set_on_reconnected_callback(rt_on_reconnected)

ortc_client.connect(application_key)

print('Performing a test analysis...')
#for i in tqdm(range(100)):

#set path to a local image for a test analysis
PATH_TO_TEST_IMAGE = ''
image_np_bgr, image_np_rgb = read_local_image(PATH_TO_TEST_IMAGE)
capture_timestamp = datetime.datetime.utcnow()
capture_timestamp_str = capture_timestamp.strftime("%d_%m_%y_%H_%M_%S_%f")
analyzer.process_image(image_np_rgb, None,
                       file_name=capture_timestamp_str+'_va-controler-test', 
                       timestamp=capture_timestamp)
print('Done.')

ortc_client.subscribe('VA_RS_channel', True, rt_on_message)

while True:
    #listen for incoming messages
    continue

#image_listener_thread = Thread(target=image_main)
#image_listener_thread.start()
