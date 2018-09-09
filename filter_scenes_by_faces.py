#encoding=utf8

import subprocess
import os
import glob
import json

from PIL import Image

import face_recognition as face


DIR_VIDEOS = 'data/videos/'
DIR_MATCH_PERSONS = 'data/persons/'
DIR_VIDEO_IMAGES = 'data/tmp/images/'
DIR_TIMES = 'data/times/'
DIR_VIDEOS_OUTPUT = 'data/output/'
 
FFMPEG = 'ffmpeg'
MODEL = 'cnn'
UPSAMPLES = [0, 1]
TIME_NEARBY = 3
MERGE_TIME_INTERVAL = 4
# VIDEO_IMAGE_INTERVAL = '1/60'
# VIDEO_IMAGE_INTERVAL = '1/30'
VIDEO_IMAGE_INTERVAL = '1'
# MATCH_RATE = 0.45
MATCH_RATE = 0.44
MATCH_RATE_GEN_CLIP = 0.10
DEBUG = 0


def mkpdir(dir):
    arr = dir.split('/')
    tmp_dir = ''
    
    for i in arr:
        if i:
           tmp_dir += i + '/'
           if not os.path.exists(tmp_dir):
               os.mkdir(tmp_dir)
        else:
            tmp_dir += '/'
            
    return tmp_dir


def empty_dir(dir):
   files = glob.glob(dir + '/*')
   for f in files:
       os.remove(f) 
    

def get_files(dir, exts=None):
    return (os.path.join(dir, f) for f in os.listdir(dir) 
            if os.path.isfile(os.path.join(dir, f))
            and (not exts or os.path.splitext(f)[1] in exts))


def flattern(xs):
    ret = []
    for row in xs:
        for col in row:
            ret.append(col)
    return ret
         
    
def get_persons_enc(persons):
    persons_enc = []
    person_enc_idx = {}
    idx = 0
    
    for filepath in persons:
        image = face.load_image_file(filepath)
        
        person_encs = []
        for upsample in UPSAMPLES:
            faces_loc = face.face_locations(image, number_of_times_to_upsample=upsample, model=MODEL)
            n_face = len(faces_loc)

            if n_face > 1:
                raise Exception('File %s has %s person, please use one person image or cut other person on this image' 
                        % (filepath, n_face))
            elif n_face == 1:
                person_enc_idx[idx] = filepath
                person_encs.append(face.face_encodings(image, known_face_locations=faces_loc)[0])
                idx += 1
            
        if len(person_encs) == 0:
            raise Exception('File %s no person' % filepath)
         
        persons_enc.extend(person_encs)
           
    return persons_enc, person_enc_idx
    
    
def create_images(filename):
    subprocess.run([FFMPEG, '-i', DIR_VIDEOS + filename, 
                    '-vf', 'fps=%s' % VIDEO_IMAGE_INTERVAL, DIR_VIDEO_IMAGES + 'img%05d.jpg'],
                    stdout=open(os.devnull, 'wb'), stderr=subprocess.STDOUT)
    

def exists(pred, xs):
    return len(list(filter(pred, xs))) > 0

def min_pos(xs):
    m = 1
    pos = 0

    for i, v in enumerate(xs):
        if v < m:
            m = v
            pos = i

    return m, pos


def match_persons_time():
    matched_times = []
    persons = list(get_files(DIR_MATCH_PERSONS, exts=['.jpg', '.jpeg', '.png']))

    if len(persons) == 0:
        raise Exception('Persons image format have to be jpg/png')

    persons_enc, person_enc_idx = get_persons_enc(persons)
    video_images = list(get_files(DIR_VIDEO_IMAGES, exts=['.jpg']))

    if len(video_images) == 0:
        raise Exception('Video image format have to be jpg')

    # sort by time in video 
    video_images.sort()
   
    for i, filepath in enumerate(video_images):
        time = i+1
        image = face.load_image_file(filepath)
        faces_loc = flattern([face.face_locations(image, number_of_times_to_upsample=u, model=MODEL)
                             for u in UPSAMPLES])
        
        if len(faces_loc) == 0:
            # print('Time %ss no person' % time)
            continue
            
        faces_enc = face.face_encodings(image, known_face_locations=faces_loc)
        
        for face_enc in faces_enc:
            dists = face.face_distance(persons_enc, face_enc)

            if exists(lambda x: x<MATCH_RATE, dists):
                # auto gen targets from predict
                if len(persons_enc) < 300 and exists(lambda x: x<MATCH_RATE - MATCH_RATE_GEN_CLIP, dists):
                    persons_enc.append(face_enc)
                
                # TODO: maybe use ffmpeg api to check if frames nearby have many actions, can't get face, so we should enlarge time nearby
                matched_times.append((time-TIME_NEARBY, time+TIME_NEARBY))

                min_dist, dist_idx = min_pos(dists)
                print('Matched person in %ss %s, dist: %s, person: %s' 
                        % (time, filepath, min_dist, person_enc_idx[dist_idx]))

                break
            
    return matched_times
   

def merge_persons_time(matched_times):
    merged_times = []
    tlen = len(matched_times)
    skip_next = False
    
    for i, (start, end) in enumerate(matched_times):
        if skip_next:
            skip_next = False
            continue
            
        if start < 0:
            start = 0
        
        if i < tlen - 1:
            next_start, next_end = matched_times[i+1]
            if next_start - end <= MERGE_TIME_INTERVAL:
                end = next_end
                skip_next = True
                
        merged_times.append((start, end))
        
    return merged_times
        
    
def create_video_cut_times(filename):
    if not DEBUG and len(os.listdir(DIR_VIDEO_IMAGES)) < 10:
        print('Creating images...')
        create_images(filename)
    
    print('Matching persons...')
    matched_times = match_persons_time()

    print('Merge times...')
    merged_times = merge_persons_time(matched_times)

    for i in range(20):
        merged_times = merge_persons_time(merged_times)
    
    print(merged_times)
    with open(DIR_TIMES + filename + '.txt', 'w') as f:
        f.write(json.dumps(merged_times))

        
def extract_video(filename):
    times = []
    with open(DIR_TIMES + filename + '.txt', 'r') as f:
        times = json.loads(f.read())
        
    if len(times) == 0:
        raise Exception('File %s no times' % filename)
        
    print(times)
    
    v_selects = []
    a_selects = []
    
    for start, end in times:
        v_selects.append('between(t,%s,%s)' % (start, end))
        a_selects.append('between(t,%s,%s)' % (start, end))
        
    output = DIR_VIDEOS_OUTPUT + filename + '.extracted.mp4'
    subprocess.run([FFMPEG, '-i', DIR_VIDEOS + filename, '-vf',
                    "select='%s',setpts=N/FRAME_RATE/TB" % '+'.join(v_selects), '-af',
                    "aselect='%s',asetpts=N/SR/TB" % '+'.join(a_selects), output],
                    stdout=open(os.devnull, 'wb'), stderr=subprocess.STDOUT)   
    
        
def filter_scenes_by_faces():
    if not os.path.exists(DIR_MATCH_PERSONS):
        raise Exception('Person path and files %s did not exists' % DIR_MATCH_PERSONS)

    if not os.path.exists(DIR_VIDEO_IMAGES):
        mkpdir(DIR_VIDEO_IMAGES)
        
    if not os.path.exists(DIR_TIMES):
        mkpdir(DIR_TIMES)

    if not os.path.exists(DIR_VIDEOS_OUTPUT):
        mkpdir(DIR_VIDEOS_OUTPUT)
 
    for filepath in get_files(DIR_VIDEOS):
        print('Processing %s...' % filepath)

        if not DEBUG:
            empty_dir(DIR_VIDEO_IMAGES)
        
        filename = os.path.basename(filepath)

        if filename == '.gitkeep': 
            continue

        create_video_cut_times(filename)
        extract_video(filename)
        

if __name__ == '__main__':
    filter_scenes_by_faces()
