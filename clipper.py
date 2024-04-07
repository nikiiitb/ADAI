from os import listdir , rename 
from os.path import isfile, join , splitext
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import xml.etree.ElementTree as ET
import re
# ffmpeg_extract_subclip("SSBD/videos/v_ArmFlapping_01.avi", 18, 24, targetname="test.mp4")

annot_location = "./SSBD/annotations/" #the annotations per video
vid_location = "./inputs/full" #the preprocessed videos
out_location = "./inputs/clipped"

vid_names = [f for f in listdir(vid_location) if isfile(join(vid_location, f))]
vids = [join(vid_location, f) for f in vid_names]
annots = [join(annot_location, splitext(f)[0] + ".xml") for f in vid_names]

categories = {}

for vid , annot in zip(vids , annots):
    tree = ET.parse(annot)
    root = tree.getroot()
    print(vid , annot)
    for behaviour in root.iter('behaviour'):
        t , cat = behaviour.find('time').text , behaviour.find('category').text
        start , end = list(map(int , re.split(':|-' , t)))
        ind = categories.get(cat , 0)
        out = join(out_location , cat , str(ind) + ".avi")

        ffmpeg_extract_subclip(vid, start, end, targetname=out)

        categories[cat] = ind + 1
        print(f"{annot} CONVERTED TO {out}")

print("="*10,"DONE","="*10)