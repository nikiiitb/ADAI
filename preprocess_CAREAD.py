import cv2
import numpy as np

from ultralytics import YOLO
from sort_master.sort import Sort

from os import listdir
from os.path import isfile, join

model = YOLO("yolov8n.pt")
detection_threshold = 0.60
OUTPUT_SHAPE = (224,224)

tracker = Sort()
# colors = np.random.randint(0 , 256 , size=(10,3))

vid_location = "./SSBD/videos/"
out_location = "./inputs/full"

vid_names = [f for f in listdir(vid_location) if isfile(join(vid_location, f))]
vids = [join(vid_location, f) for f in vid_names]
vid_outs = [join(out_location, f) for f in vid_names]

# vids = [
#     # "./SSBD/videos/v_ArmFlapping_01.avi",
#     # "./SSBD/videos/v_ArmFlapping_02.avi",
#     # "./SSBD/videos/v_ArmFlapping_03.avi",
#     # "./SSBD/videos/v_HeadBanging_01.avi",
#     # "./SSBD/videos/v_HeadBanging_02.avi",
#     # "./SSBD/videos/v_HeadBanging_03.avi",
#     # "./SSBD/videos/v_Spinning_01.avi",
#     # "./SSBD/videos/v_Spinning_02.avi",
#     "./SSBD/videos/v_Spinning_04.avi",
# ]

# vid_outs = [
#     # "./inputs/full/v_ArmFlapping_01.avi",
#     # "./inputs/full/v_ArmFlapping_02.avi",
#     # "./inputs/full/v_ArmFlapping_03.avi",
#     # "./inputs/full/v_HeadBanging_01.avi",
#     # "./inputs/full/v_HeadBanging_02.avi",
#     # "./inputs/full/v_HeadBanging_03.avi",
#     # "./inputs/full/v_Spinning_01.avi",
#     # "./inputs/full/v_Spinning_02.avi",
#     "./inputs/full/v_Spinning_04.avi",
# ]

for video , out in zip(vids , vid_outs):
    frames = []
    video = cv2.VideoCapture(video)
    while True:
        read, frame= video.read()
        if not read:
            break
        outp = frame
        minx1 , miny1 , maxx2 , maxy2 = float("inf"),float("inf"),0,0
        results = model(frame)
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score , class_id])
            detections = np.array(detections)
            track_bbs_ids = tracker.update(detections)


            for track in track_bbs_ids:
                # bbox = track.bbox
                # x1, y1, x2, y2 = bbox
                # track_id = track.track_id
                x1, y1, x2, y2, id = list(map(int , track.tolist()))
                minx1 = min(minx1 , x1)
                miny1 = min(miny1 , y1)
                maxx2 = max(maxx2 , x2)
                maxy2 = max(maxy2 , y2)
                name = f"ID : {id}"
                # color = (255,0,0)
                # cv2.rectangle(outp, (int(x1), int(y1)), (int(x2), int(y2)), color , 7)
                # cv2.putText(outp , name , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2 , color , 7)
                # print(track)
                
        ext = 1.2 #extend video by 1.5x
        midx = (maxx2+minx1)//2
        midy = (maxy2+miny1)//2

        extx = int((maxx2-minx1) * ext)
        exty = int((maxy2-miny1) * ext)

        cropx1 = max(midx - extx//2 , 0)
        cropx2 = min(midx + extx//2 , frame.shape[1])
        cropy1 = max(midy - exty//2 , 0)
        cropy2 = min(midy + exty//2 , frame.shape[0])

        frame = frame[cropy1:cropy2 , cropx1:cropx2]
        frame = cv2.resize(frame , OUTPUT_SHAPE)
        frames.append(frame)
    frames = np.array(frames)

    CLIP_SIZE = 32
    w , h = OUTPUT_SHAPE
    # BATCH_SIZE = 1
    frame_size = frames.shape[0]
    num_batches = (frame_size // 32)
    clipped_frames_size = num_batches * 32
    clipped_frames = frames[:clipped_frames_size]
    clipped_frames = clipped_frames.reshape(num_batches , 32 , w , h , 3)
    clipped_frames = np.rollaxis(clipped_frames, 4, 1)
    # # clipped_frames = clipped_frames[np.newaxis , ...]
    # clipped_frames = clipped_frames.astype(np.float32)
    clipped_frames = np.average(clipped_frames , axis=0)
    clipped_frames = clipped_frames.astype(np.uint8)
    print(f"Video Averaged every {num_batches} frames : ", clipped_frames.shape)

    # clipped_frames.shape
    # plt.imshow(clipped_frames[0])
    
    np.save(out , clipped_frames)