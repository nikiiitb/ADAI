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
colors = np.random.randint(0 , 256 , size=(10,3))

vid_location = "./SSBD/videos/"
out_location = "./inputs"

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
#     # "./inputs/v_ArmFlapping_01.avi",
#     # "./inputs/v_ArmFlapping_02.avi",
#     # "./inputs/v_ArmFlapping_03.avi",
#     # "./inputs/v_HeadBanging_01.avi",
#     # "./inputs/v_HeadBanging_02.avi",
#     # "./inputs/v_HeadBanging_03.avi",
#     # "./inputs/v_Spinning_01.avi",
#     # "./inputs/v_Spinning_02.avi",
#     "./inputs/v_Spinning_04.avi",
# ]

for vid , vid_out in zip(vids , vid_outs):
    vid = cv2.VideoCapture(vid)
    # vid = cv2.VideoCapture("./SSBD/videos/v_ArmFlapping_02.avi")
    ret, frame = vid.read()

    vid_out = cv2.VideoWriter(vid_out, cv2.VideoWriter_fourcc(*'MP4V'), vid.get(cv2.CAP_PROP_FPS),
                            (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))
    # vid_out = cv2.VideoWriter("./out2.mp4", cv2.VideoWriter_fourcc(*'MP4V'), vid.get(cv2.CAP_PROP_FPS),
    #                           (frame.shape[1], frame.shape[0]))

    prev_crop = (0 , 0 , frame.shape[0] , frame.shape[1])

    while ret:
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
                if class_id == 0 and score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score , class_id])
            detections = np.array(detections)
            if len(detections) >= 1:
                track_bbs_ids = tracker.update(detections)
            else:
                track_bbs_ids = []

            for track in track_bbs_ids:
                # bbox = track.bbox
                # x1, y1, x2, y2 = bbox
                # track_id = track.track_id
                print("TRACK : " , track)
                x1, y1, x2, y2, id = list(map(int , track.tolist()))
                prev_crop = (x1 , y1 , x2 , y2)
                name = f"Child : {id}"
                color = colors[int(id) % len(colors)]
                color = tuple(color.tolist())
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color , 7)
                # cv2.putText(frame , name , (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2 , color , 7)
                # print(track)
        x1 , y1 , x2 , y2 = prev_crop
        x1 , y1 , x2 , y2 = max(0,x1) , max(0,y1) , max(0,x2) , max(0,y2)

        ext = 1.5 #extend video by 1.5x

        midx = (x2-x1)//2
        midy = (y2-y1)//2

        extx = int((x2-x1) * ext)
        exty = int((y2-y1) * ext)

        cropx1 = max(midx - extx//2 , 0)
        cropx2 = max(midx + extx//2 , frame.shape[1])
        cropy1 = max(midy - exty//2 , 0)
        cropy2 = max(midy + exty//2 , frame.shape[0])

        outp = frame[cropy1:cropy2 , cropx1:cropx2]
        outp = cv2.resize(outp , OUTPUT_SHAPE)
        print("FRAME : " , outp.shape)
        # outp = frame

        # cv2.imshow('frame' , frame)
        # cv2.waitKey(25)
        vid_out.write(outp)
        ret,frame = vid.read()

    vid.release()
    vid_out.release()
    cv2.destroyAllWindows()