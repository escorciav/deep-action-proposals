from subprocess import check_output

import cv2


def count_frames(filename, method=None):
    # Return number of frames of a video
    counter, fail_ffprobe = 0, False
    if method == 'ffprobe':
        cmd = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams',
               'v:0', '-show_entries', 'stream=nb_read_frames', '-of',
               'default=nokey=1:noprint_wrappers=1', filename]
        try:
            counter = int(check_output(cmd).replace('\n', ''))
        except:
            counter, fail_ffprobe = 0, True

    if method is None or fail_ffprobe:
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret = cap.grab()
            if ret:
                counter += 1
            else:
                break
        cap.release()
    return counter
