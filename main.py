import cv2 as opencv
import numpy as np
from datetime import datetime
import os


def main():
    """
    Uses a live video feed to detect motion and write the frames that contain motion to disk. The application of this
    is to only write the frames that show motion to conserve space on a limited resource disk.

    Each captured frame is passed though a background subtraction algorithm to separate out any foreground from the
    model that the system already knows about. The foreground pixels are an indicator of movement. The extracted
    foreground frame has noise deduction applied to make the motion segmented and dilation applied to make the motion
    more prevalent in the image. Once the pre-processing has been applied, the number of black pixels are summed and
    checked against a threshold value to check if there is definitive motion in the frame. If the previous frame also
    contained motion then we're capturing a scene live as such keep track of all the frames. Otherwise, we have just
    captured a live scene which has now ended as the current frame does not contain motion, so write out the frames
    that have been captures as a video.

    Features
    -------
    ./ Quite resilient to general movement such as leaves blowing or branches shaking
    ./ Good for detecting people walking and cars driving or parking
    ./ Considerate of hardware with limited storage (such as Rasp Pi) as it only writes during times of motion detected

    Bugs
    -------
    ./ Should the motion during a detection stop for an extended period, then the video will stop recording and the
       frames it has already generated will be written out. It will record once again when the motion starts again but
       during experiments a car attempting to reverse park stopped long enough for the background subtraction model to
       incorporate the car which caused numerous videos being written.
    ./ The application does not scale well. The threshold of 15% black pixels to be indicative of movement was an
       arbitrary value. 
    ./ Many other hardcoded values that are to be parameterised or determined at run time preferably.


    """
    # TODO: modify the size of the history to change the model size to make it more or less sensitive to change
    background_subtractor = opencv.createBackgroundSubtractorKNN(history=5000)
    was_previous_frame_interesting = None
    list_of_interesting_frames = []

    cap = opencv.VideoCapture(0)
    while True:
        ret, capture = cap.read()

        if capture is None:
            break

        frames_width, frames_height, _ = capture.shape

        # Subtract the capture from the background
        frame = background_subtractor.apply(capture)

        # Create the threshold. This should be a parameter we can add into the command line arguments
        threshold_value = opencv.THRESH_BINARY_INV + opencv.THRESH_OTSU
        _, thresh = opencv.threshold(frame, 0, 255, threshold_value)

        # Create a kernel to apply the dilation to smooth the frame and make the foreground more prevalent
        kernel = np.ones((3, 3), np.uint8)
        closing = opencv.morphologyEx(thresh, opencv.MORPH_CLOSE, kernel, iterations=2)

        # Background area using dilation
        _ = opencv.dilate(closing, kernel, iterations=1)

        # Finding foreground area
        dist_transform = opencv.distanceTransform(closing, opencv.DIST_L2, 0)
        _, frame = opencv.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)

        # does the frame contain a number of black pixels that can be classed as motion?
        is_current_frame_interesting = contains_various_black_pixels(frame)

        # The previous frame had black pixels indicating movement
        if was_previous_frame_interesting:
            # The current frame also has some movement - so add it to the list of frames
            if is_current_frame_interesting:
                list_of_interesting_frames.append(capture)
            else:
                # We've lost movement in the current frame but we do have in previous frames so write the frames we do
                # have.
                write_frames_as_video(list_of_interesting_frames)
                # Once we've written the interesting frames to disk, clear the list ready for the next
                list_of_interesting_frames.clear()

        # keep track of if the frame was interesting or not for the next iteration
        was_previous_frame_interesting = is_current_frame_interesting

        # show the image
        opencv.imshow('motion detector', frame)
        opencv.imshow('original', capture)

        if opencv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    opencv.destroyAllWindows()


def write_frames_as_video(list_of_frames):
    """Writes a list of frames to the disk as a video if the number of frames are more than 20

    Parameters
    -------
    list_of_frames - The frames to be written to the disk
    """
    # Won't be very exciting if it's only a small number of frames, so don't bother writing it out
    if len(list_of_frames) < 20:
        return

    print("about to write " + str(len(list_of_frames)) + " frames to a video")

    # get the time and date for the file name
    date_time = get_current_date_time()
    # TODO: This path is to become parametrised. Also create a folder for video capture on a day to day basis.
    full_path = 'C:\\security_cam\\video_' + str(date_time) + ".avi"

    # Get the frame's width and height
    width, height, _ = list_of_frames[0].shape
    out = opencv.VideoWriter(full_path, opencv.VideoWriter_fourcc(*'XVID'), 30, (width, height), True)

    # write all of the frames out to the video file
    for frame_to_write in list_of_frames:
        out.write(frame_to_write)
    out.release()

    print('completed writing file: %s' % full_path)


def contains_various_black_pixels(frame):
    """Determines if a frame contains a number of black pixels which in the context of this application indicate
       movement from the frame

    Parameters
    -------
    frame - The frame to test

    Returns
    -------
    True if the frame contains a threshold of black pixels, otherwise False
    """
    # The threshold of black pixels is 10% of the frame
    width, height = frame.shape
    threshold = ((width * height) / 100) * 10

    # Sum the number of black pixels
    n_black_pixels = np.sum(frame == 0)
    return n_black_pixels >= threshold


def get_current_date_time():
    """Gets the current date and time as a string

    Returns
    -------
    the date and time in the format DD-MM-YYY_HH-MM-SS
    """

    now = datetime.now()
    return now.strftime("%d-%m-%Y_%H-%M-%S")


if __name__ == '__main__':
    """Entry point of the application.
    """
    main()
