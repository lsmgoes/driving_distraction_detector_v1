import time
import cv2
from src.utils import initialize_video, process_frame


def main():
    """Main function to start video capture and process frames."""
    video = initialize_video()
    prev_frame_time = time.time()

    while True:
        ret, frame = video.read()
        if not ret:
            print("Error capturing frame.")
            break

        processed_frame, prev_frame_time = process_frame(
            frame, prev_frame_time
        )

        cv2.imshow("Monitoring", processed_frame)

        if cv2.waitKey(1) == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
