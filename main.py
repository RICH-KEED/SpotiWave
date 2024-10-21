import cv2
import mediapipe as mp
import time
import math
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Initialize Spotify API
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(                   client_id='',client_secret='',redirect_uri='',scope='user-modify-playback-state user-read-playback-state user-library-modify'))

class handDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.3, trackCon=0.5):
        self.mode = mode
        self.maxHands = int(maxHands)  # Ensuring maxHands is an integer
        self.detectionCon = float(detectionCon)  # Ensuring detectionCon is a float
        self.trackCon = float(trackCon)  # Ensuring trackCon is a float

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                              (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def play_pause():
    current_playback = sp.current_playback()
    if current_playback and current_playback['is_playing']:
        sp.pause_playback()
        print("Paused the song.")
    else:
        sp.start_playback()
        print("Playing the song.")

def next_track():
    sp.next_track()
    print("Skipped to the next track.")

def previous_track():
    sp.previous_track()
    print("Skipped to the previous track.")

def like_track():
    current_playback = sp.current_playback()
    if current_playback:
        track_id = current_playback['item']['id']
        sp.current_user_saved_tracks_add([track_id])
        print(f"Liked the track: {current_playback['item']['name']}")

def set_volume(volume_level):
    sp.volume(volume_level)
    print(f"Volume set to {volume_level}%")

def volume_max():
    set_volume(100)
    print("Volume set to maximum.")

def volume_mute():
    set_volume(0)
    print("Volume muted.")

def volume_medium():
    set_volume(50)
    print("Volume set to 50%.")

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = handDetector(detectionCon=0.6)
    frame_skip = 5  # Process every 5th frame to improve performance
    frame_count = 0
    
    while True:
        frame_count += 1
        success, img = cap.read()
        if not success:
            break

        if frame_count % frame_skip == 0:
            img = detector.findHands(img)
            lmList, bbox = detector.findPosition(img)
            if len(lmList) != 0:
                print(lmList[4])

                # Example gestures
                if detector.fingersUp() == [0, 1, 0, 0, 0]:  # Only index finger up
                    play_pause()
                    time.sleep(1)  # Add a delay to avoid multiple triggers
                elif detector.fingersUp() == [0, 1, 1, 0, 0]:  # Index and middle fingers up
                    next_track()
                    time.sleep(1)
                elif detector.fingersUp() == [1, 1, 1, 1, 1]:  # All fingers up
                    volume_max()
                    time.sleep(1)
                elif detector.fingersUp() == [0, 0, 0, 0, 1]:  # Only pinky finger up
                    volume_mute()
                    time.sleep(1)
                elif detector.fingersUp() == [1, 1, 0, 0, 0]:  # Thumb and index finger up (sideways)
                    previous_track()
                    time.sleep(1)
                elif detector.fingersUp() == [1, 0, 0, 0, 0]:  # Only Thumb up
                    like_track()
                    time.sleep(1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
