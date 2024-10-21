# Hand Gesture Spotify Controller

This project allows you to control Spotify playback using hand gestures, leveraging computer vision techniques. Using a webcam, it recognizes specific hand gestures and translates them into Spotify commands like play/pause, next track, previous track, and volume control.

## Project Overview
This project was created for fun to explore the potential of hand gesture recognition using OpenCV and MediaPipe. You can control your Spotify playlist using predefined hand gestures, such as raising a finger or making a fist, detected via a webcam feed.

### Features
- **Play/Pause**: Raise your index finger.
- **Next Track**: Raise both your index and middle fingers.
- **Previous Track**: Raise your thumb and index finger sideways.
- **Mute Volume**: Raise your pinky finger.
- **Max Volume**: Raise all five fingers.
- **Like Track**: Raise your thumb.

### Requirements
Ensure you have Python 3.11.8 installed. Install the required libraries using the `requirements.txt` file.

### Installation

1. Clone the repository or download the files.
2. Ensure you have Python version 3.11.8 installed.
3. Install the required dependencies by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

4. Get your Spotify API credentials from the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications).
5. Update the `client_id`, `client_secret`, and `redirect_uri` in the script with your Spotify API credentials.

### Running the Project

After setting up your Spotify API credentials and installing the necessary modules, you can run the project by executing:

```bash
python hand_gesture_spotify.py
