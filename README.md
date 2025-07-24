# 👁️ Multimodal Lie Detector System

![Application Screenshot](assets/img1.jpg)
![Application Screenshot](assets/img2.jpg)
![Application Screenshot](assets/img3.jpg)

> A real-time system that analyzes physiological and behavioral cues from video to estimate the likelihood of deception.

---

## 📜 Description

This project is a sophisticated **Multimodal Lie Detection System** that leverages computer vision and machine learning to analyze human behavior for indicators of deception. By integrating data from multiple modalities—facial expressions, physiological signals derived from video, and body language—the system provides a more holistic and nuanced assessment than traditional single-modality methods.

The core of the system is its ability to establish a **personal baseline** for a subject and then detect significant deviations from it across various metrics. This approach acknowledges that stress and deception cues are highly individual, making the analysis more robust and context-aware.

---

## ✨ Key Features

- **Real-Time Video Analysis:** Processes a video feed to extract cues without significant delay.
- **Baseline Calibration:** Establishes a 15-second physiological and behavioral baseline at the start for accurate deviation analysis.
- **Multimodal Cue Detection:**
  - **Physiological (rPPG):** Measures heart rate and respiration rate from subtle changes in skin color on the forehead.
  - **Ocular Analysis:** Tracks blink rate and eye gaze direction to detect nervousness or gaze aversion.
  - **Facial Expressions:** Identifies dominant emotions (e.g., fear, anger, surprise) using the FER library.
  - **Behavioral Cues:** Monitors head movement, hand-to-face gestures, and lip compression.
- **Deception Likelihood Score:** A weighted algorithm calculates a final score (0-95%) representing the estimated likelihood of deception based on anomalies detected.
- **Informative UI:** A clean, semi-transparent overlay displays all real-time data without completely obstructing the subject's face.

---

## 🛠️ Technology Stack

- **Core Language:** Python 3.8+
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning / Deep Learning:** TensorFlow, FER (Facial Emotion Recognition)
- **Data & Scientific Computing:** NumPy, SciPy
- **Audio & Video Processing:** MoviePy, Librosa

---

## ⚙️ Installation

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)
- `git` (for cloning the repository)

### Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jeno22ndr/Multimodal-Lie-detector-system.git](https://github.com/jeno22ndr/Multimodal-Lie-detector-system.git)
    cd Multimodal-Lie-detector-system
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Assets:**
    - Ensure you have a `meter.png` file for the deception meter in the root directory.
    - Place a screenshot of the app named `app_screenshot.jpg` inside an `assets` folder.

---

## 🚀 Usage

1.  **Configure the Video Path:**
    - Open the main Python script (e.g., `lie_detector.py`).
    - Find the `video_path` variable and update it with the full path to your input video file.

2.  **Run the application from your terminal:**
    ```bash
    python lie_detector.py
    ```

3.  **Application Controls:**
    - **`P` key:** Pause or resume the video playback.
    - **`1`, `2`, `3` keys:** Adjust the playback speed (1x, 2x, 3x).
    - **`Q` key:** Quit the application.

---

## ⚠️ Disclaimer

This tool is developed for educational and experimental purposes only. The detection of deception is a complex and nuanced field, and the physiological and behavioral indicators used by this software are not guaranteed to be accurate. This tool should **not** be used to make real-life judgments about an individual's truthfulness or character.

---

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

---

## 📧 Contact

Jeno - [your.email@example.com](mailto:jeno22ndr@gmail.com)

Project Link: (https://github.com/jeno22ndr/Multimodal_Lie-_detector_system)
