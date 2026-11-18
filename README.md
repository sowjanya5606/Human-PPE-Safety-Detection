# Human & PPE Detection with Safety Zone Alert

## 1. Project Overview

This project implements a computer vision-based safety monitoring system to detect humans, identify helmet usage, monitor entry into a predefined critical safety region, and generate visual alerts in recorded videos.

The system uses a custom-trained YOLOv8 model, ByteTrack object tracking, helmet-to-person association, and Region of Interest (ROI) monitoring.

## 2. Objectives

- Detect humans in video frames.
- Detect helmets and associate them with individual persons.
- Define a critical safety region (ROI).
- Track persons across video frames.
- Generate visual alerts for safety violations.
- Process recorded videos with near-real-time performance.

## 3. Technologies Used

- Python 3.11
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- NumPy
- ByteTrack
- Jupyter Notebook
- Roboflow for dataset annotation and augmentation

## 4. Dataset Preparation and Model Training

Frames were extracted from the supplied input videos and annotated using Roboflow.

Two object classes were annotated:

- Person
- Helmet

Data augmentation was applied to improve training data diversity.

A YOLOv8 Nano model was trained using the following configuration:

| Parameter | Value |
|---|---|
| Base model | YOLOv8n |
| Epochs | 10 |
| Image size | 640 |
| Batch size | 8 |
| Training images | 8,882 |
| Validation images | 2,538 |
| Test images | 1,269 |

The trained model contains two classes:

- Class 0: Helmet
- Class 1: Person

### Test Dataset Performance

| Metric | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Overall | 0.957 | 0.932 | 0.976 | 0.732 |
| Helmet | 0.969 | 0.914 | 0.970 | 0.678 |
| Person | 0.946 | 0.950 | 0.982 | 0.787 |

These metrics are from the model's test split. Since frames were extracted from the supplied videos, similar or adjacent frames may occur across dataset splits. Therefore, these results should not be interpreted as independent validation on completely unseen environments.

## 5. System Workflow

1. Load the trained YOLOv8 model.
2. Read the input video frame by frame.
3. Detect persons and helmets.
4. Track detected objects using ByteTrack.
5. Associate helmet detections with person detections using bounding-box overlap in the upper head region.
6. Check whether each person's bottom-centre point lies inside the defined safety ROI.
7. Determine the person's safety status.
8. Draw bounding boxes, tracking IDs, ROI, and alert labels.
9. Save the processed output video.

## 6. Helmet Association

The system associates helmets with persons using the overlap between the helmet bounding box and the upper 35% of the person's bounding box.

An overlap threshold of 0.30 is used.

A greedy matching strategy assigns each helmet to at most one person.

This is a geometric association method and does not use pose estimation or dedicated head detection.

## 7. Safety Zone (ROI) Monitoring

A polygonal Region of Interest is defined for each camera view.

### Video 0 ROI

```python
roi_video_0 = np.array([
    [477, 311],
    [1118, 311],
    [1200, 658],
    [510, 658]
], dtype=np.int32)
```

### Videos 1, 2 and 3 ROI

```python
roi_video_1_2_3 = np.array([
    [350, 200],
    [1050, 200],
    [1300, 607],
    [250, 607]
], dtype=np.int32)
```

The bottom-centre point of each person's bounding box is checked against the polygon using OpenCV's `pointPolygonTest`.

These ROI coordinates were manually selected for the demonstration and are not officially approved hazard boundaries.

## 8. Safety Decision Logic

| Condition | Status | Bounding Box |
|---|---|---|
| Helmet detected and person outside ROI | SAFE | Green |
| Person enters critical ROI | ROI INTRUSION | Red |
| Helmet not matched to person | HELMET NOT DETECTED | Red |

ROI intrusion takes priority if both conditions occur.

A helmet-not-detected alert means the system did not find a matching helmet detection. It does not necessarily prove that the person is not wearing a helmet.

## 9. Object Tracking

ByteTrack is used to maintain tracking IDs across video frames.

Tracking helps identify persons across consecutive frames and supports person-level visual status labels.

Tracking IDs may change if a person is temporarily lost due to occlusion or detection interruption.

## 10. Results

All four supplied videos were processed successfully.

| Video | Frames | Safe | ROI Intrusion | Helmet Alert |
|---|---:|---:|---:|---:|
| Video 0 | 173 | 301 | 100 | 156 |
| Video 1 | 400 | 44 | 166 | 70 |
| Video 2 | 153 | 20 | 42 | 55 |
| Video 3 | 4,600 | 1,917 | 194 | 1,210 |
| **Total** | **5,326** | **2,282** | **502** | **1,491** |

Status counts represent person detections across frames, not unique persons or unique violation incidents.

Processed videos are saved in the `outputs` directory.

## 11. Performance Benchmark

The complete pipeline was benchmarked three times on the first 200 frames of Video 3.

The benchmark included detection, ByteTrack tracking, helmet-to-person association, ROI checking, visualization, and video writing. Processing was performed on the CPU with an image size of 640.

| Run | Frames processed | Processing time | Pipeline FPS |
|---|---:|---:|---:|
| 1 | 200 | 12.17 seconds | 16.44 |
| 2 | 200 | 20.92 seconds | 9.56 |
| 3 | 200 | 13.23 seconds | 15.12 |
| **Average per run** | **200** | **15.44 seconds** | **13.71** |

Across all three runs, 600 frames were processed in 46.32 seconds, giving an aggregate throughput of **12.95 FPS**. The 13.71 FPS figure is the arithmetic mean of the three individual run FPS values; aggregate throughput is calculated from total frames divided by total elapsed time.

The source video FPS was 10. These are sample measurements on the development machine, not a guarantee of performance under all deployment conditions. Results can vary with CPU load and video encoding/writing overhead.

## 12. Project Structure

```text
Human_PPE_Safety_Detection/
│
├── models/
│   └── human_ppe_best.pt
│
├── notebooks/
│   └── PPE_Detection_Development_clean.ipynb
│
├── outputs/
│   ├── video_0_safety_output.mp4
│   ├── video_1_safety_output.mp4
│   ├── video_2_safety_output.mp4
│   └── video_3_safety_output.mp4
│
├── dataset/
│   └── input_videos/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 13. Installation and Execution

### Requirements

- Python 3.11
- pip
- Jupyter Notebook or VS Code

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment (Windows)

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

Open:

`notebooks/PPE_Detection_Development_clean.ipynb`

Run the notebook cells in order after placing the input videos in `dataset/input_videos`.

The trained model is available at:

`models/human_ppe_best.pt`

## 14. Limitations and Future Improvements

- Helmet association is based on bounding-box overlap and may fail under occlusion, overlapping persons, or unusual camera angles.
- A missing helmet detection can result from poor visibility, small object size, or model confidence limitations.
- ROI coordinates are manually configured for the supplied camera views.
- Tracking IDs may change when detection is interrupted.
- Model generalization should be evaluated on independent videos from different sites and lighting conditions.
- Further improvements could include pose estimation, improved person-helmet association, camera-specific ROI configuration, and testing on additional unseen environments.

## 15. Conclusion

The project demonstrates an end-to-end computer vision pipeline for human detection, helmet monitoring, object tracking, and safety-zone intrusion alerts on recorded videos.

The system successfully processed all four supplied videos and generated annotated output videos with visual safety alerts.
