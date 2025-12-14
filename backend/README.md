# PicSort Backend

PicSort is an intelligent, automated image organization and sorting pipeline designed to process large photo collections. It leverages advanced computer vision models to group, filter, and organize images based on quality, content, and facial recognition.

## 🚀 Features

-   **Smart Deduplication**: Automatically identifies and filters out duplicate or near-duplicate images to save space.
-   **Quality Assessment**: Analyzes image sharpness and focus to prioritize high-quality shots.
-   **Face Detection & Recognition**: Detects faces and clusters distinct individuals using deep learning embeddings.
-   **Scene Understanding**: Classifies images into semantic categories (e.g., landscapes, indoor) to understand context.
-   **Automated Grouping**: Intelligently groups photos into events and categories based on time, location, and content.
-   **Comprehensive Analytics**: Generates detailed reports on your photo library's composition.

---

## 🛠️ Installation & Setup

### Prerequisites

-   **Python 3.12.8**
-   **Hardware Acceleration**: CUDA (NVIDIA) or MPS (Apple Silicon) is highly recommended for optimal performance.

### Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/suryaabhinav/PicSort.git
    cd PicSort
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model Weights**
    Ensure the following model weights are present in the root directory:
    -   `yolo11m-seg.pt` (YOLO segmentation model)
    -   `yolov8n-face-lindevs.pt` (Face detection model)

---

## 💻 Usage

### Starting the Server

Start the FastAPI backend server using `uvicorn`:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`.

---

## 📡 API Documentation

### System
-   **Health Check**
    -   `GET /health`
    -   Returns: `{"ok": True}`

### Pipeline Control

-   **Start Pipeline Run**
    -   `POST /api/pipeline/start`
    -   Initiates a new background processing run.
    -   **Body**:
        ```json
        {
          "root": "/absolute/path/to/images",
          "batch_size": 32,      // Optional
          "dry_run": true,       // Optional, default: true
          "run_faces": true      // Optional, default: true
        }
        ```
    -   Returns: `{"run_id": "uuid-string"}`

-   **Stop Pipeline Run**
    -   `POST /api/pipeline/stop/{run_id}`
    -   Requests the cancellation of an active run.

-   **Apply Organization**
    -   `POST /api/move/{run_id}`
    -   Executes the final file organization (move/copy) based on the run results.
    -   **Query Parameters**: `dry_run` (boolean) - Set to `false` to actually move files.

### Monitoring

-   **Get Run Status**
    -   `GET /api/runs/{run_id}/status`
    -   Returns current progress, stage, errors, and timing metrics.

-   **Stream Events**
    -   `GET /api/runs/{run_id}/events`
    -   Server-Sent Events (SSE) stream providing real-time updates on pipeline progress.

---

## ⚙️ Pipeline Architecture

The PicSort pipeline is orchestrated by `picsort/pipeline/orchestrator.py` and executes a series of sequential stages to analyze and sort images.

### 1. Initialization
-   **Goal**: Setup runtime context and load necessary AI models onto the appropriate device (CPU/GPU/MPS).

### 2. Stage 1: Deduplication (`stage_duplicates`)
-   **Goal**: Identify redundant images.
-   **Process**: Scans the dataset for exact duplicates or visually identical images.
-   **Output**: Marks images as duplicates to exclude them from resource-intensive downstream processing.

### 3. Stage 2: Focus Analysis (`stage_a`)
-   **Goal**: Assess image technical quality.
-   **Process**: Computes sharpness scores and other quality metrics to distinguish between blurry and sharp images.
-   **Output**: Focus scores added to image metadata.

### 4. Stage 3: Face Detection (`stage_b`)
-   **Goal**: Catalog people in the collection.
-   **Process**:
    -   Detects faces in non-duplicate images.
    -   Extracts facial embeddings.
    -   (Optional) Interactive step involves clustering these embeddings to identify unique individuals.
-   **Output**: Person counts and face locations.

### 5. Stage 4: Scene Classification (`stage_c`)
-   **Goal**: Understand image context.
-   **Process**: Uses segmentation models (YOLO) to distinct scene elements (sky, grass, building, etc.) and classifies the overall scene type.
-   **Output**: Scene labels and content tags.

### 6. Stage 5: Grouping (`grouping`)
-   **Goal**: Create logical folders/events.
-   **Process**: Aggregates all collected data (Time, Faces, Scene, Quality) to cluster images into meaningful groups (e.g., "Camping Trip 2023", "Family Dinner").

### 7. Finalization & Artifacts
-   **Finalize**: Consolidates results from all stages into a master manifest.
-   **Artifacts**: The pipeline generates Parquet files for each stage in `runs/{run_id}/`:
    -   `stage_a.parquet` (Focus data)
    -   `stage_b.parquet` (Face data)
    -   `stage_c.parquet` (Scene data)
    -   `grouping.parquet` (Proposed structure)
    -   `final.parquet` (Complete dataset with move instructions)

---

## 📊 Analytics

Upon completion, the pipeline produces a comprehensive `analytics` object containing:
-   **Total Counts**: Total images, images with/without people.
-   **Distributions**: Breakdowns of scene types, person counts per image, and focus quality.
-   **Clustering**: Number of groups found and their sizes.

This data is available via the status endpoint and crucial for understanding the composition of the sorted library.
