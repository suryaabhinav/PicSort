# PicSort

PicSort is an intelligent, automated image organization and sorting pipeline designed to process large photo collections. It leverages advanced computer vision models to group, filter, and organize images based on quality, content, and facial recognition.

## Backend Documentation

The backend is a FastAPI application that provides a REST API for the PicSort pipeline. It includes the following endpoints:

-   `/health`: Returns a JSON object with a `ok` field set to `true` if the server is running.
-   `/api/pipeline/start`: Starts a new pipeline run. The request body should contain a JSON object with the following fields:
    -   `root`: The root directory of the image collection to process.
    -   `batch_size`: The number of images to process in each batch. Optional, default: 32.
    -   `dry_run`: Whether to perform a dry run. Optional, default: true.
    -   `run_faces`: Whether to run the face detection stage. Optional, default: true.
-   `/api/pipeline/stop/{run_id}`: Stops a running pipeline run. The `run_id` parameter should be the ID of the run to stop.
-   `/api/move/{run_id}`: Moves the images in the pipeline run to their final locations. The `run_id` parameter should be the ID of the run to move.

[Read Complete API Documentation from the backend directory](/backend/README.md)

## Frontend Documentation

Work in Progress...