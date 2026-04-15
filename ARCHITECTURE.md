# DeepfakeGuard: Architecture & File Structure

This document provides a comprehensive breakdown of the file structure used in the **DeepfakeGuard** project and explains how the system separates its frontend (client-side) and backend (server-side) components.

---

## 1. Frontend Structure (Client-Side)
The frontend is responsible for the user interface, capturing the webcam stream, and sending that data to the server. It is primarily contained within the `app/` folder.

### HTML Layouts (`app/templates/`)
These define the structure of the webpages. As a Flask application, these are rendered server-side and sent to the browser.
*   **`base.html`**: The master template containing the common layout (headers, footers, navigation bars) that other pages inherit from.
*   **`index.html`**: The landing/home page.
*   **`login.html` & `register.html`**: The forms where users enter their credentials.
*   **`dashboard.html`**: The main interface users see after successfully authenticating.
*   **`analyze.html` & `dataset.html`**: Pages likely used for deeper views or admin features.

### Static Assets (`app/static/`)
These are the files the browser downloads directly to style the pages and execute client-side behavior.
*   **`css/`**: Contains your stylesheets (like `style.css`) that dictate colors, layouts, and animations.
*   **`js/`**: Contains client-side JavaScript. This code runs in the user's browser, handles UI button clicks, accessing the device's webcam, and transmitting the video feed to the backend over WebSockets.
*   **`assets/`**: Images, logos, and icons used on the pages.

---

## 2. Backend Structure (Server-Side)
The backend does all the heavy lifting: serving web pages, connecting to the database, and running the complex AI models. 

### The Web Server (Root Directory)
*   **`run.py`**: The main script that starts the backend Flask server and initializes the WebSocket (SocketIO) service.
*   **`config.py`**: Stores backend configuration variables (ports, database URLs, secret keys).

### API & Communication (`app/`)
*   **`routes.py`**: The HTTP backend. It handles standard web requests (e.g., when a user submits the `POST` request to register, or when they request the `/dashboard` page).
*   **`socketio_events.py`**: The real-time backend. It maintains a constant connection with the frontend Javascript. It receives frames from the user's webcam, passes them to the ML pipeline, and sends back real-time results (like "Face detected" or "Deepfake alert").

### The Core Intelligence (`ml/`)
This is the most critical backend component. It does not communicate with the frontend directly; instead, `app/socketio_events.py` uses these scripts.
*   **`face_detector.py`**: Locates faces in the images sent from the frontend.
*   **`face_recognizer.py`**: Compares the face against the database.
*   **`liveness_detector.py`**: Prevents spoofing.
*   **`deepfake_detector.py`**: Analyzes the image for digital manipulation (MesoNet implementation).
*   **`decision_engine.py`**: The master backend logic that aggregates the results from the four modules above to grant or deny access.
*   **`models/`**: Stores the heavy pre-trained weights required by the ML scripts (e.g., `.keras`, `.caffemodel`, `.dat`).

### Database & Storage (`utils/` & `data/`)
*   **`utils/db_utils.py`**: The backend code responsible for SQLite database connections and operations (adding users, verifying passwords, retrieving face encodings).
*   **`data/database.db`**: The SQLite database file storing text data (user tables, generic logs).
*   **`data/encodings/`**: A folder where mathematical representations (facial encodings/pickles) of users' faces are saved securely by the backend for future logins.

---

## 3. Maintenance & Documentation
*   **`scripts/`**: Utility scripts for system maintenance, such as:
    *   `download_models.py`: Fetches large ML weights.
    *   `train_mesonet_kaggle.py`: Model training scripts.
    *   `evaluate_metrics.py`: Performance evaluation script.
    *   `generate_pptx.py`: Automation script for generating presentation slides.
*   **`docs/`**: Documentation folder containing `project_report.md`, `ppt_slides.md`, and presentation assets.
*   **`utils/image_utils.py`**: Image processing helpers.
*   **`utils/logger.py`**: Standardized logging utility across all modules.
