# app.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import cv2
from typing import List
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import supervision as sv
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("CORS middleware added")

# Ensure the static directory exists
static_dir = os.path.join(os.getcwd(), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount the static files directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Load your AutoDistill model
ontology = CaptionOntology({"rooftop": "rooftop of a building"})
base_model = GroundedSAM(ontology=ontology)

class Polygon(BaseModel):
    coordinates: List[List[float]]

class SolarPotentialResponse(BaseModel):
    polygons: List[Polygon]
    areas: List[float]
    total_area: float
    solar_potential: float
    annotated_image_url: str

def read_image(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    image = np.array(Image.open(io.BytesIO(contents)))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

def annotate_image(image: np.ndarray):
    detections = base_model.predict(image)
    return detections

def calculate_area(detections: sv.Detections, gsd: float) -> List[float]:
    areas = []
    for xyxy in detections.xyxy:
        width = (xyxy[2] - xyxy[0]) * gsd
        height = (xyxy[3] - xyxy[1]) * gsd
        areas.append(width * height)
    return areas

def calculate_solar_potential(area: float, efficiency: float = 0.15, performance_ratio: float = 0.75, peak_sun_hours: float = 4) -> float:
    solar_irradiance = 1  # kW/m²
    return area * solar_irradiance * efficiency * performance_ratio * peak_sun_hours

@app.post("/analyze_rooftop", response_model=SolarPotentialResponse)
async def analyze_rooftop(file: UploadFile = File(...)):
    logger.info(f"Received request to /analyze_rooftop")
    try:
        logger.info("Starting rooftop analysis")
        image = read_image(file)
        logger.info("Image read successfully")
        
        detections = annotate_image(image)
        logger.info(f"Image annotated successfully. Detections: {detections}")
        
        gsd = 0.1  # Ground Sample Distance in meters/pixel. Adjust as needed.
        areas = calculate_area(detections, gsd)
        total_area = sum(areas)
        solar_potential = calculate_solar_potential(total_area)
        logger.info(f"Calculated total area: {total_area}, solar potential: {solar_potential}")
        
        # Annotate the image
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        
        # Save the annotated image
        annotated_image_path = os.path.join(static_dir, "annotated_rooftop.jpg")
        cv2.imwrite(annotated_image_path, annotated_image)
        logger.info(f"Annotated image saved at {annotated_image_path}")
        
        # Convert detections to Polygon objects
        polygons = [Polygon(coordinates=[
            [float(xyxy[0]), float(xyxy[1])],  # top-left
            [float(xyxy[2]), float(xyxy[1])],  # top-right
            [float(xyxy[2]), float(xyxy[3])],  # bottom-right
            [float(xyxy[0]), float(xyxy[3])]   # bottom-left
        ]) for xyxy in detections.xyxy]
        logger.info(f"Created {len(polygons)} polygons")
        
        response = SolarPotentialResponse(
            polygons=polygons,
            areas=areas,
            total_area=total_area,
            solar_potential=solar_potential,
            annotated_image_url=f"/static/annotated_rooftop.jpg"
        )
        logger.info(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/static/{filename}")
async def serve_static(filename: str):
    return FileResponse(os.path.join(static_dir, filename))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting the server...")
    try:
        uvicorn.run(app, host='0.0.0.0', port=8000, log_level="debug")
    except Exception as e:
        logger.error(f"Failed to start the server: {str(e)}", exc_info=True)