from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import vision
import dummy_llm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/inference")
async def perform_inference(image: UploadFile = File(...), text: str = Form(...)):
    # 1. Extract bytes from the uploaded image
    image_bytes = await image.read()
    
    # 2. Route the image bytes to the Vision Encoder (if implemented)
    vision_features = None
    if hasattr(vision, 'encode'):
        vision_features = vision.encode(image_bytes)
        
    # 3. Route the text and vision features to the Dummy LLM (if implemented)
    language_output = None
    if hasattr(dummy_llm, 'generate'):
        language_output = dummy_llm.generate(text, vision_features)
        
    return {
        "status": "success",
        "message": f"Received {len(image_bytes)} bytes of image and text: '{text}'.",
        "vision_features_extracted": vision_features is not None,
        "language_output": language_output
    }
