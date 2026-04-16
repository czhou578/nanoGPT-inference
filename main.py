from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

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
    # Simulating some processing
    return {
        "status": "success",
        "message": f"Processed image '{image.filename}' with text '{text}'"
    }
