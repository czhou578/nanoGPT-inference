from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import vision
import dummy_llm
from continuous_batching import Engine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the vision encoder and LLM once globally
vision_model = vision.Vision()
llm_model = dummy_llm.Dummy_LLM()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/inference")
async def perform_inference(image: UploadFile = File(...), text: str = Form(...)):
    # 1. Extract bytes from the uploaded image
    image_bytes = await image.read()

    # 2. Encode the image into vision feature vectors
    vision_features = vision_model.encode(image_bytes)

    # 3. Create a fresh Engine for this request and submit the job
    #    (In a real server this Engine would be long-lived and shared;
    #     we create it per-request here to keep the example stateless.)
    engine = Engine(llm_model)
    seq_id = engine.submit(text, vision_features)

    # 4. Run the continuous batching loop to completion
    results = engine.run(max_new_tokens=10)
    generated_token_ids = results.get(seq_id, [])

    return {
        "status": "success",
        "message": f"Received {len(image_bytes)} bytes of image and text: '{text}'.",
        "vision_features_extracted": vision_features is not None,
        "language_output": {
            "generated_token_ids": generated_token_ids,
            "num_tokens_generated": len(generated_token_ids),
        }
    }
