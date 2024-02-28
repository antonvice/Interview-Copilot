import time
import torch
from transformers import pipeline
import tempfile
from fastapi import FastAPI, Request,  File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from transformers.utils import is_flash_attn_2_available
import ollama

class ModelManager:
    def __init__(self):
        self.model_name = None

    def load_model(self, model_name):
        try:
            ollama.pull(model_name)
            self.model_name = model_name
            return True, f"Model {model_name} successfully loaded."
        except Exception as e:
            return False, f"Error loading model {model_name}: {e}"

    def get_model_name(self):
        return self.model_name
    

model_manager = ModelManager()

device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
pipe = pipeline(
        "automatic-speech-recognition",
        model="distil-whisper/distil-large-v2", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device=device, 
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

# Define a decorator to measure execution time and print arguments
def timeit(func):
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # Represent positional arguments
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # Represent keyword arguments
        signature = ", ".join(args_repr + kwargs_repr)  # Combine argument representations
        print(f"Calling {func.__name__}({signature})")  # Print function call details

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(f"{func.__name__} took {end - start} seconds to execute")
        return result
    return wrapper

# Example usage with the transcribe function
@timeit
def transcribe(audio_path):
    # Assuming pipe is defined globally and used here
    outputs = pipe(audio_path, chunk_length_s=30, batch_size=64, return_timestamps=True)
    return outputs['text']


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.exception_handler(422)
async def handle_422_error(request, exc):
    """Handle 422 Unprocessable Entity errors."""
    print(exc)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    print(file)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
        # Save the uploaded file to a temporary file
        contents = await file.read()
        temp_file.write(contents)
        temp_file.seek(0)  # Go back to the beginning of the file

        # Transcribe the audio file
        text = transcribe(temp_file.name)

    # Return the transcription as JSON
    return JSONResponse(content={"transcript": text})

@app.post("/suggestions")
async def get_suggestions(request: Request):
    data = await request.json()
    transcript = data.get("transcript", "")
    
    prompt = data.get("prompt", "")
    
    prompt = f"{prompt}\n---\n{transcript}"
    r = await respond(prompt)
    #return json "suggestion : r"
    return JSONResponse(content={"suggestion": r})

    

# Modify the load_model endpoint to use the ModelManager
@app.post("/submit-model")
async def load_model(request: Request):
    data = await request.json()
    model_name = data.get("model_name")
    if model_name:
        success, message = model_manager.load_model(model_name)
        if success:
            return HTMLResponse(content=f"<p>{message}</p>")
        else:
            return HTMLResponse(content=f"<p>{message}</p>")
    else:
        return HTMLResponse(content="<p>ERROR: No model name provided.</p>")


async def respond(query):
    model_name = model_manager.get_model_name()
    messages = [{"role": "user", "content": query}]
    response = ollama.chat(model=model_name, messages=messages)
    r = response['message']['content']
    return r