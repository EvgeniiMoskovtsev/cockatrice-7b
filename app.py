from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging
from datetime import datetime
from vllm import LLM, SamplingParams
import argparse
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_metrics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cockatrice API")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

class GenerationResponse(BaseModel):
    generated_text: str
    metrics: dict
    status: str

def load_model(use_quantization: bool = False):
    logger.info(f"Загрузка модели с квантизацией: {use_quantization}")
    
    model_path = "/root/models/cockatrice"
    
    return LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        quantization="fp8" if use_quantization else None,
        trust_remote_code=True,
        max_model_len=23000
    )

# Глобальная переменная для хранения модели
llm = None

@app.on_event("startup")
async def startup_event():
    global llm
    # Получаем аргументы командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", action="store_true", help="Использовать квантизацию")
    args, _ = parser.parse_known_args()
    
    # Загружаем модель с нужными параметрами
    llm = load_model(args.quantize)
    logger.info("Модель успешно загружена!")

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    try:
        request_start_time = time.time()
        
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_length,
            top_p=0.95
        )
        
        generation_start = time.time()
        outputs = llm.generate(request.prompt, sampling_params)
        generation_time = time.time() - generation_start
        
        generated_text = outputs[0].outputs[0].text
        
        total_time = time.time() - request_start_time
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'prompt_length': len(request.prompt),
            'output_length': len(generated_text),
            'generation_time': round(generation_time, 3),
            'total_time': round(total_time, 3)
        }
        
        logger.info(f"Request metrics: {metrics}")
        
        return GenerationResponse(
            generated_text=generated_text,
            metrics=metrics,
            status='success'
        )
    
    except Exception as e:
        error_time = time.time() - request_start_time
        logger.error(f"Error occurred after {error_time:.3f} seconds: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False) 