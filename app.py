from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_metrics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_NAME = "openerotica/cockatrice-7b-v0.1"

print("Загрузка модели и токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    # load_in_8bit=True 
)
print("Модель загружена успешно!")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Начало измерения общего времени запроса
        request_start_time = time.time()
        
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.7)
        
        # Измеряем время токенизации
        tokenization_start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        tokenization_time = time.time() - tokenization_start
        
        # Измеряем время генерации
        generation_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generation_time = time.time() - generation_start
        
        # Измеряем время декодирования
        decoding_start = time.time()
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoding_time = time.time() - decoding_start
        
        # Общее время запроса
        total_time = time.time() - request_start_time
        
        # Логируем метрики
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'prompt_length': len(prompt),
            'output_length': len(generated_text),
            'tokenization_time': round(tokenization_time, 3),
            'generation_time': round(generation_time, 3),
            'decoding_time': round(decoding_time, 3),
            'total_time': round(total_time, 3)
        }
        
        logger.info(f"Request metrics: {metrics}")
        
        return jsonify({
            'generated_text': generated_text,
            'metrics': metrics,
            'status': 'success'
        })
    
    except Exception as e:
        error_time = time.time() - request_start_time
        logger.error(f"Error occurred after {error_time:.3f} seconds: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 