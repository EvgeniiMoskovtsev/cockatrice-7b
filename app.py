from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Инициализация модели и токенизатора
MODEL_NAME = "openerotica/cockatrice-7b-v0.1"

print("Загрузка модели и токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True  # Использование квантизации для оптимизации памяти
)
print("Модель загружена успешно!")

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 100)
        temperature = data.get('temperature', 0.7)
        
        # Токенизация входного текста
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Генерация текста
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Декодирование результата
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            'generated_text': generated_text,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 