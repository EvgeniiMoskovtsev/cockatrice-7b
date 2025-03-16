# Cockatrice-7B API Server

Сервер API для модели Cockatrice-7B v0.1 на базе Flask.

## Системные требования

- Python 3.8+
- CUDA-совместимая видеокарта с минимум 16GB VRAM
- Минимум 32GB оперативной памяти
- Около 15GB свободного места на диске

## Установка

1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
# или
.\venv\Scripts\activate  # для Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Запуск сервера

```bash
python app.py
```

Сервер запустится на `http://localhost:5000`

## Использование API

Отправьте POST-запрос на endpoint `/generate`:

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ваш текст здесь",
    "max_length": 100,
    "temperature": 0.7
  }'
```

### Параметры:

- `prompt`: Текст для продолжения (обязательный)
- `max_length`: Максимальная длина генерации (по умолчанию: 100)
- `temperature`: Температура генерации (по умолчанию: 0.7)

### Пример ответа:

```json
{
  "generated_text": "Сгенерированный текст",
  "status": "success"
} 