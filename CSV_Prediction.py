"""
FastAPI приложение для предсказаний на CSV файлах
Использует сохраненную модель 
"""

# ========================================
# 1. ИМПОРТЫ - все необходимые библиотеки
# ========================================
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd  # Работа с CSV и DataFrame
import joblib  # Загрузка модели .pkl
import io  # Работа с файлами в памяти
import os  # Работа с файловой системой
import re  # Регулярные выражения
from typing import List, Dict, Any  # Типизация
from pydantic import BaseModel  # Валидация данных
from pathlib import Path  # Кроссплатформенные пути

# ========================================
# 2. ИНИЦИАЛИЗАЦИЯ FastAPI ПРИЛОЖЕНИЯ
# ========================================
app = FastAPI(
    title="CSV Prediction Service",  # Название в документации
    description="Сервис для предсказаний на CSV файлах с помощью ML модели",
    version="1.0.0"
)

# Настройка папок для статики и шаблонов
BASE_DIR = Path(__file__).parent
os.makedirs(BASE_DIR / "templates", exist_ok=True)
os.makedirs(BASE_DIR / "static", exist_ok=True)

# Подключение статики и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ========================================
# 3. Pydantic МОДЕЛИ ДЛЯ ВАЛИДАЦИИ
# ========================================
class PredictionResponse(BaseModel):
    """Формат ответа API"""
    filename: str  # Имя загруженного файла
    rows_count: int  # Количество строк в CSV
    predictions: List[str]  # Список предсказаний
    success_rate: float  # Доля положительных предсказаний (пример)

# ========================================
# 4. ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ И НАСТРОЙКИ
# ========================================
MODEL_PATH = BASE_DIR / "best_model.pkl"  
PRE_PATH = BASE_DIR / "preprocessor.pkl"  

# Загружаем модель при запуске приложения (один раз!)
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PRE_PATH)
    print(f"✅ Модель загружена: {MODEL_PATH}, {PRE_PATH}")
except FileNotFoundError:
    print(f"❌ Модель не найдена: {MODEL_PATH}")
    model = None

# ========================================
# 5. HTML ШАБЛОН (СОЗДАЕТСЯ АВТОМАТИЧЕСКИ)
# ========================================
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Prediction Service</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .upload-box { border: 3px dashed #007bff; padding: 50px; text-align: center; border-radius: 10px; }
        .upload-box:hover { background: #f8f9ff; }
        input[type=file] { margin: 20px 0; }
        button { background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .result { margin-top: 30px; padding: 20px; background: #d4edda; border-radius: 5px; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>🎯 ML Prediction Service</h1>
    <p>Загрузите CSV файл тестовой выборки для предсказаний</p>
    
    <form action="/predict" method="post" enctype="multipart/form-data">
        <div class="upload-box">
            <p><strong>📁 Выберите CSV файл</strong></p>
            <input type="file" name="file" accept=".csv" required>
            <br>
            <button type="submit">🚀 Получить предсказания</button>
        </div>
    </form>
    
    {% if result %}
    <div class="result">
        <h2>✅ Результат предсказаний</h2>
        <p><strong>Файл:</strong> {{ result.filename }}</p>
        <p><strong>Строк обработано:</strong> {{ result.rows_count }}</p>
        <p><strong>Успешных предсказаний:</strong> {{ "%.1f"|format(result.success_rate * 100) }}%</p>
        <h3>Первые 10 предсказаний:</h3>
        <pre>{{ result.sample_predictions }}</pre>
        <a href="/download/{{ result.filename }}" style="color: #007bff; font-weight: bold;">
            📥 Скачать полный CSV с предсказаниями
        </a>
    </div>
    {% endif %}
</body>
</html>
"""

# Сохраняем HTML шаблон
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write(html_template)

# ========================================
# 6. ОСНОВНЫЕ ЭНДПОИНТЫ (МАРШРУТЫ)
# ========================================

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    """Главная страница с веб-интерфейсом"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_csv(
    file: UploadFile = File(..., description="CSV файл тестовой выборки"),
    request: Request = None
):
    """
    ✅ ГЛАВНЫЙ ЭНДПОИНТ
    1. Принимает CSV файл через веб-форму
    2. Загружает и проверяет данные
    3. Делает предсказания с помощью модели
    4. Сохраняет результат
    5. Возвращает веб-страницу с результатом
    """
    
    # Шаг 1: Проверяем тип файла
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="❌ Только CSV файлы!")
    
    try:
        # Шаг 2: Читаем CSV файл в память
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Шаг 3: Проверяем, что данные не пустые
        if df.empty:
            raise HTTPException(status_code=400, detail="❌ CSV файл пустой!")
        
        print(f"📊 Загружен CSV: {file.filename} ({len(df)} строк, {len(df.columns)} колонок)")
        
        # Шаг 4: Проверяем наличие модели
        if model is None:
            raise HTTPException(status_code=500, detail="❌ Модель best_model.pkl не найдена!")
        
        # Шаг 5: Подготавливаем данные для предсказания
        
        df.columns = [
            re.sub(r'[^a-z0-9_]', '',
            column_name.replace(' ', '_')
           .lower()
            )
            for column_name in df.columns
            ]
        
        df = df.drop(columns=['unnamed_0'])
        
        X = df[[
            'bmi', 'age', 'triglycerides',
            'diastolic_blood_pressure', 'income',
            'heart_rate', 'exercise_hours_per_week', 'sedentary_hours_per_day',
            'cholesterol', 'systolic_blood_pressure',  'physical_activity_days_per_week',
            'stress_level', 'sleep_hours_per_day','medication_use', 'ckmb', 'previous_heart_problems', 'alcohol_consumption', 
            'diet', 'blood_sugar', 'troponin', 'gender', 'diabetes', 'smoking', 'family_history', 'obesity'
        ]]
        
        if X.empty:
            raise HTTPException(status_code=400, detail="❌ Нет числовых колонок для предсказания!")
        
        # Подготовка данных X_test_preproc  
        X_test_preproc = preprocessor.transform(X)
  
        # Шаг 6: Делаем предсказания
        predictions = model.predict(X_test_preproc)
        
        # Шаг 7: Добавляем предсказания в DataFrame
        df['prediction'] = predictions
        
        # Шаг 8: Считаем статистику
        success_rate = sum(predictions) / len(predictions) if len(predictions) > 0 else 0
        
        # Шаг 9: Сохраняем результат в static папку
        output_filename = f"predictions_{file.filename}"
        output_path = Path("static") / output_filename
        df.to_csv(output_path, index=False)
        
        # Шаг 10: Формируем ответ для веб-страницы
        result = {
            "filename": output_filename,
            "rows_count": len(df),
            "predictions": predictions,
            "sample_predictions": predictions[:10],
            "success_rate": success_rate
        }
        
        print(f"✅ Предсказания выполнены: {success_rate:.1%} положительных")
        
        # Возвращаем HTML страницу с результатом
        return templates.TemplateResponse(
            "index.html", 
            {"request": request, "result": result}
        )
        
    except Exception as e:
        print(f"💥 Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/download/{filename}")
async def download_result(filename: str):
    """Скачивание CSV с предсказаниями"""
    file_path = Path("static") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(file_path, filename=filename, media_type='text/csv')

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }

# ========================================
# 7. ЗАПУСК ПРИЛОЖЕНИЯ
# ========================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Запуск FastAPI Prediction Service...")
    uvicorn.run(
        "csv_prediction:app",  # Имя модуля:переменная
        host="0.0.0.0",
        port=8000,
        reload=True  # Автоперезагрузка при изменениях кода
    )
