# inference.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def model_fn(model_dir):
    """모델과 토크나이저 로드"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)  # train.py와 동일
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)  # num_labels=2 추가
    return {"model": model, "tokenizer": tokenizer}

def input_fn(request_body, request_content_type):
    """입력 데이터 전처리"""
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data["contents"]  # "text" → "contents" (train.py 컬럼명과 일치)
    else:
        raise ValueError("Unsupported content type")

def predict_fn(input_data, model_dict):
    """예측 수행"""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # train.py의 tokenize_function과 동일한 설정
    inputs = tokenizer(input_data, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
    
    return {
        "predicted_class": predicted_class,  # 0 또는 1
        "probabilities": predictions.numpy().tolist()[0],  # [prob_0, prob_1]
        "label": "Y" if predicted_class == 1 else "N"  # 원래 라벨 형태
    }

def output_fn(prediction, content_type):
    """출력 데이터 후처리"""
    return json.dumps(prediction)
