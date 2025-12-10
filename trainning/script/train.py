# scripts/train.py
import argparse
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)
import numpy as np
from datasets import Dataset
import pandas as pd 

# Pandas DataFrame을 Hugging Face Dataset으로 변환
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)  # 64 → 16
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    
    # 모델 및 토크나이저 로드
    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 데이터셋 로드 - trust_remote_code=True 추가
    
    def tokenize_function(example):
        texts = [str(text) if text is not None else "" for text in example["contents_final"]]
        return tokenizer(texts, truncation=True, padding=False)

    df_tmp_train = pd.read_csv('train_path', on_bad_lines='skip')
    df_tmp_test = pd.read_csv('test_path', on_bad_lines='skip')

    # Drop rows with missing values in critical columns
    df_tmp_train = df_tmp_train.dropna(subset=['contents_final', 'label'])
    df_tmp_test = df_tmp_test.dropna(subset=['contents_final', 'label'])

    print("train shape ->", df_tmp_train.shape)
    print("test shape ->", df_tmp_test.shape)

    dataset_train = Dataset.from_pandas(df_tmp_train)
    dataset_test = Dataset.from_pandas(df_tmp_test)
    tokenized_datasets_train = dataset_train.map(tokenize_function, batched=True)
    tokenized_datasets_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_datasets_test = dataset_test.map(tokenize_function, batched=True)
    tokenized_datasets_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    

    training_args = TrainingArguments(
        output_dir=os.environ.get("SM_MODEL_DIR", "./results"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=8,  # 16 → 8 (4 GPU이므로 effective batch = 32)
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        logging_steps=10,
        report_to=[],
        # Multi-GPU 및 메모리 최적화 설정
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        fp16=True,                      # Mixed precision (메모리 50% 절약)
        gradient_checkpointing=True,    # 메모리 절약
        gradient_accumulation_steps=2,  # 필요시 추가
        remove_unused_columns=True,     # 불필요한 컬럼 제거
        dataloader_num_workers=0,       # 메모리 절약
    )



    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        return {"accuracy": np.mean(predictions == labels)}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model()
    save_dir=os.environ.get("SM_MODEL_DIR", "./results")
    tokenizer.save_pretrained(save_dir)
    print(f"Model, tokenizer saved to: {save_dir}")



if __name__ == "__main__":
    main()




