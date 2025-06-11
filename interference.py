import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

class Config:
    model_path = "hate_speech_model"  # 训练好的模型保存路径
    test_path = "test1.json"    # 测试集路径
    output_path = "predictions.txt"   # 输出文件路径
    max_input_length = 128     # 输入最大长度
    max_target_length = 256    # 输出最大长度
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        for item in raw_data:
            self.data.append({
                'id': item['id'],
                'content': item['content']
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 编码输入文本
        input_encoding = self.tokenizer(
            item['content'],
            max_length=Config.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'id': item['id'],
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze()
        }

def predict(model, dataloader, tokenizer, device):
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_ids = batch['id']
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=Config.max_target_length,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True
            )
            
            decoded_preds = tokenizer.batch_decode(
                outputs, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            predictions.extend(decoded_preds)
            ids.extend(batch_ids)
    
    return ids, predictions

def main():
    print("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(Config.model_path)
    model = T5ForConditionalGeneration.from_pretrained(Config.model_path)
    model.to(Config.device)
    
    print("Preparing test dataset...")
    test_dataset = TestDataset(Config.test_path, tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False  # 不打乱顺序，保持ID对应
    )
    
    print("Generating predictions...")
    ids, predictions = predict(model, test_dataloader, tokenizer, Config.device)
    
    # 保存预测结果
    print(f"Saving predictions to {Config.output_path}")
    with open(Config.output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            # 确保输出格式正确
            if '[END]' not in pred:
                pred = pred + ' [END]'
            f.write(pred + '\n')
    
    # 同时保存带ID的JSON格式结果
    json_output_path = Config.output_path.replace('.txt', '_with_ids.json')
    results = []
    for id_, pred in zip(ids, predictions):
        if '[END]' not in pred:
            pred = pred + ' [END]'
        results.append({
            'id': id_,
            'output': pred
        })
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Predictions with IDs saved to {json_output_path}")

if __name__ == "__main__":
    main()