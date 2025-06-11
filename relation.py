import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import random
import os

# 配置参数
class Config:
    # 使用另一个公开的中文T5模型
    model_name = "Langboat/mengzi-t5-base"  # 更换为mengzi-t5-base
    train_path = "train.json"  # 训练集路径
    test_path = "test1.json"    # 测试集路径
    output_path = "demo0.txt"   # 输出文件路径
    max_input_length = 128     # 输入最大长度
    max_target_length = 256    # 输出最大长度
    batch_size = 16  # 减小批次大小
    gradient_accumulation_steps = 4  # 添加梯度累积
    learning_rate = 2e-5
    num_epochs = 20
    warmup_ratio = 0.1        # warmup比例
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "hate_speech_model"  # 模型保存路径
    use_auth_token = False  # 是否使用认证token

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 自定义数据集类
class HateSpeechDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        for item in raw_data:
            self.data.append({
                'content': item['content'],
                'output': item['output']
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
        
        # 编码输出文本
        target_encoding = self.tokenizer(
            item['output'],
            max_length=Config.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 设置标签，将padding token设为-100以在计算损失时忽略
        labels = target_encoding.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')
    optimizer.zero_grad()  # 在epoch开始时清零梯度
    
    for idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 损失缩放
        loss = outputs.loss / Config.gradient_accumulation_steps
        total_loss += loss.item() * Config.gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (idx + 1) % Config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        progress_bar.set_postfix({'loss': loss.item() * Config.gradient_accumulation_steps})
    
    return total_loss / len(dataloader)

def predict(model, dataloader, tokenizer, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
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
    
    return predictions

def main():
    # 设置随机种子
    set_seed(Config.seed)
    
    # 初始化tokenizer和模型，添加错误处理
    print("Loading tokenizer...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            Config.model_name,
            trust_remote_code=True,  # 添加此参数
            local_files_only=False   # 允许在线下载
        )
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("\nTrying to download the model first...")
        # 如果在线加载失败，可以尝试先手动下载
        print("Please run the following commands in your terminal:")
        print("pip install huggingface_hub")
        print("huggingface-cli login")
        print(f"huggingface-cli download {Config.model_name}")
        return
        
    print("Loading model...")
    try:
        model = T5ForConditionalGeneration.from_pretrained(
            Config.model_name,
            trust_remote_code=True,  # 添加此参数
            local_files_only=False   # 允许在线下载
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 启用内存优化
    if hasattr(model, "config"):
        model.config.use_cache = False  # 禁用 KV 缓存以节省内存
    
    model.to(Config.device)
    
    # 准备数据集
    train_dataset = HateSpeechDataset(Config.train_path, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True
    )
    
    # 调整优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=0.01  # 添加权重衰减
    )
    
    # 考虑梯度累积后的实际步数
    total_steps = (len(train_dataloader) // Config.gradient_accumulation_steps) * Config.num_epochs
    warmup_steps = int(total_steps * Config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
        
        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        avg_loss = train_epoch(model, train_dataloader, optimizer, scheduler, Config.device)
        print(f"Average loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not os.path.exists(Config.save_path):
                os.makedirs(Config.save_path)
            model.save_pretrained(Config.save_path)
            tokenizer.save_pretrained(Config.save_path)
            print(f"Best model saved with loss: {best_loss:.4f}")
    
    # 加载最佳模型进行预测
    model = T5ForConditionalGeneration.from_pretrained(Config.save_path)
    model.to(Config.device)
    
    # 准备测试数据
    test_dataset = HateSpeechDataset(Config.test_path, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
    # 进行预测
    predictions = predict(model, test_dataloader, tokenizer, Config.device)
    
    # 保存预测结果
    with open(Config.output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            # 确保输出格式正确
            if '[END]' not in pred:
                pred = pred + ' [END]'
            f.write(pred + '\n')
    
    print(f"Predictions saved to {Config.output_path}")

if __name__ == "__main__":
    main()