# main.py

from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
from training_args import TrainingArguments
from trainer import Trainer
from data_collator import DataCollator
from callbacks import LoggingCallback, EarlyStoppingCallback

def main():
    # 加载预训练的分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    # 加载数据集
    dataset = load_dataset('glue', 'mrpc')
    train_dataset = dataset['train']
    eval_dataset = dataset['validation']

    # 对数据集进行预处理
    def preprocess_function(examples):
        return tokenizer(
            examples['sentence1'],
            examples['sentence2'],
            truncation=True,
            max_length=128,
            padding='max_length'  # 在预处理时进行填充
        )

    # 使用 `map` 方法对数据集进行批量分词
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    # **重命名标签字段为 'labels'**
    train_dataset = train_dataset.rename_column('label', 'labels')
    eval_dataset = eval_dataset.rename_column('label', 'labels')

    # 设置数据格式
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # 定义数据整理器
    data_collator = DataCollator(tokenizer=tokenizer, padding=False)

    # 定义训练参数
    args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        logging_steps=50,
        save_steps=200,
        evaluation_strategy='steps',
        eval_steps=100,
        seed=42
    )

    # 定义回调函数列表
    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(patience=2)
    ]

    # 创建Trainer实例
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks
    )

    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main()
