import time

import evaluate
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification, EvalPrediction, EarlyStoppingCallback

from configuration.config import *

# 1. 分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 标签映射
id2label = { id:label for id, label in enumerate(LABELS) }
label2id = { label:id for id, label in enumerate(LABELS) }

# 2. 模型
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
)

# 3. 加载数据集
train_dataset = load_from_disk( PROCESSED_DATA_DIR / 'train' )
valid_dataset = load_from_disk( PROCESSED_DATA_DIR / 'valid' )

# 4. 数据整理器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt"
)

# 5. 训练参数
args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR / NER_DIR),
    logging_dir=str(LOG_DIR / NER_DIR / time.strftime("%Y-%m-%d-%H-%M-%S")),

    num_train_epochs=EPOCHS,    # 训练总轮次
    per_device_train_batch_size=BATCH_SIZE, # 批大小

    save_strategy="steps",      # 保存策略
    save_steps=SAVE_STEPS,      # 每20次迭代进行一次保存
    save_total_limit = 3,       # 最多保存3个检查点

    fp16=True,                  # 开启混合精度训练

    logging_strategy = 'steps', # 日志写入策略
    logging_steps= SAVE_STEPS,

    eval_strategy = 'steps',    # 评估策略
    eval_steps = SAVE_STEPS,

    metric_for_best_model = 'eval_overall_f1', # 模型评估指标
    greater_is_better = True,
    load_best_model_at_end = True   # 训练结束加载最佳模型
)

# 6. 评估指标函数
seqeval = evaluate.load("seqeval")

def compute_metrics(prediction: EvalPrediction):
    # 提取模型的预测输出和真实标签
    logits = prediction.predictions
    preds = logits.argmax(axis=-1)  # 预测分类标签
    labels = prediction.label_ids   # 真实分类标签
    # 将标签id转换为真正的标注标签BIO
    unpad_labels = []
    unpad_preds = []
    for pred, label in zip(preds, labels):
        # 去掉填充对应的id
        unpad_label = label[ label != -100 ]
        unpad_pred = pred[ label != -100 ]
        # 转BIO标签
        unpad_pred = [ id2label[id] for id in unpad_pred ]
        unpad_label = [ id2label[id] for id in unpad_label ]
        # 添加到列表
        unpad_labels.append(unpad_label)
        unpad_preds.append(unpad_pred)

    result = seqeval.compute(predictions=unpad_preds, references=unpad_labels)
    return result

# 7. 早停回调
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=50)


# 创建训练器
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

# 训练
trainer.train()

# 模型保存
trainer.save_model( CHECKPOINT_DIR / NER_DIR / 'best_model' )