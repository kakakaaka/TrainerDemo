# training_args.py

from dataclasses import dataclass, field

@dataclass
class TrainingArguments:
    """
    训练参数类，包含训练过程中的各种配置。
    """
    output_dir: str = field(
        default="output",
        metadata={"help": "模型和检查点的输出目录"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "训练的总轮数"}
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "训练时每个设备的批次大小"}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "评估时每个设备的批次大小"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "学习率"}
    )
    logging_steps: int = field(
        default=500,
        metadata={"help": "记录日志的步数间隔"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "保存检查点的步数间隔"}
    )
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "评估策略：'no'、'steps'、'epoch'"}
    )
    eval_steps: int = field(
        default=None,
        metadata={"help": "评估的步数间隔，如果evaluation_strategy为'steps'时生效"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    # 可以根据需要添加更多参数
