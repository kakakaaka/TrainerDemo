# data_collator.py

class DataCollator:
    """
    数据整理器，将样本列表转换为批次数据。
    """
    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features):
        """
        将样本列表转换为批次数据。

        参数：
            features (List[Dict]): 样本列表，每个样本是一个字典。

        返回：
            批次数据，包含张量格式的输入和标签。
        """
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return batch
