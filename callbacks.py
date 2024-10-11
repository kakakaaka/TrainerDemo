# callbacks.py

class TrainerCallback:
    """
    回调函数的基类。
    """
    def on_step_end(self, trainer, loss, global_step):
        pass

class LoggingCallback(TrainerCallback):
    """
    日志记录回调，在每个步骤结束时记录损失值。
    """
    def on_step_end(self, trainer, loss, global_step):
        if global_step % trainer.args.logging_steps == 0:
            print(f"[LoggingCallback] Step {global_step}: loss = {loss.item()}")

# callbacks.py
class EarlyStoppingCallback(TrainerCallback):
    """
    提前停止回调，当验证损失不再下降时停止训练。
    """
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float('inf')
        self.num_bad_epochs = 0

    def on_evaluate(self, trainer, eval_loss):
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                print("Early stopping triggered.")
                trainer.should_stop = True  # 设置should_stop为True

