from treetune.trainers.base_trainer import Trainer


@Trainer.register("empty")
class EmptyTrainer(Trainer):
    def __init__(self, **kwargs):
        pass
