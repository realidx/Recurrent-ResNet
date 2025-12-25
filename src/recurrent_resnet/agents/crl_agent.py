class CRLAgent:
    def __init__(self, config) -> None:
        self.config = config

    def act(self, observation):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError
