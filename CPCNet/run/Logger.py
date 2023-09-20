from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def add_scalar(self, name, scalar, epoch):
        self.writer.add_scalar(name, scalar, epoch)
        self.writer.flush()

    def add_scalars(self, name, scalars, epoch):
        self.writer.add_scalars(name, scalars, epoch)
        self.writer.flush()

    def add_text(self, title, content):
        self.writer.add_text(title, content)
        self.writer.flush()

    def add_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)
        self.writer.flush()

    def close(self):
        self.writer.close()
