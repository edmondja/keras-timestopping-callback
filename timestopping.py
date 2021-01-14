import time, datetime
import numpy as np
from keras.callbacks import EarlyStopping

class TimeStopping(EarlyStopping):
    """Stop training when a specified amount of time has passed.
    Args:
        seconds: maximum amount of time before stopping.
            Defaults to 86400 (1 day).
        verbose: verbosity mode. Defaults to 0.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
        monitor: quantity to be monitored.
    """

    def __init__(self, seconds=86400, verbose=0, restore_best_weights=True, 
                 monitor='val_loss'):
        super(TimeStopping, self).__init__()
        self.seconds = seconds
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor

    def on_train_begin(self, logs=None):
        self.stopping_time = time.time() + self.seconds
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        if time.time() >= self.stopping_time:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights:
                if self.verbose > 0:
                    print('Restoring model weights from the end of '
                          'the best epoch (Timestopping)')
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            formatted_time = datetime.timedelta(seconds=self.seconds)
            msg = 'Timed stopping at epoch {} after training for {}'.format(
                self.stopped_epoch + 1, formatted_time)
            print(msg)

    def get_config(self):
        config = {
            'seconds': self.seconds,
            'verbose': self.verbose,
        }

        base_config = super(TimeStopping, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
