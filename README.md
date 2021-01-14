TimeStopping callback for Keras.

It is similar to Early Stopping but it is based on the time spent rather than on the number of epochs passed.

This implementation is compatible with restore_best_weights and with the monitor parameter you can find in EarlyStopping. 
It is being clear about which stopping callback was triggered when restore_best_weights is set to True. 

It is based on https://github.com/tensorflow/addons/pull/757/commits/57c19081b1130f5dd02f60c2b3d1b61579632a57 and it works at least with keras 2.2.5 with tensorflow 1.15.4 as backend.
