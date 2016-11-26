class Config(object):
    learning_rate = 1e-3
    max_grad_norm = 10
    num_layers = 1
    num_steps = 20
    hidden_size = 300
    max_epoch = 100
    keep_prob = 0.5
    batch_size = 20
    vocab_size = 10000
    nr_epoch_first_stage = 40
    nr_epoch_second_stage = 80
    w_bit = 2
    f_bit = 2
    cell_type = 'gru'
