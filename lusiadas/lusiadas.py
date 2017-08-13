import tflearn
from tflearn.data_utils import *
import time

path = "lusiadas.txt"
maxlen = 120
EPOCH = 0

print('Start..')

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

print('Creating Model...')
g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.01)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_luis')
print('Done.')

def train():
    print('Train...')
    for i in range(50):
        seed = random_sequence_from_textfile(path, maxlen)
        m.fit(X, Y, validation_set=0.1, batch_size=128,
              n_epoch=1, run_id='luis')
        save()
        generate(seed)
        EPOCH += 1
        

def save():
    filename = 'models/epoch-{}-'.format(EPOCH)
    filename += time.strftime("%d%m%Y-%H%M%S") + '.model'
    m.save(filename)
    print('SAVED ' + filename)

def generate(seed):
    filename = '/epoch-{}-'.format(EPOCH)
    filename += time.strftime("%d%m%Y-%H%M%S") + '.txt'
    with open('seeds/' + filename, 'w') as f:
        f.write(seed)
        f.close()
    with open('temp1/' + filename, 'w') as f:
        gen = m.generate(500, temperature=1.0, seq_seed=seed)
        f.write(gen)
        f.close()
    with open('temp0.5/' + filename, 'w') as f:
        gen = m.generate(500, temperature=0.5, seq_seed=seed)
        f.write(gen)
        f.close()


train()

    
