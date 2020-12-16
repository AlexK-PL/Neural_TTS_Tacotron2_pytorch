import matplotlib.pyplot as plt
import numpy as np

training_files = 'filelists/ljs_audio_text_train_filelist.txt'

with open(training_files, encoding='utf-8') as f:
    training_audiopaths_and_text = [line.strip().split("|") for line in f]

L = len(training_audiopaths_and_text)
lengths = np.zeros([L])

for i in range(L):
    sentence = training_audiopaths_and_text[i][1]
    lengths[i] = len(sentence)
    print(len(sentence))

hist, bin_edges = np.histogram(lengths)
plt.hist(lengths)
plt.show()
