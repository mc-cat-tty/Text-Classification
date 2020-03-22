import matplotlib.pyplot as plt
import numpy as np
import itertools
from threading import Thread, Lock
from queue import Queue
from classification_tools import *


TEST_FOLDER = os.path.join(DATASET, 'Test')
THREAD_NUM = 4

def plot_confusion_matrix(cm, labels, title='Confusion matrix', out_filename=None):

    accuracy = np.trace(cm) / float(np.sum(cm))  # Sum along diagonals / Total sum
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    threshold = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if out_filename:
        plt.savefig(out_filename)

    plt.show()

def main():  # Build and save confusion matrix
    labels = ['happiness', 'sadness']
    title = 'Confusion Matrix Normalized'
    data = {'happiness': {'happiness': 0, 'sadness': 0}, 'sadness': {'happiness': 0, 'sadness': 0}}
    happiness_test_folder = os.path.join(TEST_FOLDER, 'Happiness')
    sadness_test_folder = os.path.join(TEST_FOLDER, 'Sadness')
    Classificator.init_stopwords(STOPWORDS_MODEL_FILENAME)
    happiness_vocabulary = Vocabulary.load(os.path.join(MODELS, 'happiness_vocabulary'))
    sadness_vocabulary = Vocabulary.load(os.path.join(MODELS, 'sadness_vocabulary'))

    def compute_file(queue, lock):
        while True:
            d = queue.get()
            filename = d['filename']
            current_class = d['current_class']
            l = LabelledText.from_text_file(filename, [happiness_vocabulary, sadness_vocabulary], cleaning_level=HIGH, fast=True)
            with lock:
                data[current_class][l.get_label()] += 1
            queue.task_done()

    file_queue = Queue()
    data_lock = Lock()

    for i in range(THREAD_NUM):
        worker = Thread(target=compute_file, args=(file_queue, data_lock, ), name='worker{}'.format(i))
        worker.setDaemon(True)
        worker.start()

    for file in os.listdir(happiness_test_folder):
        file_queue.put({'filename': os.path.join(happiness_test_folder, file), 'current_class': happiness_vocabulary.label})
        # compute_file(os.path.join(happiness_test_folder, file), happiness_vocabulary.label)

    for file in os.listdir(sadness_test_folder):
        # compute_file(os.path.join(sadness_test_folder, file), sadness_vocabulary.label)
        file_queue.put({'filename': os.path.join(sadness_test_folder, file), 'current_class': sadness_vocabulary.label})

    file_queue.join()
    print(data)
    cm = np.array([list(data['happiness'].values()), list(data['sadness'].values())])
    plot_confusion_matrix(cm=cm, labels=labels, title=title, out_filename='confusion_matrix.png')


if __name__ == "__main__":
    main()
