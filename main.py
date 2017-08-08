import tensorflow as tf
import os
from tqdm import tqdm
import random
import numpy as np
import nltk
from collections import Counter, deque, OrderedDict
from ptranse import pTransE
import re
import sys
from sklearn import metrics


dir_path = os.path.dirname(os.path.realpath(__file__))
object_file = os.path.join(dir_path, 'mappingbased_objects_en.ttl')
corpus_file = os.path.join(dir_path, 'wiki2')
analogy_file = os.path.join(dir_path, 'questions-words.txt')

flags = tf.flags
flags.DEFINE_string("object_file", object_file, "Object facts of DBpedia.")
flags.DEFINE_string("corpus_file", corpus_file, "Corpus file.")
flags.DEFINE_string("analogy_file", analogy_file, "Analogy file.")
flags.DEFINE_integer("margin", 7, "Sensible parameter for TransE.")
flags.DEFINE_integer("embedding_size", 100, "The embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 15, "Number of epochs to train. Each epoch processes the training data once completely.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 4, "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 64, "Number of training v processed per step (size of a minibatch).")
flags.DEFINE_integer("window_size", 5, "The number of words to predict to the left and right of the target word.")
flags.DEFINE_integer("min_count", 3, "The minimum number of word occurrences for it to be included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3, "Subsample threshold for word occurrence. Words that appear with higher frequency will be randomly down-sampled. Set to 0 to disable.")
flags.DEFINE_string("alignment", "AA_AN", "You can pick AA/AN/AA_AN.")
flags.DEFINE_integer("statistics_interval", 1, "Print statistics every n epochs.")
flags.DEFINE_integer("summary_interval", 5, "Save training summary to file every n seconds (rounded up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600, "Checkpoint the model (i.e. save the parameters) every n seconds (rounded up to statistics interval).")
FLAGS = flags.FLAGS


class Config(object):
    """Configurations used by the model."""
    def __init__(self):
        # Embedding dimension
        self.emb_dim = FLAGS.embedding_size
        # The training files
        self.object_file = FLAGS.object_file
        self.corpus_file = FLAGS.corpus_file
        # Number of negative samples per example
        self.num_samples = FLAGS.num_neg_samples
        # The initial learning rate
        self.learning_rate = FLAGS.learning_rate
        # Number of epochs to train
        # Traverse the whole knowledge base once at every epoch.
        self.epochs_to_train = FLAGS.epochs_to_train
        self.statistics_interval = FLAGS.statistics_interval
        # Batch size
        self.batch_size = FLAGS.batch_size
        # The number of words to predict to the left and right of the target word
        self.window_size = FLAGS.window_size
        # The minimum number of word occurrences for it to be included in the vocabulary
        self.min_count = FLAGS.min_count
        # Subsampling threshold for word occurrence
        self.subsample = FLAGS.subsample
        # Sensible choice b
        self.margin = FLAGS.margin
        # Alignment mechanisms
        self.alignment = FLAGS.alignment
        # Number of entities
        self.ent_size = 0
        # Number of relations
        self.rel_size = 0
        # Number of unique words
        self.words_size = 0
        # Number of facts
        self.triples_size = 0
        self.namegraph_size = 0
        # Number of words in corpus
        self.corpus_size = 0
        # Counter for vocabulary
        self.freq = {}
        # Dictionary for vocabulary
        self.word_id = {}
        self.id_word = {}


def build_k_dataset(object_file, words_size):
    """Process triples into a dataset."""
    entity_id = {}
    relation_id = {}
    triples = []
    j, k = words_size, 0
    _ = 0
    total = 17
    done = 0
    with open(object_file) as of:
        for i, line in enumerate(of):
            line = line.split()
            h, r, t = line[0], line[1], line[2]
            if h not in entity_id:
                entity_id[h] = j
                j += 1
            if t not in entity_id:
                entity_id[t] = j
                j += 1
            if r not in relation_id:
                relation_id[r] = k
                k += 1
            triples.append([entity_id[h], relation_id[r], entity_id[t]])
            _ += 1
            if _ == 1e6:
                _ = 0
                done += 1
                print('{}M trples done! {}M to go!'.format(done, total-done))
    id_entity = dict(zip(entity_id.values(), entity_id.keys()))
    return entity_id, id_entity, relation_id, triples


def build_t_dataset(corpus_file, mincount):
    """
    Process raw text into a dataset.
    The universal embedding: [entities, words].
    So the start index of words is ent_eize.
    """
    with open(corpus_file) as f:
        words = f.read().split()
    # unique words
    count = Counter(words)
    unique_size = len(count)
    # frequent unique words
    word_id = {}
    freq = {}
    for word in count:
        if count[word] >= mincount:
            id_ = len(word_id) + 1
            word_id[word] = id_
            freq[id_] = count[word]
    data = []
    unk_count = 0
    for word in words:
        if word in word_id:
            index = word_id[word]
        else:
            index = 0  # word_id['UNK']
            unk_count += 1
        data.append(index)
    del words
    word_id['UNK'] = 0
    freq[0] = unk_count
    id_word = dict(zip(word_id.values(), word_id.keys()))
    return data, freq, word_id, id_word, unique_size


def build_analogy_dataset(analogy_file, word_id):
    ana_question = []
    with open(analogy_file) as af:
        for line in af:
            l = [w.lower() for w in line.split()]
            if l[0] != ':':
                if l[0] in word_id and l[1] in word_id and l[2] in word_id and l[3] in word_id:
                    ana_question.append([word_id[l[0]], word_id[l[1]], word_id[l[2]], word_id[l[3]]])
    return ana_question


def generate_t_batch(data, batch_size, 
    subsample, skip_window, freq, corpus_size):
    '''
    Since we want to traverse whole dataset, we need sample's index.
    '''
    global t_word_index
    w = []
    v = []
    span = 2 * skip_window + 1  # [skip_window target skip_window]
    buffer = deque(maxlen=span)
    for i in range(span):
        buffer.append(data[t_word_index])
        t_word_index = (t_word_index + 1) % corpus_size
    i = 0
    while i < batch_size:
        context = list(range(span))
        context.remove(skip_window) # remove target from context
        for j in context:
            c = buffer[j]
            if sub_sample(subsample, c, corpus_size, freq):
                continue
            w.append(buffer[skip_window])
            v.append(c)
            i += 1
            if i == batch_size:
                break
        buffer.append(data[t_word_index])
        t_word_index = (t_word_index + 1) % corpus_size
    return w, v


def sub_sample(subsample, word, corpus_size, freq):
    if subsample > 0:
        word_freq = freq[word]
        keep_prob = (np.sqrt(word_freq / (subsample * corpus_size)) + 1) * subsample * corpus_size / word_freq
        if random.random() > keep_prob:
            return True
    return False


def generate_AA_batch(entity_id, id_word, batch_size, skip_window, corpus_size):
    global aa_word_index
    w = []
    ev = []
    span = 2 * skip_window + 1  # [skip_window target skip_window]
    buffer = deque(maxlen=span)
    for i in range(span):
        buffer.append(data[aa_word_index])
        aa_word_index = (aa_word_index + 1) % corpus_size
    i = 0
    while i < batch_size:
        context = list(range(span))
        context.remove(skip_window) # remove target from context
        for j in context:
            c = buffer[j]
            entity_name = id_word[c]
            entity = entityname_to_entity(entity_name)
            if entity in entity_id:
                w.append(buffer[skip_window])
                ev.append(entity_id[entity])
                i += 1
                if i == batch_size:
                    break
        buffer.append(data[aa_word_index])
        aa_word_index = (aa_word_index + 1) % corpus_size
    return w, ev


def entityname_to_entity(entity_name):
    entity = '<http://dbpedia.org/resource/{}>'.format(entity_name.title())
    return entity


def build_namegraph(triples, id_entity, word_id):
    namegraph = []
    for h, r, t in triples:
        wh = entity_to_entityname(id_entity[h])
        if wh in word_id:
            namegraph.append([word_id[wh], r, t])
        wt = entity_to_entityname(id_entity[t])
        if wt in word_id:
            namegraph.append([h, r, word_id[wt]])
        if wt in word_id and wh in word_id:
            namegraph.append([word_id[wh], r, word_id[wt]])
    return namegraph


def entity_to_entityname(entity):
    p = re.compile(r'<http://dbpedia.org/resource/(\w+[a-zA-Z0-9]+)(.*)>')
    e = p.search(entity)
    if e:
        ee = e.group(1)
        return re.sub(r'\_+', ' ', ee).lower()
    else:
        return None


config = Config()
data, freq, word_id, id_word, unique_size = build_t_dataset(FLAGS.corpus_file, FLAGS.min_count)
freq = OrderedDict(sorted(freq.items()))
words_size = len(word_id)
corpus_size = len(data)
config.corpus_size = corpus_size
config.freq = freq
config.words_size = words_size
print('There are', corpus_size, 'words,', unique_size, 'unique words,', words_size, 'unique frequent words.')
entity_id, id_entity, relation_id, triples = build_k_dataset(FLAGS.object_file, words_size)
triples_size = len(triples)
ent_size = len(entity_id)
rel_size = len(relation_id)
config.ent_size = ent_size
config.rel_size = rel_size
config.triples_size = triples_size
print('There are', triples_size, 'triples,', ent_size, 'unique entities,', rel_size, 'unique relations.')
ana_question = build_analogy_dataset(FLAGS.analogy_file, word_id)
q_size = len(ana_question)
print('There are', q_size, 'analogy questions.')
namegraph = build_namegraph(triples, id_entity, word_id)
namegraph_size = len(namegraph)
config.namegraph_size = namegraph_size
print('There are', namegraph_size, 'new edges being added to the namegraph.')
namegraph_batches = zip(range(0, namegraph_size - config.batch_size, config.batch_size),
          range(config.batch_size, namegraph_size, config.batch_size))
namegraph_batches = [(start, end) for start, end in namegraph_batches]
nngbatch = len(namegraph_batches)
knowledge_batches = zip(range(0, triples_size - config.batch_size, config.batch_size),
              range(config.batch_size, triples_size, config.batch_size))
knowledge_batches = [(start, end) for start, end in knowledge_batches]
nbatch = len(knowledge_batches)
print('There are', nbatch, 'batches of triples in one epoch.')


t_word_index = 0
aa_word_index = 0
ng_index = 0
with tf.Session() as sess:
    pTransE = pTransE(config, sess)
    for times in range(config.epochs_to_train):
        np.random.shuffle(knowledge_batches)
        np.random.shuffle(namegraph_batches)
        # Since we have four different models and four different training sets, 
        # we need to define a main model to traverse. In here, we define the knowledge model as main model.
        for i in range(nbatch):
            # generate batch for knowledge model
            start, end = knowledge_batches[i][0], knowledge_batches[i][1]
            k_batch = np.asarray(triples[start:end])
            h = k_batch[:, 0]
            r = k_batch[:, 1]
            t = k_batch[:, 2]
            # generate batch for text model
            w, v = generate_t_batch(data, config.batch_size, 
                config.subsample, config.window_size, freq, corpus_size)
            # generate batch for AN alignment model
            start, end = namegraph_batches[ng_index][0], namegraph_batches[ng_index][1]
            ng_index = (ng_index + 1) % nngbatch
            a_batch = np.asarray(namegraph[start:end])
            ah = a_batch[:, 0]
            ar = a_batch[:, 1]
            at = a_batch[:, 2]
            # generate batch for AA alignment model
            aw, av = generate_AA_batch(entity_id, id_word, config.batch_size, 
                config.window_size, corpus_size)
            loss = pTransE.batch_fit(h, t, r, w, v, ah, at, ar, aw, av)
        if times % config.statistics_interval == 0:
            print(loss)
            # top 20 results
            ana_question = np.asarray(ana_question)
            q_batch = random.sample(range(0, q_size), 30)
            a = ana_question[:, 0][q_batch]
            b = ana_question[:, 1][q_batch]
            c = ana_question[:, 2][q_batch]
            d = ana_question[:, 3][q_batch]
            d_pred = pTransE.analogy(a, b, c)
            acc = metrics.accuracy_score(d, d_pred[:, 0])
            print('top1 acc:', acc)
            acc_count = 0
            for i in range(10):
                if d[i] in d_pred[i, :]:
                  acc_count += 1
            top20_acc = acc_count / 10
            print('top20 acc:', top20_acc)
            for i in range(10):
                a_word = id_word[a[i]]
                b_word = id_word[b[i]]
                c_word = id_word[c[i]]
                d_word = id_word[d[i]]
                if d_pred[:, 0][i] in id_word:
                   pred_d_word = id_word[d_pred[:, 0][i]]
                else:
                   pred_d_word = id_entity[d_pred[:, 0][i]]
                print(a_word, b_word, c_word, 'correct:'+d_word, 'pred:'+pred_d_word)
