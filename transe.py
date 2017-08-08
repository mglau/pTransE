#coding:utf-8
import tensorflow as tf

        
class pTransE(object):
    """pTransE model. All three component objectives are simultaneously optimized."""
    def __init__(self, config, session):
        self._config = config
        self._sess = session
        self._ent_size = config.ent_size
        self._rel_size = config.rel_size
        self._words_size = config.words_size
        self._freq = config.freq
        self._vocab_size = self._ent_size + self._words_size
        self._build_input()
        self._build_var()
        self._build_train()
        self._build_anology_predict()
        print('Building graph done!')
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)


    def _build_input(self):
        self._pos_h = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._pos_t = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._pos_r = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._pos_ah = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._pos_at = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._pos_ar = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._v = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._w = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._av = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._aw = tf.placeholder(dtype=tf.int32, shape=[self._config.batch_size])
        self._analogy_a = tf.placeholder(dtype=tf.int32)
        self._analogy_b = tf.placeholder(dtype=tf.int32)
        self._analogy_c = tf.placeholder(dtype=tf.int32)


    def _build_var(self):
        # global embeddings for both words and entities
        self._vocab_emb = tf.get_variable(name="vocab_emb", shape=[self._vocab_size, self._config.emb_dim], 
            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        # Relation embedding: [rel_size, emb_dim]
        self._rel_emb = tf.get_variable(name="rel_emb", shape=[self._rel_size, self._config.emb_dim], 
            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        # words auxiliary embedding w': [words_size, emb_dim]
        self._words_aux_emb = tf.get_variable(name = "words_aux_emb", shape = [self._words_size, self._config.emb_dim], 
            initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        # Global step: scalar, i.e., shape []
        self.global_step = tf.Variable(0, name="global_step")


    def _forward_k_model(self, pos_h, pos_t, pos_r):
        """Build the graph for forwarding knowledge model."""
        # Negative sampling
        # Sample from an uniform distribution over I and replace the h, r, t respectively
        # We replicate sampled noise labels for all examples in the batch by using tf.expand_dims.
        # neg_[]: [num_samples]
        neg_h = tf.random_uniform([self._config.num_samples], minval=0, maxval=self._vocab_size, dtype=tf.int32)
        neg_t = tf.random_uniform([self._config.num_samples], minval=0, maxval=self._vocab_size, dtype=tf.int32)
        neg_r = tf.random_uniform([self._config.num_samples], minval=0, maxval=self._rel_size, dtype=tf.int32)
        # Embeddings for positive labels: [batch_size, emb_dim]
        pos_h_e = tf.nn.embedding_lookup(self._vocab_emb, pos_h)
        pos_t_e = tf.nn.embedding_lookup(self._vocab_emb, pos_t)
        pos_r_e = tf.nn.embedding_lookup(self._rel_emb, pos_r)
        # Embeddings for negative labels: [num_sampled, emb_dim]
        neg_h_e = tf.nn.embedding_lookup(self._vocab_emb, neg_h)
        neg_t_e = tf.nn.embedding_lookup(self._vocab_emb, neg_t)
        neg_r_e = tf.nn.embedding_lookup(self._rel_emb, neg_r)
        # Calculate the z function
        # True logits: [batch_size, 1]
        k_z_pos = self._config.margin - tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
        # Sampled logits z(h', r, t), z(h, r', t), z(h, r, t'): [batch_size, num_sampled, 1]
        k_z_h_neg = self._config.margin - tf.reduce_sum((tf.expand_dims(pos_r_e - pos_t_e, 1) + neg_h_e) ** 2, 2, keep_dims = True)
        k_z_t_neg = self._config.margin - tf.reduce_sum((tf.expand_dims(pos_h_e + pos_r_e, 1) - neg_t_e) ** 2, 2, keep_dims = True)
        k_z_r_neg = self._config.margin - tf.reduce_sum((tf.expand_dims(pos_h_e - pos_t_e, 1) + neg_r_e ) ** 2, 2, keep_dims = True)
        return k_z_pos, k_z_h_neg, k_z_t_neg, k_z_r_neg


    def _forward_t_model(self, w, v, AA):
        """Build the graph for forwarding text model."""
        pos_v = tf.reshape(tf.cast(v, dtype=tf.int64), [self._config.batch_size, 1])
        # Negative sampling
        # Sample from the unigram distribution raised to the 3/4rd power over V
        # We replicate sampled noise context words for all examples in the batch using the expand_dims.
        # neg_v: [num_samples]
        neg_v, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
                true_classes=pos_v,
                num_true=1,
                num_sampled=self._config.num_samples,
                unique=True,
                range_max=self._words_size,
                distortion=0.75,
                unigrams=list(self._freq.values())))
        # Text model only modify words_emb and words_aux_emb
        # Alignment model modify the whole vocab_emb
        if AA:
            # Embeddings for targets: [batch_size, emb_dim]
            w_e = tf.nn.embedding_lookup(self._vocab_emb, w)
            # Auxiliary embeddings for positive context words: [batch_size, emb_dim]
            pos_v_aux_e = tf.nn.embedding_lookup(self._vocab_emb, v)
            # Auxiliary embeddings for negative context words
            neg_v_aux_e = tf.nn.embedding_lookup(self._vocab_emb, neg_v)
        else:
            # Embeddings for targets: [batch_size, emb_dim]
            w_e = tf.nn.embedding_lookup(self._vocab_emb, w)
            # Auxiliary embeddings for positive context words: [batch_size, emb_dim]
            pos_v_aux_e = tf.nn.embedding_lookup(self._words_aux_emb, v)
            # Auxiliary embeddings for negative context words
            neg_v_aux_e = tf.nn.embedding_lookup(self._words_aux_emb, neg_v)
        # Calculate z function
        # True logits for text model: [batch_size, 1]
        t_z_pos = self._config.margin - tf.reduce_sum((w_e - pos_v_aux_e) ** 2, 1, keep_dims = True)
        # Sampled logits for text model: [batch_size, num_sampled, 1]
        t_z_neg = self._config.margin - tf.reduce_sum((tf.expand_dims(w_e, 1) - neg_v_aux_e) ** 2, 2, keep_dims = True)
        return t_z_pos, t_z_neg


    def _nce_loss(self, t_z_pos, t_z_neg, k_z_pos, k_z_h_neg, k_z_t_neg, k_z_r_neg):
        """
        Build the graph for the NCE loss.
        t_z_pos, k_z_pos: [batch_size, 1]
        t_z_neg, k_z_h_neg, k_z_t_neg, k_z_r_neg: [batch_size, num_sampled, 1]
        """
        # [batch_size, 1]
        true_t_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(t_z_pos), logits=t_z_pos)
        # [batch_size, num_sampled, 1]
        sampled_t_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(t_z_neg), logits=t_z_neg)
        # [batch_size, 1]
        true_k_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(k_z_pos), logits=k_z_pos)
        # [batch_size, num_sampled, 1]
        sampled_k_h_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(k_z_h_neg), logits=k_z_h_neg)
        sampled_k_r_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(k_z_r_neg), logits=k_z_r_neg)
        sampled_k_t_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(k_z_t_neg), logits=k_z_t_neg)
        # NCE-loss is the sum of the true and noise (sampled words) contributions, averaged over the batch.
        # [], scalar
        nce_loss = (tf.reduce_sum(true_t_xent) + tf.reduce_sum(true_k_xent) * 3 + tf.reduce_sum(sampled_k_h_xent) + 
            tf.reduce_sum(sampled_k_t_xent) + tf.reduce_sum(sampled_k_r_xent) + tf.reduce_sum(sampled_t_xent)) / self._config.batch_size
        return nce_loss


    def _optimize(self, loss):
        """Build the graph to optimize the loss function."""
        lr = self._config.learning_rate
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train = optimizer.minimize(loss)
        return train


    def _build_train(self):
        print('1')
        t_z_pos, t_z_neg = self._forward_t_model(self._w, self._v, False)
        print('2')
        at_z_pos, at_z_neg = self._forward_t_model(self._aw, self._av, True)
        print('3')
        k_z_pos, k_z_h_neg, k_z_t_neg, k_z_r_neg = self._forward_k_model(self._pos_h, self._pos_t, self._pos_r)
        print('4')
        ak_z_pos, ak_z_h_neg, ak_z_t_neg, ak_z_r_neg = self._forward_k_model(self._pos_ah, self._pos_at, self._pos_ar)
        print('5')
        nce_loss = self._nce_loss(t_z_pos, t_z_neg, k_z_pos, k_z_h_neg, k_z_t_neg, k_z_r_neg)
        print('6')
        nce_loss = nce_loss + self._nce_loss(at_z_pos, at_z_neg, ak_z_pos, ak_z_h_neg, ak_z_t_neg, ak_z_r_neg)
        self._train = self._optimize(nce_loss)
        self._loss = nce_loss


    def _build_anology_predict(self):
        nemb = tf.nn.l2_normalize(self._vocab_emb, 1)
        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, self._analogy_a)  # a's embs
        b_emb = tf.gather(nemb, self._analogy_b)  # b's embs
        c_emb = tf.gather(nemb, self._analogy_c)  # c's embs
        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)
        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, nemb, transpose_b=True)
        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)
        # [N, 4]
        self._analogy_pred_idx = pred_idx


    def batch_fit(self, pos_h, pos_t, pos_r, w, v, pos_ah, pos_at, pos_ar, aw, av):
        feed_dict = {self._pos_h: pos_h, self._pos_t: pos_t, self._pos_r: pos_r, self._w: w, self._v: v, 
        self._pos_ah: pos_ah, self._pos_at: pos_at, self._pos_ar: pos_ar, self._aw: aw, self._av: av}
        _, loss = self._sess.run([self._train, self._loss], feed_dict=feed_dict)
        return loss


    def analogy(self, a, b, c):
        """Predict the top 4 answers for analogy questions."""
        d = self._sess.run(self._analogy_pred_idx, 
            feed_dict={self._analogy_a: a, self._analogy_b: b, self._analogy_c: c})
        return d



        


















