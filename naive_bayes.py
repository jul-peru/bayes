import numpy as np


class BinaryNaiveBayes:
    delta = 1.0  # add to smoothe probabilities if we lack specific value

    def fit(self, X, y, delta=delta):
        """
        Fit a NaiveBayes classifier for two classes
        Parameters:
            X: [batch_size, vocab_size] of bag-of-words features
            y: [batch_size] of binary targets {0, 1}
        """

        # first, compute marginal probabilities of every class,
        # p(y=k) for k = 0, 1
        self.p_y = np.array(((np.count_nonzero(y == 0)/len(y)),
                             (np.count_nonzero(y == 1)/len(y))),
                            dtype=float)

        # count occurences of each word in texts with
        # label 1 and label 0 separately.
        # both must be vectors of shape [vocab_size]
        possitive_list = []
        negative_list = []
        for i in range(0, y.shape[0]):
            if y[i] == 1:
                possitive_list.append(i)
            else:
                negative_list.append(i)
        possitive_ind = np.take(X, possitive_list, axis=0)
        negative_ind = np.take(X, negative_list, axis=0)
        word_counts_positive = np.sum(possitive_ind, axis=0)
        word_counts_negative = np.sum(negative_ind, axis=0)

        # finally, lets use those counts to estimate p(x | y = k) for k = 0, 1
        # both must be of shape [vocab_size]
        self.p_x_given_positive = (word_counts_positive + delta)\
            / (np.sum(word_counts_positive) + len(word_counts_positive)
               * delta).sum()
        self.p_x_given_negative = (word_counts_negative + delta)\
            / (np.sum(word_counts_negative) + len(word_counts_negative)
               * delta).sum()

        return self

    def predict_scores(self):
        """
        Returns:
            a matrix of scores [batch_size, k] of scores for k-th class
        """
        p_x_given_negative = self.p_x_given_negative
        p_x_given_positive = self.p_x_given_positive
        p_y = self.p_y
        # compute scores for positive and negative classes separately.
        # these scores should be proportional to log-probabilities
        # of the respective target {0, 1}

        score_negative = np.log(p_y[0]) + np.sum(np.log(p_x_given_negative))
        score_positive = np.log(p_y[1]) + np.sum(np.log(p_x_given_positive))

        return np.stack([score_negative, score_positive], axis=-1)

    def predict(self, X):
        return self.predict_scores(X).argmax(axis=-1)
