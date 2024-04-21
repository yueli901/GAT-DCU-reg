import tensorflow as tf

class MLP(tf.keras.Sequential):
    """ Multilayer perceptron. """
    def __init__(self, hiddens, act_type, out_act, weight_initializer=None, **kwargs):
        """
        The initializer.

        Parameters
        ----------
        hiddens: list
            The list of hidden units of each dense layer.
        act_type: str
            The activation function after each dense layer.
        out_act: bool
            Whether to apply activation function after the last dense layer.
        """
        super(MLP, self).__init__(**kwargs)
        for i, h in enumerate(hiddens):
            activation = None if i == len(hiddens) - 1 and not out_act else act_type
            self.add(tf.keras.layers.Dense(h, activation=activation, kernel_initializer=weight_initializer))


class MetaDense(tf.keras.layers.Layer):
    """ The meta-dense layer. """
    def __init__(self, input_hidden_size, output_hidden_size, meta_hiddens, **kwargs):
        """
        The initializer.

        Parameters
        ----------
        input_hidden_size: int
            The hidden size of the input.
        output_hidden_size: int
            The hidden size of the output.
        meta_hiddens: list of int
            The list of hidden units of the meta learner (a MLP).
        """
        super(MetaDense, self).__init__(**kwargs)
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.act_type = 'sigmoid'
        
        self.w_mlp = MLP(meta_hiddens + [self.input_hidden_size * self.output_hidden_size], act_type=self.act_type, out_act=False)
        self.b_mlp = MLP(meta_hiddens + [1], act_type=self.act_type, out_act=False)

    def call(self, feature, data):
        """ Forward process of a MetaDense layer

        Parameters
        ----------
        feature: Tensor with shape [n, d]
        data: Tensor with shape [n, b, input_hidden_size].

        Returns
        -------
        output: Tensor with shape [n, b, output_hidden_size]
        """
        weight = self.w_mlp(feature)  # [n, input_hidden_size * output_hidden_size]
        weight = tf.reshape(weight, (-1, self.input_hidden_size, self.output_hidden_size))
        bias = tf.reshape(self.b_mlp(feature), shape=(-1, 1, 1))  # [n, 1, 1]
        return tf.linalg.matmul(data, weight) + bias
