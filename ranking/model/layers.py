import tensorflow as tf

def scaled_dot_product_attention(q,k,v,mask=None):
    # matmul要求相乘的高维矩阵的最后两个维度满足 二维矩阵相乘条件
    depth = tf.cast(q.shape[-1],tf.float32)         # q: (batch_size,num_heads,seq_len,depth)
    
    dot_product = tf.matmul(q,k,transpose_b=True)
    scaled_dot_product = dot_product/(tf.math.sqrt(depth))

    if mask is not None:
        scaled_dot_product += tf.cast(mask*-1e9, tf.float32)

    attention_weight = tf.nn.softmax(scaled_dot_product,axis=-1)

    scaled_attention = tf.matmul(scaled_dot_product,v)

    return scaled_attention, attention_weight

class PosEncoding(tf.layers.Layer):
    '''
    inputs: list([batch,seq_list,emb_dim],...,[batch,seq_list,emb_dim])
    '''
    def __init__(self, sess_max_cnt, seed=1024, **kwargs):
        self.sess_max_cnt = sess_max_cnt
        self.seed = seed
        super(PosEncoding,self).__init__(**kwargs)
    
    def build(self,input_shape):
        seq_len = input_shape[0][1]
        emb_dim = input_shape[0][2]
        
        self.sess_pos_embedding = tf.Variable(initial_value=tf.random_normal([self.sess_max_cnt,1,1], stddev=0.35, seed=self.seed),
                                        name='sess_pos_embedding',
                                        dtype=tf.float32,
                                        )
        self.seq_pos_embedding = tf.Variable(initial_value=tf.random_normal([1,seq_len,1], stddev=0.35, seed=self.seed),
                                        name='seq_pos_embedding',
                                        dtype=tf.float32,
                                        )
        self.item_pos_embedding = tf.Variable(initial_value=tf.random_normal([1,1,emb_dim], stddev=0.35, seed=self.seed),
                                        name='item_pos_embedding',
                                        dtype=tf.float32,
                                        )              
        super(PosEncoding,self).build(input_shape)   

    def call(self, inputs):
        mha_input = []
        for i in range(self.sess_max_cnt):
            mha_input.append(
                inputs[i] + self.seq_pos_embedding + self.item_pos_embedding + self.sess_pos_embedding[i]
            )   

        return mha_input

class MultiHeadAttention(tf.layers.Layer):
    '''
    inputs: [batch_size, seq_len, emb_dim]
    '''
    def __init__(self, head_num=8,seed=1024,supports_masking=False,**kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        self.head_num = head_num
        self.seed = seed
        self.supports_masking = supports_masking
    
    def build(self, input_shape):
        self.embedding_dim = input_shape[-1].value

        assert self.embedding_dim % self.head_num == 0
        self.att_embedding_dim = self.embedding_dim // self.head_num
        
        self.w_query = tf.Variable(initial_value=tf.random_normal([self.embedding_dim, self.att_embedding_dim*self.head_num], stddev=0.35, seed=self.seed),
                                   name='w_query',
                                   dtype=tf.float32,
                                   )
        self.w_key = tf.Variable(initial_value=tf.random_normal([self.embedding_dim, self.att_embedding_dim*self.head_num], stddev=0.35, seed=self.seed),
                                   name='w_key',
                                   dtype=tf.float32,
                                   )
        self.w_value = tf.Variable(initial_value=tf.random_normal([self.embedding_dim, self.att_embedding_dim*self.head_num], stddev=0.35, seed=self.seed),
                                   name='w_value',
                                   dtype=tf.float32,
                                   )

        super(MultiHeadAttention,self).build(input_shape)

    def call(self, inputs, seq_len, max_len=None): 
        # inputs = [batch_size, length, emb_dim]
        batch_size = tf.shape(inputs)[0]
        query = tf.matmul(inputs,self.w_query) 
        key = tf.matmul(inputs, self.w_key)
        value = tf.matmul(inputs, self.w_value)

        # split_head
        query = tf.reshape(query,(batch_size,-1,self.head_num,self.att_embedding_dim))
        query = tf.transpose(query, [0,2,1,3])

        key = tf.reshape(key,(batch_size,-1,self.head_num,self.att_embedding_dim))
        key = tf.transpose(key, [0,2,1,3])

        value = tf.reshape(value,(batch_size,-1,self.head_num,self.att_embedding_dim))
        value = tf.transpose(value, [0,2,1,3])

        mask = None
        if self.supports_masking:
            mask = 1-tf.cast(tf.sequence_mask(seq_len,maxlen=max_len),tf.float32)

        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.reshape(tf.transpose(scaled_attention,[0,2,1,3]), [batch_size,-1,self.embedding_dim])

        return scaled_attention

class SessionEncoderLayer(tf.layers.Layer):
    def __init__(self,hidden_nodes=128, seed=10, head_num=2,**kwargs):
        super(SessionExtractLayer,self).__init__(**kwargs)
        self.hidden_nodes = hidden_nodes
        self.seed = seed
        self.mha = MultiHeadAttention(head_num=head_num,seed=seed)

    def build(self,input_shape):
        self.embedding_dim = input_shape[-1].value
        self.w_fnn_layer1 = tf.Variable(initial_value=tf.random_normal([self.embedding_dim, self.hidden_nodes], stddev=0.35, seed=self.seed),
                                        name='w_ffn_layer1',
                                        dtype=tf.float32,
                                        )
        self.w_fnn_layer2 = tf.Variable(initial_value=tf.random_normal([self.hidden_nodes, self.embedding_dim], stddev=0.35, seed=self.seed),
                                        name='w_ffn_layer2',
                                        dtype=tf.float32,
                                        )
        super(SessionExtractLayer,self).build(input_shape)
    
    def call(self, inputs):
        atten_output = self.mha(inputs)
        LN_output1 = tf.contrib.layers.layer_norm(inputs + atten_output)

        ffn_output = tf.matmul(tf.matmul(LN_output1,self.w_fnn_layer1),self.w_fnn_layer2)
        LN_output2 = tf.contrib.layers.layer_norm(LN_output1 + ffn_output)

        return tf.reduce_mean(LN_output2,axis=1,keep_dims=True)

class TargetAttention(tf.layers.Layer):
    def __init__(self, sess_cnt=2,stag=None):
        super(TargetAttention,self).__init__()
        self.sess_cnt = sess_cnt
        self.stag = stag

    def call(self, inputs, query, seq_len=None, max_len=None):
        '''
        inputs: [batch_size, seq_len, emb_dim]
        query: [batch_size, emb_dim]
        '''
        sess_emb_dim = inputs.get_shape().as_list()[-1]
        sess_cnt = inputs.get_shape().as_list()[1]
        # sess_emb_dim = tf.shape(inputs)[-1]
        print(sess_cnt,sess_emb_dim)
        query = tf.layers.dense(query, sess_emb_dim)

        ## [batch_size, seq_len, emb_dim]
        query = tf.reshape(tf.tile(query,[1,sess_cnt]),[-1,sess_cnt,sess_emb_dim])

        target_attention_inputs = tf.concat([inputs, query, inputs-query],axis=-1)

        target_attention1 = tf.layers.dense(target_attention_inputs,80,activation=tf.nn.sigmoid)
        target_attention2 = tf.layers.dense(target_attention1,40,activation=tf.nn.sigmoid)
        target_attention_weights = tf.layers.dense(target_attention2,1,activation=tf.nn.sigmoid)  # [batch_size, seq_len, 1]
       
        target_attention_weights = tf.transpose(target_attention_weights,[0,2,1])  # [batch_size, 1, seq_len]

        if seq_len is not None:
            mask = 1 - tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32)
            target_attention_weights += mask[:, tf.newaxis, :]*-1e9

        target_attention_weights=tf.nn.softmax(target_attention_weights)

        output = tf.matmul(target_attention_weights, inputs)

        return output,target_attention_weights

# [batch_size, sess_len, behavior_len, emb_dim]
inputs = tf.constant([[[1,2,3,4],[5,6,7,8],[0,0,0,0]],[[1,2,3,4],[5,6,7,8],[1,0,0,0]]], dtype=tf.float32) ## [batch_size, seq_len, emb_dim]
query = tf.constant([[1,1,1],[2,2,2]], dtype=tf.float32)

ta = TargetAttention()
tmp, w = ta(inputs,query,[2,3])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tmp))

# ta = TargetAttention()
# tmp = ta(query=inputs, inputs=tf.transpose(inputs,[0,2,1]))


# tr_input=[]
# tr_input.append(inputs)
# tr_input.append(inputs)

# mha_input = pe(tr_input)

# tr_out = []
# for i in range(2):
#     tr_out.append(
#         sfe(mha_input[i])
#     )
# attention = sfe()

