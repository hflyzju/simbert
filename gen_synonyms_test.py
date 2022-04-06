#! -*- coding: utf-8 -*-
# SimBERT base 基本例子
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7

import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.snippets import uniout
from keras.layers import *
from simbert import *

# maxlen = 32

# # bert配置
# bert_dirt = '/root/huxiang/data/'
# config_path = bert_dirt + 'bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = bert_dirt + 'bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = bert_dirt + 'bert/chinese_L-12_H-768_A-12/vocab.txt'

# # 建立分词器
# tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# # 建立加载模型
# bert = build_transformer_model(
#     config_path,
#     checkpoint_path,
#     with_pool='linear',
#     application='unilm',
#     return_keras_model=False,
# )

# encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
# seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        """拿到每个句子的embedding，用于做相似度计算和排序"""
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate(
            [segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        """根据text生成相似句子"""
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(start_id=None,
                                       end_id=tokenizer._token_end_id,
                                       maxlen=maxlen)


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    # 1. 随机采样生成n个句子
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text] # 去重
    r = [text] + r # 加上原始句子
    # 2. 拿到每个句子的token和segment
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    # 3. 预测每个句子的embedding
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    # 4. 与原始text做点积后排序
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    # 5. 取k个最相似的句子
    return [r[i + 1] for i in argsort[:k]]




# print("input:糖尿病的早期症状有哪些？")
# print("output:")
# print(gen_synonyms(u'糖尿病的早期症状有哪些？'))



print("input:糖尿病吃什么食物")
print("output:")
print(gen_synonyms(u'糖尿病吃什么食物'))