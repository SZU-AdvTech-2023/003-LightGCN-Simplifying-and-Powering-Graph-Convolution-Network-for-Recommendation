import numpy as np

def minibatch(*tensors,batch_size):
    '''

    :param tensors: tuple(user_id,pos_item,neg_item)
    :param batch_size: int
    :return:tuple(user_id,pos_item,neg_item)after split
    '''
    for i in range(0,len(tensors[0]),batch_size):
        yield tuple(x[i:i+batch_size] for x in tensors)

def shuffle(*tensors):
    '''
    :param tensors: tuple(user_id,pos_item,neg_item)
    :return: tuple(user_id,pos_item,neg_item)after shuffle
    '''
    if len(set(len(x) for x in tensors)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_index=np.arange(len(tensors[0]))
    np.random.shuffle(shuffle_index)

    result = tuple(x[shuffle_index] for x in tensors)
    return result