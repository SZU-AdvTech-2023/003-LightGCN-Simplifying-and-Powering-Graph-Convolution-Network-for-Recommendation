import numpy as np

def minibatch(*array,batch_size):
    '''

    :param tensors: tuple(user_id,pos_item,neg_item)
    :param batch_size: int
    :return:tuple(user_id,pos_item,neg_item)after split
    '''
    for i in range(0,len(array[0]),batch_size):
        yield tuple(x[i:i+batch_size] for x in array)

def shuffle(*array):
    '''
    :param tensors: tuple(user_id,pos_item,neg_item)
    :return: tuple(user_id,pos_item,neg_item)after shuffle
    '''
    if len(set(len(x) for x in array)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_index=np.arange(len(array[0]))
    np.random.shuffle(shuffle_index)

    result = tuple(x[shuffle_index] for x in array)
    return result