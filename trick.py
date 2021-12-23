from math import sqrt,ceil
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
import tensorflow as tf
def plot_ckalist_resume(cka_list,save_name):
    n = len(cka_list)
    y = ceil(sqrt(n))
    if n == sqrt(n)*sqrt(n):
        x = y
    elif (y-1) * y < n:
        x = y
    else:
        x = y - 1
    print("x | y :",x,y)    
    fig = plt.figure(figsize=(y*4,x*4),frameon=False)
    #plt.title(save_name)
    sc = None
    for i,cka in enumerate(cka_list):
        ax = fig.add_subplot(x,y,i+1)
        ll = cka.shape[0]
        sc = ax.imshow(cka, cmap='magma', vmin=0.0,vmax=1.0)
        tick = [i for i in range(0,ll,int(ll/5))]
        ax.set_xticks(tick) 
        tick.reverse()
        ax.set_yticks([]) 
        ax.axes.invert_yaxis()
    
    l = 0.92
    b = 0.35
    w = 0.015
    h = 0.35
    #对应 l,b,w,h；设置colorbar位置；
    rect = [l,b,w,h] 
    cbar_ax = fig.add_axes(rect) 

    plt.colorbar(sc,cax = cbar_ax)
    #plt.suptitle('the number of layers',fontsize = 30, color = 'black')
    #fig.tight_layout()  
    plt.savefig('{}.png'.format(save_name),dpi=700)  
def plot_probelist_resume(p,save_name):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot([i+1 for i in range(len(p))],[i for i in p],color='red',marker='o',markersize=6)
    plt.ylabel('Accuracy',fontsize=14)
    plt.xlabel('Layer',fontsize=14)
    plt.ylim(0.1,1)
    x_major_locator=MultipleLocator(int(len(p)/5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')    # top边框属性设置为none 不显示
    plt.savefig('{}.png'.format(save_name),dpi=700) 
class MinibatchCKA(tf.keras.metrics.Metric):

  def __init__(self,
               num_layers,
               num_layers2=None,
               across_models=False,
               dtype=tf.float32):
    super(MinibatchCKA, self).__init__()
    if num_layers2 is None:
      num_layers2 = num_layers
    self.hsic_accumulator = self.add_weight(
        'hsic_accumulator',
        shape=(num_layers, num_layers2),
        initializer=tf.keras.initializers.zeros,
        dtype=dtype)
    self.across_models = across_models
    if across_models:
      self.hsic_accumulator_model1 = self.add_weight(
          'hsic_accumulator_model1',
          shape=(num_layers,),
          initializer=tf.keras.initializers.zeros,
          dtype=dtype)
      self.hsic_accumulator_model2 = self.add_weight(
          'hsic_accumulator_model2',
          shape=(num_layers2,),
          initializer=tf.keras.initializers.zeros,
          dtype=dtype)

  def _generate_gram_matrix(self, x):
    """Generate Gram matrix and preprocess to compute unbiased HSIC.

    This formulation of the U-statistic is from Szekely, G. J., & Rizzo, M.
    L. (2014). Partial distance correlation with methods for dissimilarities.
    The Annals of Statistics, 42(6), 2382-2412.

    Args:
      x: A [num_examples, num_features] matrix.

    Returns:
      A [num_examples ** 2] vector.
    """
    x = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(x, x, transpose_b=True)
    n = tf.shape(gram)[0]
    gram = tf.linalg.set_diag(gram, tf.zeros((n,), gram.dtype))
    gram = tf.cast(gram, self.hsic_accumulator.dtype)
    means = tf.reduce_sum(gram, 0) / tf.cast(n - 2, self.hsic_accumulator.dtype)
    means -= tf.reduce_sum(means) / tf.cast(2 * (n - 1),
                                            self.hsic_accumulator.dtype)
    gram -= means[:, None]
    gram -= means[None, :]
    gram = tf.linalg.set_diag(gram, tf.zeros((n,), self.hsic_accumulator.dtype))
    gram = tf.reshape(gram, (-1,))
    return gram

  def update_state(self, activations):
    """Accumulate minibatch HSIC values.

    Args:
      activations: A list of activations for all layers.
    """
    # tf.assert_equal(
    #     tf.shape(self.hsic_accumulator)[0], len(activations),
    #     'Number of activation vectors does not match num_layers.')
    layer_grams = [self._generate_gram_matrix(x) for x in activations]
    layer_grams = tf.stack(layer_grams, 0)
    self.hsic_accumulator.assign_add(
        tf.matmul(layer_grams, layer_grams, transpose_b=True))

  def update_state_across_models(self, activations1, activations2):
    """Accumulate minibatch HSIC values from different models.

    Args:
      activations1: A list of activations for all layers in model 1.
      activations2: A list of activations for all layers in model 2.
    """
    tf.assert_equal(
        tf.shape(self.hsic_accumulator)[0], len(activations1),
        'Number of activation vectors does not match num_layers.')
    tf.assert_equal(
        tf.shape(self.hsic_accumulator)[1], len(activations2),
        'Number of activation vectors does not match num_layers.')
    layer_grams1 = [self._generate_gram_matrix(x) for x in activations1]
    layer_grams1 = tf.stack(layer_grams1, 0)  #(n_layers, n_examples ** 2)
    layer_grams2 = [self._generate_gram_matrix(x) for x in activations2]
    layer_grams2 = tf.stack(layer_grams2, 0)
    self.hsic_accumulator.assign_add(
        tf.matmul(layer_grams1, layer_grams2, transpose_b=True))
    self.hsic_accumulator_model1.assign_add(
        tf.einsum('ij,ij->i', layer_grams1, layer_grams1))
    self.hsic_accumulator_model2.assign_add(
        tf.einsum('ij,ij->i', layer_grams2, layer_grams2))

  def result(self):
    mean_hsic = tf.convert_to_tensor(
        self.hsic_accumulator)  #(num_layers, num_layers2)
    if self.across_models:
      normalization1 = tf.sqrt(
          tf.convert_to_tensor(self.hsic_accumulator_model1))  #(num_layers,)
      normalization2 = tf.sqrt(
          tf.convert_to_tensor(self.hsic_accumulator_model2))  #(num_layers2,)
      mean_hsic /= normalization1[:, None]
      mean_hsic /= normalization2[None, :]
    else:
      normalization = tf.sqrt(tf.linalg.diag_part(mean_hsic))
      mean_hsic /= normalization[:, None]
      mean_hsic /= normalization[None, :]
    return mean_hsic
