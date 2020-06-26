# Facial_Expression_Recognition
<p>Trained on a dataset published in Kaggle compitition and got around 65% accuracy(71% is accuracy of winner)</p>
<p>Dataset contains Imbalenced data on 7 Universal expressions</p>
<p>Done 2 resognition system one with inbuilt 'categorical_crossentropy' loss and another with custom 'weighted_categorical_crossentropy' to counter class Imbalence</p>
<img src='ClassFrequency.png'>
'''python
def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
  def weighted_loss(y_true,y_pred):
    loss = 0.0
    for i in range(len(pos_weights)):
      loss += K.mean(-1*(pos_weights[i]*y_true[:,i]*K.log(y_pred[:,i]+epsilon)+neg_weights[i]*(1-y_true[:,i])*K.log(1-y_pred[:,i]+epsilon)))
    return loss
  return weighted_loss
'''
</code>
Facial expression recognition system with custom loss function to counter class Imbalance problem 
