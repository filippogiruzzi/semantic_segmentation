import tensorflow as tf

from semseg.training.model import UNet


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))
        losses = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        return tf.reduce_mean(losses, axis=-1)

    return loss


class SemsegEstimator(object):
    def __init__(self, params):
        self._instantiate_model(params)

    def _instantiate_model(self, params, training=False):
        if params['model'] == 'unet':
            self.model = UNet(params=params, is_training=training)

    def _output_network(self, features, params, training=False):
        self._instantiate_model(params=params, training=training)
        output = self.model(inputs=features)
        return output

    @staticmethod
    def loss_fn(labels, predictions, params):
        pred = predictions['semseg']
        label = labels['label']
        one_hot_label = tf.one_hot(label, depth=params['n_classes'])

        loss_func = focal_loss(alpha=0.25, gamma=2)
        losses = loss_func(y_true=one_hot_label, y_pred=pred)

        loss_mask = label > 0
        masked_loss = tf.where(loss_mask, losses, tf.zeros_like(losses, dtype=tf.float32))
        loss = tf.reduce_sum(masked_loss, axis=(1, 2))
        loss = tf.reduce_mean(loss)

        tf.summary.scalar('loss', tensor=loss)
        return loss

    def model_fn(self, features, labels, mode, params):
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        preds = self._output_network(features, params, training=training)

        # Training op
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

            predictions = {'semseg': preds}
            loss = self.loss_fn(labels, predictions, params)
            train_op = tf.contrib.training.create_train_op(loss, optimizer, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Evaluation op
        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = {'semseg': preds}
            loss = self.loss_fn(labels, predictions, params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

        # Prediction op
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'img_input': features['img_input'],
                'semseg': tf.nn.softmax(preds, axis=-1)
            }
            export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
