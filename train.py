from absl import app, flags, logging
from absl.flags import FLAGS
import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.metrics import (
    Precision,
    Recall
)
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset
from yolov3_tf2.models import post_process_block

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_string('checkpoint_dir', './checkpoints/training', 'path to save checkpoints')

def log_batch(logging, epoch, batch, total_loss, pred_loss):
    # TODO: calculate precision, recall, and mAP
    loss_total = "ep:{}, batch:{}, loss_total:{:.4f}".format(
        epoch, batch, total_loss.numpy())
    loss_by_output = ''.join([', out_{}:{:.4f}'.format(
        i, np.sum(tf.reduce_sum(x).numpy())) 
        for i, x in enumerate(pred_loss)])
    logging.info(loss_total + loss_by_output)

def true_false_positives(boxes, scores, y_true, iou_thres=0.5, score_thres=0.5):
    """
    Compute number of true and false positives with same class.

    Inputs:
        boxes - Predicted boxes in (xmin, ymin, xmax, ymax) format.
        scores - Corresponding scores for each predicted box.
        y_true - Label boxes in (xmin, ymin, xmax, ymax) format.
        iou_thres - Minimum IOU for positive detection.
        score_thres - Minimum prediction score for positive detection.
    Returns:
        true_pos - Number of correct positive predictions.
        false_pos - Number of incorrect positive predictions.
    """

    n_pos = y_true.shape[0]
    # no true positives
    if n_pos == 0:
        return 0., float(boxes.shape[0])

    sorted_ind = np.argsort(-scores)
    try:
        boxes = boxes[sorted_ind, :]
    except:
        return 0., 0.
    
    true_found_yet = np.zeros(y_true.shape[0])
    true_pos = 0.
    false_pos = 0.
    for box, score in zip(boxes, scores):
        if score < score_thres:
            continue
        ovmax = -np.Inf
        ixmin = np.maximum(y_true[:, 0], box[0])
        iymin = np.maximum(y_true[:, 1], box[1])
        ixmax = np.minimum(y_true[:, 2], box[2])
        iymax = np.minimum(y_true[:, 3], box[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) + (y_true[:, 2] - y_true[:, 0] + 1.) * (
                    y_true[:, 3] - y_true[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        max_gt_idx = np.argmax(overlaps)

        if ovmax > iou_thres:
            # gt not matched yet
            if not true_found_yet[max_gt_idx]:
                true_pos += 1.
                true_found_yet[max_gt_idx] = 1
            else:
                false_pos += 1.
        else:
            false_pos += 1.
    return true_pos, false_pos

def calc_precision(true_pos, false_pos):
    return true_pos / np.maximum(true_pos + false_pos,
        np.finfo(np.float64).eps)

def calc_recall(true_pos, n_pos):
    return true_pos / float(n_pos)

def batch_true_false_positives(preds, labels, num_classes):
    """
    Compute true positives, false positives, and number of positives across a
        batch of predictions.

    Inputs:
        preds - Tensor containing predictions in format: 
            (boxes, scores, classes, n_valid_detections).
        labels - Tensor of labels in format: (xmin, ymin, xmax, ymax, class).
        num_classes - The total number of classes in dataset.
    Returns:
        true_pos - List with number of correct positive predictions for each observation.
        false_pos - List with number of incorrect positive predictions for each observation.
        n_pos - List with number of ground truth positives for each observation.
    """

    true_pos = np.zeros(num_classes)
    false_pos = np.zeros(num_classes)
    n_pos = np.zeros(num_classes)
    for boxes, scores, classes, valid_dets, img_labels in zip(*preds, labels):
        boxes = boxes[:valid_dets]
        scores = scores[:valid_dets]
        classes = classes[:valid_dets]
        img_labels = img_labels
        
        # remove zero padding
        img_labels_size = np.multiply.reduce(
            img_labels[..., 2:4] - img_labels[..., 0:2], axis=-1)
        img_labels = img_labels[img_labels_size > 1e-8]
        for c in range(num_classes):
            c_img_labels = img_labels[img_labels[..., -1] == c]
            c_idxs = np.nonzero(classes == c)
            n_pos[c] += len(c_idxs)
            tp, fp = true_false_positives(boxes[c_idxs], scores[c_idxs], c_img_labels)
            true_pos[c] += tp
            false_pos[c] += fp

    return true_pos, false_pos, n_pos

def batch_precision_recall(true_pos, false_pos, n_pos):
    """
    Compute precision and recall across a batch of observations.

    Inputs:
        true_pos - List with number of correct positive predictions for each observation.
        false_pos - List with number of incorrect positive predictions for each observation.
        n_pos - List with number of ground truth positives for each observation.
    Returns:
        precision - The precision for each observation.
        recall - The recall for each observation.
    """

    precision = np.zeros(true_pos.shape)
    recall = np.zeros(true_pos.shape)
    for c in range(true_pos.shape[0]):
        precision[c] = calc_precision(true_pos[c], false_pos[c])
        recall[c] = calc_recall(true_pos[c], n_pos[c])
    return precision, recall
    

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    post_process_outputs = post_process_block(model.outputs, 
        classes=FLAGS.num_classes)
    post_process_model = Model(model.inputs, post_process_outputs)

    train_dataset = dataset.load_fake_dataset()
    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        y))
        # dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        y))
        # dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)
    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    # (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
    # model.outputs shape: [[N, 13, 13, 3, 85], [N, 26, 26, 3, 85], [N, 52, 52, 3, 85]]
    # labels shape: ([N, 13, 13, 3, 6], [N, 26, 26, 3, 6], [N, 52, 52, 3, 6])
    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    transf_labels = dataset.transform_targets(labels, anchors, anchor_masks, FLAGS.size)
                    for output, label, loss_fn in zip(outputs, transf_labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss, axis=None) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                log_batch(logging, epoch, batch, total_loss, pred_loss)
                avg_loss.update_state(total_loss)

                if batch >= 100:
                    break

            true_pos_total = np.zeros(FLAGS.num_classes)
            false_pos_total = np.zeros(FLAGS.num_classes)
            n_pos_total = np.zeros(FLAGS.num_classes)
            for batch, (images, labels) in enumerate(val_dataset):
                # get losses
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                transf_labels = dataset.transform_targets(labels, anchors, anchor_masks, FLAGS.size)
                for output, label, loss_fn in zip(outputs, transf_labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss
                log_batch(logging, epoch, batch, total_loss, pred_loss)
                avg_val_loss.update_state(total_loss)

                # get true positives, false positives, and positive labels
                preds = post_process_model(images)
                true_pos, false_pos, n_pos = batch_true_false_positives(preds.numpy(), 
                    labels.numpy(), FLAGS.num_classes)
                true_pos_total += true_pos
                false_pos_total += false_pos
                n_pos_total += n_pos

                if batch >= 20:
                    break

            # precision-recall by class
            precision, recall = batch_precision_recall(true_pos_total, 
                false_pos_total, n_pos_total)
            for c in range(FLAGS.num_classes):
                print('Class {} - Prec: {}, Rec: {}'.format(c, 
                    precision[c], recall[c]))
            # total precision-recall
            print('Total - Prec: {}, Rec: {}'.format(
                calc_precision(np.sum(true_pos_total), np.sum(false_pos_total)), 
                calc_recall(np.sum(true_pos_total), np.sum(n_pos_total))))
            import pdb; pdb.set_trace()

            # log losses
            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            # reset loop and save weights
            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                os.path.join(FLAGS.checkpoint_dir, 'yolov3_train_{}.tf'\
                    .format(epoch)))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint(
                os.path.join(FLAGS.checkpoint_dir, 'yolov3_train_{epoch}.tf'),
                verbose=1, save_weights_only=True),
            TensorBoard(log_dir=FLAGS.log_dir)
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
