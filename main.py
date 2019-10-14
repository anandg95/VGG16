import tensorflow as tf
from model import model, loss_op, optimizer_op, accuracy_op
from create_tf_record import total_images_train

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("train_data_dir", "./data_dir/train", "train data directory")
flags.DEFINE_string("eval_data_dir", "./data_dir/eval", "eval data directory")
flags.DEFINE_string("test_data_dir", "./data_dir", "test data directory")
flags.DEFINE_integer("batch_size", 4, "batch size for training")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_integer("num_classes", 200, "Number of output classes")
flags.DEFINE_boolean("train", True, "is training?")
flags.DEFINE_integer("num_epochs", 10, "number of epochs")

tf.logging.set_verbosity(tf.logging.INFO)

def model_fn_builder():
    def model_fn(features, labels, mode, params):
        x = features["image"]
        logits = model(
            x, num_classes=FLAGS.num_classes, is_training=params["is_training"]
        )
        sparse_label = features["label"]

        if mode == tf.estimator.ModeKeys.PREDICT:
            pred_probs = pred_probs(logits)
            spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_probs)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            loss = loss_op(logits, sparse_label)
            optimizer = optimizer_op(learning_rate=params["learning_rate"])

            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step()
            )
            metrics = {
                "train_accuracy": accuracy_op(logits, sparse_label),
                "train_loss": loss,
            }

            train_logging_hook = tf.train.LoggingTensorHook(metrics, every_n_iter=10)
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                # eval_metric_ops=metrics,
                training_hooks=[train_logging_hook],
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            loss = loss_op(logits, sparse_label)
            metrics = {"eval_accuracy": accuracy_op(logits, sparse_label), "eval_loss": loss}
            eval_logging_hook = tf.train.LoggingTensorHook(metrics, every_n_iter=10)
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                # eval_metric_ops=metrics,
                evaluation_hooks=[eval_logging_hook],
            )

        return spec

    return model_fn


def input_fn_builder(
    input_file, batch_size=FLAGS.batch_size, repeat=True, is_training=True
):
    name_to_features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def input_fn():
        def decode_record(record):
            ex = tf.io.parse_single_example(record, name_to_features)
            image = tf.image.decode_jpeg(ex["image"], channels=3)
            image = tf.image.resize(image, [224, 224])
            image /= 255.0
            # label = tf.one_hot(ex["label"], depth=FLAGS.num_classes)
            label = ex["label"]
            return dict(image=image, label=label)

        d = tf.data.TFRecordDataset(input_file)
        if repeat:
            d = d.repeat()
        if is_training:
            d = d.shuffle(buffer_size=1000)
        d = d.map(decode_record)
        d = d.batch(batch_size, drop_remainder=False)
        d = d.prefetch(1)
        return d

    return input_fn


def run_model():
    params = {"learning_rate": FLAGS.lr, "is_training": True}
    if not FLAGS.train:
        params.update(is_training=False)

    model = tf.estimator.Estimator(
        model_fn=model_fn_builder(), params=params, model_dir="./model_files/"
    )

    if FLAGS.train:
        # train
        total_steps = int(total_images_train / FLAGS.batch_size * FLAGS.num_epochs)
        input_fn_train = input_fn_builder(
            input_file=f"{FLAGS.train_data_dir}/train/data.tfrecord"
        )
        input_fn_eval = input_fn_builder(
            input_file=f"{FLAGS.train_data_dir}/eval/data.tfrecord"
        )
        train_spec = tf.estimator.TrainSpec(
            input_fn=input_fn_train, max_steps=total_steps
        )
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_eval, steps=100)
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        # test/predict
        pass


if __name__ == "__main__":
    run_model()
