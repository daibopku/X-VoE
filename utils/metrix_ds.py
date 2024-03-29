import pandas as pd
import tensorflow as tf
import os


def count(value):
    out = tf.cast(value < 0,
                  tf.float32) + tf.cast(value == 0, tf.float32) * 0.5
    return out


def holistic(data):
    if len(data) == 6:
        loss1 = data[0]['loss_all']
        loss2 = data[1]['loss_all']
        loss3 = data[2]['loss_all']
        loss4 = data[3]['loss_all']
        loss5 = data[4]['loss_all']
        loss6 = data[5]['loss_all']
        possible = tf.concat([loss1, loss3, loss4, loss5], axis=0)
        impossible = tf.concat([loss2, loss6], axis=0)
    elif len(data) == 4:
        loss1 = data[0]['loss_all']
        loss2 = data[1]['loss_all']
        loss3 = data[2]['loss_all']
        loss4 = data[3]['loss_all']
        possible = tf.concat([loss1, loss3, loss4], axis=0)
        impossible = tf.concat([loss2], axis=0)
    else:
        return 0
    possible = tf.expand_dims(possible, axis=1)
    impossible = tf.expand_dims(impossible, axis=0)
    return tf.reduce_mean(count(possible - impossible))


def comparative(ext, sur):
    possible = ext['loss_all']
    impossible = sur['loss_all']
    return tf.reduce_mean(count(possible - impossible))


def relative(ext, sur, test, reverse=False):
    ext = ext['loss_all']
    sur = sur['loss_all']
    test = test['loss_all']
    out = test - (ext + sur) * 0.5
    if reverse:
        out = out * -1.
    return tf.reduce_mean(count(out))


def cal_holistic(data, type_name, model_name, figure_dir):
    out = []
    out.append(holistic(data).numpy())
    out = {"Type": [type_name], "Model": [model_name], "Metrix": out}
    ds = pd.DataFrame(out)
    ds['Metrix'] = ds["Metrix"] * 100.0
    if not os.path.exists(figure_dir + '/holistic'):
        os.makedirs(figure_dir + '/holistic')
    ds.to_csv(figure_dir + '/holistic/{}_{}.csv'.format(type_name, model_name))
    return ds


def cal_comparative(data, type_name, model_name, figure_dir):
    out = []
    if len(data) == 6:
        out.append(comparative(data[0], data[1]).numpy())
        out.append(comparative(data[2], data[3]).numpy())
        out.append(comparative(data[4], data[5]).numpy())
        out = {
            "Situation": ["Predictive", "Hypothetical", "Explicative"],
            "Type": [type_name, type_name, type_name],
            "Model": [model_name, model_name, model_name],
            "Metrix": out
        }
    elif len(data) == 4:
        out.append(comparative(data[0], data[1]).numpy())
        out.append(comparative(data[2], data[3]).numpy())
        out = {
            "Situation": ["Predictive", "Hypothetical"],
            "Type": [type_name, type_name],
            "Model": [model_name, model_name],
            "Metrix": out
        }
    else:
        return 0
    ds = pd.DataFrame(out)
    ds['Metrix'] = ds["Metrix"] * 100.0
    if not os.path.exists(figure_dir + '/comparative'):
        os.makedirs(figure_dir + '/comparative')
    ds.to_csv(figure_dir +
              '/comparative/{}_{}.csv'.format(type_name, model_name))
    return ds


def cal_relative(data, type_name, model_name, figure_dir):
    out = []
    if len(data) == 6:
        out.append(relative(data[0], data[1], data[2], reverse=False).numpy())
        out.append(relative(data[0], data[1], data[3], reverse=False).numpy())
        out.append(relative(data[0], data[1], data[4], reverse=False).numpy())
        out.append(relative(data[0], data[1], data[5], reverse=True).numpy())
        out = {
            "Situation": ["S2-ext1", "S2-ext2", "S3-ext", "S3-sur"],
            "Type": [type_name, type_name, type_name, type_name],
            "Model": [model_name, model_name, model_name, model_name],
            "Metrix": out
        }
    elif len(data) == 4:
        out.append(relative(data[0], data[1], data[2], reverse=False).numpy())
        out.append(relative(data[0], data[1], data[3], reverse=False).numpy())
        out = {
            "Situation": ["S2-ext1", "S2-ext2"],
            "Type": [type_name, type_name],
            "Model": [model_name, model_name],
            "Metrix": out
        }
    else:
        return 0
    ds = pd.DataFrame(out)
    ds['Metrix'] = ds["Metrix"] * 100.0
    if not os.path.exists(figure_dir + '/relative'):
        os.makedirs(figure_dir + '/relative')
    ds.to_csv(figure_dir + '/relative/{}_{}.csv'.format(type_name, model_name))
    return ds


if __name__ == '__main__':
    test = []
    for _ in range(4):
        test.append({'loss_all': tf.random.uniform([64])})
    ds = cal_holistic(test,
                      type_name="test",
                      model_name="test",
                      figure_dir="./output/")
    ds = cal_comparative(test,
                         type_name="test",
                         model_name="test",
                         figure_dir="./output/")
    ds = cal_relative(test,
                      type_name="test",
                      model_name="test",
                      figure_dir="./output/")
