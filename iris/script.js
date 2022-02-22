import * as tf from '@tensorflow/tfjs';
import * as tfvis from "@tensorflow/tfjs-vis";
// getIrisData 用于获取训练集和验证集
import { getIrisData, IRIS_CLASSES } from './data';

window.onload = async () => {
  // getIrisData的参数为小于1的一个比例作为验证集（注意，验证集和训练集可以用同一批数据），比如：0.15 表示有 15% 的数据作为验证集
  const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

  const model = tf.sequential();

  model.add(tf.layers.dense({
    units: 10,        // 凭直觉写一个，后期再慢慢调整
    inputShape: [xTrain.shape[1]],  // 可以直接写[4] 也可以 [xTrain.shape]
    activation: 'sigmoid'
  }));

  model.add(tf.layers.dense({
    units: 3,        // 凭直觉写一个，后期再慢慢调整
    activation: 'softmax'
  }));

  // 设置损失函数和优化器
  model.compile({
    loss: 'categoricalCrossentropy',      // 这次是字符串添加损失函数方式，不同于之前从tf从取
    optimizer: tf.train.adam(0.1),
    metrics: ['accuracy']
  })

  // 开始训练
  await model.fit(xTrain, yTrain, {
    epochs: 100,
    validationData: [xTest, yTest],
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练效果' },
      ['loss', 'val_loss', 'acc', 'val_acc'],      // val_loss🈯️验证集的损失       val_acc指验证集的准确度
      { callbacks: ['onEpochEnd'] }
    )
  })

  window.predict = (form) => {
    const input = tf.tensor([[
        form.a.value * 1,
        form.b.value * 1,
        form.c.value * 1,
        form.d.value * 1,
    ]]);
    const pred = model.predict(input);
    // tf提供了一个方法，可以输出某个维度的最大值
    alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`)
  };
}