import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data.js";
import * as tf from '@tensorflow/tfjs';

window.onload = async () => {
  const data = getData(400);

  // 散点展示400个点的布局
  tfvis.render.scatterplot(
    { name: '逻辑回归训练数据' },
    {
      values: [
        data.filter( p => p.label === 1),
        data.filter( p => p.label === 0),
      ]
    }
  );


  const model = tf.sequential();


  model.add(tf.layers.dense({
    units: 1,
    inputShape: [2],
    activation: 'sigmoid'
  }));


  // 设置损失函数和优化器
  // adam 优化器优点：可以自动调节学习率
  model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1)})

  const inputs = tf.tensor(data.map(p => [p.x, p.y]));
  const labels = tf.tensor(data.map(p => p.label));

  await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 20,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练过程' },
      ['loss']
    )
  })

  window.predict = (form) => {
    // 乘1，为了转为数字
    const pred = model.predict(tf.tensor([[form.x.value*1, form.y.value*1]]));
    alert(`预测结果：${pred.dataSync()[0]}`);
  }
}