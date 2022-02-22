import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';

window.onload = async () => {
  const xs = [1,2,3,4];
  const ys = [1,3,5,7];

  tfvis.render.scatterplot(
    { name: '线性回归训练集' },
    { values: xs.map((x, i) => ({x, y: ys[i]})) },
    { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }
  );

  // sequential表示连续的，即这一层的输入一定是上一层的输出
  const model = tf.sequential();
  // 添加层，tf.layers.dense为“全连接层”，至于为什么要用到这个，可以联想到，它可以解决关于 “输入 * 权重 + 偏置” 的相关问题
  // unit 指神经元个数, inputShape中必须至少为一维
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1)       // sgd内参数为学习率，学习率最不宜过高，容易过头，当然也不能太低，这样子得出结果的速度慢了
  })

  const inputs = tf.tensor(xs);
  const labels = tf.tensor(ys);     // 正确的标签

  // fit 方法是一个异步的，返回Promise，
  // 第一个参数inoputs就是输入的值，第二个参数label就是正确的值
  await model.fit(inputs, labels, {
    batchSize: 4,                                 // 指小批量随机梯度中，每个小梯度设置成多大,即每批样本数据有多大
    epochs: 200,                                  // 迭代整个训练次数
    callbacks: tfvis.show.fitCallbacks(           // 利用tfvis来可视化整个训练过程，具体参数看接口文档
      { name: '训练过程' },
      ['loss']                                    // 想看哪个度量单位
    )
  });

  // 再训练好模型之后，根据数据的x=5（需要转tensor结构），得到返回的y（y的结构根据训练时y的结构保持一致，比如此次，之前训练时y是数组，所以这次也为数组）
  const output = model.predict(tf.tensor([5]));
  // 如果想把tensor变为普通数据，则用dataSync方法
  alert(`如果 x 为 5，那么预测 y 为 ${output.dataSync()[0]}`);
};