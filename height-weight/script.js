import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';

window.onload = async () => {
  const heights = [150, 160, 170];
  const weights = [40, 50, 60];

  tfvis.render.scatterplot(
    { name: '线性回归训练集' },
    { values: heights.map((x, i) => ({x, y: weights[i]})) },
    {
      xAxisDomain: [140, 180],
      yAxisDomain: [30, 70]
    }
  );
  
  // 归一化后的数据放入inputs，分析一下如何把 [150, 160, 170] 归一化，即压缩到0到1之间？
  /**
   * 150为0，170为1
   * 方式：可以先将数组里每个数“减150”，然后每个数字再“除20”，20表示170-150=20这个间距
   * 解决：
   *  第一个（150-150）/ 20 = 0
   *  第二个（160-150）/ 20 = 0.5
   *  第三个（170-150）/ 20 = 1
   */
  // API：sub表示减法，div表示除法
  // tf.tensor(heights).sub(20) 表示对heights中每个数，先减去150，再除20
  const inputs = tf.tensor(heights).sub(150).div(20);
  // 对体重也是如此
  const labels = tf.tensor(weights).sub(40).div(20);



  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1)       // sgd内参数为学习率，学习率最不宜过高，容易过头，当然也不能太低，这样子得出结果的速度慢了
  })

  // fit 方法是一个异步的，返回Promise，
  // 第一个参数inoputs就是输入的值，第二个参数label就是正确的值
  await model.fit(inputs, labels, {
    batchSize: 3,                                 // 指小批量随机梯度中，每个小梯度设置成多大，即每批样本数据有多大，默认值是32
    epochs: 200,                                  // 迭代整个训练次数
    callbacks: tfvis.show.fitCallbacks(           // 利用tfvis来可视化整个训练过程，具体参数看接口文档
      { name: '训练过程' },
      ['loss']                                    // 想看哪个度量单位
    )
  });

  // 准备预测180的人对应体重，因为训练时候输出数据做了归一化，所以这里也要做归一化
  const output = model.predict(tf.tensor([180])).sub(150).div(20);
  // 如果想把tensor变为普通数据，则用dataSync方法
  alert(`如果 x 为 180，那么预测 y 为 ${output.mul(20).add(40).dataSync()[0]}kg`);
}