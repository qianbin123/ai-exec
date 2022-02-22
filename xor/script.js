import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

window.onload = async () => {
    const data = getData(400);

    tfvis.render.scatterplot(
        { name: 'XOR 训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );

    // 选择神经网络模型
    const model = tf.sequential();

    // 为模型添加隐藏层
    model.add(tf.layers.dense({
      units: 4,
      inputShape: [2], //长度为2的一维数组，因为数据特征为x,y的坐标
      activation: 'relu'   // 只需要在意能带来非线性的变化就行，不用台在意哪个激活函数
    }))
    // 为模型添加输出层
    model.add(tf.layers.dense({
      units: 1,        // 设置神经元个数为1，因为我们只需要最后输出一个概率就行
      // inputShape: [2],        // 只要设置第一层的inputShape就行，后面的层是会自动计算所需的inputShape要多少
      activation: 'sigmoid'   // 因为它要输出一个0-1之间的概率，所以只能选择用sigmoid
    }))

    // 定义模型的损失函数和优化器
    // adam 优化器优点：可以自动调节学习率，这次初始化设置为0.1
    model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1)})

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    // 可视化训练过程
    await model.fit(inputs, labels, {
      batchSize: 40,
      epochs: 20,
      callbacks: tfvis.show.fitCallbacks(
        { name: '训练过程' },
        ['loss']                  // 度量单位只看损失
      )
    })

    window.predict = (form) => {
      // 乘1，为了转为数字
      const pred = model.predict(tf.tensor([[form.x.value*1, form.y.value*1]]));
      alert(`预测结果：${pred.dataSync()[0]}`);
    }
};