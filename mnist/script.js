import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

window.onload = async () => {
  const data = new MnistData();
  // 加载图片和二进制文件
  await data.load();
  // 加载验证集
  const examples = data.nextTestBatch(20);
  
  // 将数据图片放到tfvis中
  const surface = tfvis.visor().surface({ name: '输入示例' });

  // 把tensor转化为20个图片，并把他渲染到浏览器里面，用tf.slice方法
  for (let i = 0; i < 20; i += 1) {
    // tf.tidy 用来防止内存泄漏
    const imageTensor = tf.tidy(() => {
        return examples.xs
            .slice([i, 0], [1, 784])     // 切出20个图
            .reshape([28, 28, 1]);       // 改变tansor的形状，最后一个为1，表示黑白的图
    });

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px';
    // tf.browser.toPixels的参数要求：第一个参数要求维tansor（要求必须是二维或者三维的，如果是二维的那就是一个黑白的图，他的一维和二维分别代表长度和宽度；
    // 如果是三维的那就是一个彩色的rgb图，）
    // 第二个参数表示要把像素渲染到canvas上
    await tf.browser.toPixels(imageTensor, canvas);       // 转像素

    // document.body.appendChild(canvas);
    // 样式赚到tfvis看板上
    surface.drawArea.appendChild(canvas);
  }

  const model = tf.sequential();

  // 选择卷积层，因为这次图片是二维的，所以选 conv2d
  model.add(tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,                // 卷积核的大小（一般都是奇数，因为奇数有中心点，也有边缘，方便提取更丰富的特征）
      filters: 8,                   // 指需要提取的特征，有点多，不过这个可以调，先随便设8个
      strides: 1,                   // 移动步长
      activation: 'relu',           // 激活函数，如果特征小于0则删了，移除不常用的特征
      kernelInitializer: 'varianceScaling'        // 可以不设置，但是设置了可以加快他的收敛速度
  }));

  // 添加最大池化层
  model.add(tf.layers.maxPool2d({
      poolSize: [2, 2],             // 2 x 2
      strides: [2, 2]               // 移动步长
  }));

  // 进行特征组合，比如组合横和竖的特征成为一个直角，这里需要重复另一个卷积层以及池化层操作
  model.add(tf.layers.conv2d({     // 这里不用设置inputShape，因为这里他会自动算出来的
      kernelSize: 5,
      filters: 16,                 // 这里要设置更大一点，因为这里要提取更加复杂的特征，这里的特征排列组合能产生更多的特征
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
  }));

  model.add(tf.layers.maxPool2d({
      poolSize: [2, 2],
      strides: [2, 2]
  }));
  // 经过以上两轮提取差不多了，因为手写数字确实很简单，第一轮提取一个横竖，第二轮提取一个直角，接下来就要开始做分类了
  // 因为第一次是多个filter，第二次就filter更多了，他是一个二维的数据，多个filter，每个filter都提取出来一个特征图，现在输出了很多特征图，他是一个高维的数据。
  // 但是之前用过dense层我们知道，输出的是一维的数据，我们知道特征的数量当然是大于1的，那么如何将高维的数据转成一维呢，需要使用tf.layers.flatten()来做摊平
  // 这个操作其实也是把高维的filetr数据放到最后一层dense层去做分类的一种非常普遍的操作
  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({
      units: 10,                 // 因为最后输出的是0-9这10个分类
      activation: 'softmax',     // 多分类
      kernelInitializer: 'varianceScaling'
  }));

  model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: tf.train.adam(),
      metrics: ['accuracy']
  });

  // 训练集（准备时候需要把tensor放到tidy方法里面，这样子中间的tensor就会被清除掉），这样子就不会太影响性能，关于tensor操作要养成放在tidy里的习惯）
  const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(1000);        // 设置拿多少个
      console.log(d);            // 其中对象中shape字段 => shape:【1000, 784】      1000:最外层是1000个图片   784:表示里面的图片是由24 x 24 x 1组成   =>   所以他这个shape不满足我们的要求，需要reshape一下
      return [
          d.xs.reshape([1000, 28, 28, 1]),     // 把一个一维数组转成三维的，同样还是放在1000里面
          d.labels               
      ];
  });

  const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(200);
      return [
          d.xs.reshape([200, 28, 28, 1]),
          d.labels
      ];
  });

  // 调用fit方法进行训练
  await model.fit(trainXs, trainYs, {
      validationData: [testXs, testYs],
      epochs: 50,
      callbacks: tfvis.show.fitCallbacks(
          { name: '训练效果' },
          ['loss', 'val_loss', 'acc', 'val_acc'],
          { callbacks: ['onEpochEnd'] }
      )
  });

  const canvas = document.querySelector('canvas');

  canvas.addEventListener('mousemove', (e) => {
    // 左键
    if (e.buttons === 1) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgb(255,255,255)';
      ctx.fillRect(e.offsetX, e.offsetY, 25, 25);
    }
  });

  window.clear = () => {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgb(0,0,0)';      // 黑底
      ctx.fillRect(0, 0, 300, 300);      // 从（0，0）开始长和宽都为300
  };

  clear();

  window.predict = () => {
      const input = tf.tidy(() => {
          return tf.image.resizeBilinear(           
              tf.browser.fromPixels(canvas),  // 将canvas转化为tensor
              [28, 28],         // 将canvas的300 x 300转成28 x 28       
              true              // 设置边角
          ).slice([0, 0, 0], [28, 28, 1])         // 注意：这里canvas看着是黑白的，其实是彩色图片，这里需要转化为黑白图片，即将三个通道中删除两个通道，只需要一个通道，用slice来删除
          .toFloat()                              // 做归一化，先toFloat，再div
          .div(255)
          .reshape([1, 28, 28, 1]);               // 与上面训练数据保持一致
      });
      const pred = model.predict(input).argMax(1);        // 预测结果最大可能性argMax
      alert(`预测结果为 ${pred.dataSync()[0]}`);
  };
};