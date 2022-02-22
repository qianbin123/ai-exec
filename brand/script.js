import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getInputs } from './data';
import { img2x, file2img } from './utils';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8081/mobilenet/web_model/model.json';
const NUM_CLASSES = 3;
const BRAND_CLASSES = ['android', 'apple', 'windows'];

window.onload = async () => {
    const { inputs, labels } = await getInputs();
    const surface = tfvis.visor().surface({ name: '输入示例', styles: { height: 250 } });
    inputs.forEach(img => {
        surface.drawArea.appendChild(img);
    });
    // 加载模型
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    // 查看模型概况
    mobilenet.summary();

    // 查看模型概况则选择开始截断的层，这次选conv_pw_13_relu
    const layer = mobilenet.getLayer('conv_pw_13_relu');

    const truncatedMobilenet = tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output     // 从截断的这层作为输出
    });
    console.log(layer);  // 输出一个数组 [null, 7, 7 , 256]  第一个元素，代表它的输入或者输出的个数，null表示个数是不定的

    // --- 开始简单的双层神经网络 ---
    const model = tf.sequential();
    // 之前我们学过卷积神经网络，卷积神经网络为了把高维的卷积提取到的特征做一个分类，通常都需要flatten一下
    model.add(tf.layers.flatten({
        inputShape: layer.outputShape.slice(1)     // 输出层的输出形状，切出[7, 7, 256]
    }));
    // 后面这两层都是dense层
    model.add(tf.layers.dense({
        units: 10,
        activation: 'relu'      // 设置relu为了有非线性变化，加别的激活函数试一下也行
    }));
    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'  // 为了能做多分类
    }));
    // 损失函数 categoricalCrossentropy交叉熵，优化器 adam
    model.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.adam() });

    const { xs, ys } = tf.tidy(() => {
        // tf.concat用于合并tensor
        const xs = tf.concat(inputs.map(imgEl => truncatedMobilenet.predict(img2x(imgEl))));
        const ys = tf.tensor(labels);
        return { xs, ys };
    });

    await model.fit(xs, ys, {
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    window.predict = async (file) => {
        const img = await file2img(file);
        document.body.appendChild(img);
        const pred = tf.tidy(() => {
            const x = img2x(img);
            // 截断模型预测
            const input = truncatedMobilenet.predict(x);
            // 普通神经网络模型预测
            return model.predict(input);
        });

        const index = pred.argMax(1).dataSync()[0];
        setTimeout(() => {
            alert(`预测结果：${BRAND_CLASSES[index]}`);
        }, 0);
    };

    window.download = async () => {
        // downloads协议表示下载成文件；也可以改成其他协议，保存成其他形式，具体看官网
        await model.save('downloads://model');
    };
};