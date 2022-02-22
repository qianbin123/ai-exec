import * as tf from '@tensorflow/tfjs';
import { IMAGENET_CLASSES } from './imagenet_classes';
import { file2img } from './utils';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8081/mobilenet/web_model/model.json';

window.onload = async () => {
    const model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);    
    window.predict = async (file) => {
        const img = await file2img(file);
        document.body.appendChild(img);
        // tf.tidy 便于清除中间的tensor，优化webgl内存，防止内存泄漏
        const pred = tf.tidy(() => {
            // 拿到图片版tensor
            const input = tf.browser.fromPixels(img)
                .toFloat()
                .sub(255 / 2)     // 归一化
                .div(255 / 2)
                .reshape([1, 224, 224, 3]);  // 1个224x224的彩色图片
            return model.predict(input);
        });

        const index = pred.argMax(1).dataSync()[0];
        setTimeout(() => {
            alert(`预测结果：${IMAGENET_CLASSES[index]}`);
        }, 0);
    };
};