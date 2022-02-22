import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8081';

let transferRecognizer;       // 迁移学习的识别器

window.onload = async () => {
    // 创建识别器
    const recognizer = speechCommands.create(
        'BROWSER_FFT',      // 使用浏览器的傅立叶方法
        null,               // 词汇
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json'           // 语言信息链接
    );

    await recognizer.ensureModelLoaded();              // 确保识别器加载完成

    transferRecognizer = recognizer.createTransfer('轮播图');
};

// 涉及到I/O操作一般都是异步的
window.collect = async (btn) => {
    btn.disabled = true;
    const label = btn.innerText;

    // 要区分“有用信息（上一张和下一张）和无用信息（噪音）”，噪音是不能传进去的，否则会认为背景噪音这四个字也是一个label
    await transferRecognizer.collectExample(
        label === '背景噪音' ? '_background_noise_' : label
    );

    btn.disabled = false;
    // 通过countExamples可以看见当前的收集情况
    document.querySelector('#count').innerHTML = JSON.stringify(transferRecognizer.countExamples(), null, 2);
};

window.train = async () => {
    // 直接使用迁移学习器的train方法，和fit方法很像
    await transferRecognizer.train({
        epochs: 30,
        callback: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });
};

window.toggle = async (checked) => {
    if (checked) {
        // 如果checkbox是打开状况下，这里开始使用迁移学习器的listen方法进行接听，这个 的作用和之前用的recognizer.listen是一样的
        await transferRecognizer.listen(result => {
            const { scores } = result;                                      // scores里有每个语音的得分情况
            const labels = transferRecognizer.wordLabels();
            const index = scores.indexOf(Math.max(...scores));
            console.log(labels[index]);
        }, {
            overlapFactor: 0,               // 表示别设置的很频繁
            probabilityThreshold: 0.75
        });
    } else {
        // 如果checkbox是关闭状况下，则停止监听
        transferRecognizer.stopListening();
    }
};

window.save = () => {
    // 将采集好的数据序列化成二进制数据，即arrayBuffer格式
    const arrayBuffer = transferRecognizer.serializeExamples();

    // blob为下载做准备
    const blob = new Blob([arrayBuffer]);
    // 一般下载都是用a标签
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    // 定义下载名称，用bin后缀，主要为了体现这个是二进制文件
    link.download = 'data.bin';
    link.click();
};


























// import * as tfvis from '@tensorflow/tfjs-vis';


// window.onload = async () => {
//     const recognizer = speechCommands.create(
//         'BROWSER_FFT',
//         null,
//         MODEL_PATH + '/speech/model.json',
//         MODEL_PATH + '/speech/metadata.json'
//     );
//     await recognizer.ensureModelLoaded();
//     transferRecognizer = recognizer.createTransfer('轮播图');
// };




