import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8081/speech';

window.onload = async () => {
    // 建立识别器
    const recognizer = speechCommands.create(
        'BROWSER_FFT',              // FFT 表示傅立叶变换的意思，这里选择浏览器自带的傅立叶变换
        null,
        MODEL_PATH + '/model.json',
        MODEL_PATH + '/metadata.json'
    );

    // 确保模型加载完毕
    await recognizer.ensureModelLoaded();

    // 查看模型能识别哪些单词
    const labels = recognizer.wordLabels().slice(2);

    const resultEl = document.querySelector('#result');
    resultEl.innerHTML = labels.map(l => `
        <div>${l}</div>
    `).join('');
 
    // 监听麦克风输入
    recognizer.listen(result => {
        const { scores } = result;
        // 拿取最大值
        const maxValue = Math.max(...scores);
        const index = scores.indexOf(maxValue) - 2;
        resultEl.innerHTML = labels.map((l, i) => `
        <div style="background: ${i === index && 'green'}">${l}</div>
        `).join('');
    }, {
        overlapFactor: 0.3,                    // 识别频率，决定识别次数
        probabilityThreshold: 0.9              // 可能性预值，表示预测这个数据需要达到90%以上的相似度，它才会执行这个回调函数
    });
};