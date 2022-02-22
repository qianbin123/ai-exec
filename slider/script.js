import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8081';
let transferRecognizer;
let curIndex = 0;

window.onload = async () => {
    // 识别器
    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json',
    );
    await recognizer.ensureModelLoaded();
    // 迁移学习器
    transferRecognizer = recognizer.createTransfer('轮播图');

    // 这里不需要重新采集数据，直接用二进制文件来训练数据
    const res = await fetch(MODEL_PATH + '/slider/data.bin');
    // 转成arrayBuffer数据
    const arrayBuffer = await res.arrayBuffer();
    // 将arrayBuffer加载到学习器
    transferRecognizer.loadExamples(arrayBuffer);
    console.log(transferRecognizer.countExamples());
    await transferRecognizer.train({ epochs: 30 });
    console.log('done');
};

window.toggle = async (checked) => {
    if (checked) {
        await transferRecognizer.listen(result => {
            const { scores } = result;
            const labels = transferRecognizer.wordLabels();
            const index = scores.indexOf(Math.max(...scores));
            window.play(labels[index]);
        }, {
            overlapFactor: 0,
            probabilityThreshold: 0.5
        });
    } else {
        transferRecognizer.stopListening();
    }
};

window.play = (label) => {
    const div = document.querySelector('.slider>div');
    if (label === '上一张') {
        if (curIndex === 0) { return; }
        curIndex -= 1;
    } else {
        if (curIndex === document.querySelectorAll('img').length - 1) { return; }
        curIndex += 1;
    }
    div.style.transition = "transform 1s"
    div.style.transform = `translateX(-${100 * curIndex}%)`;
};