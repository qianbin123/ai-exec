const IMAGE_SIZE = 224;

const loadImg = (src) => {
    return new Promise(resolve => {
        const img = new Image();
        // 由于前端端口是1234，图片链接的端口是8081端口，会产生跨域，如果要让tensorflow能识别，需要加crossOrigin
        img.crossOrigin = "anonymous";
        img.src = src;
        img.width = IMAGE_SIZE;
        img.height = IMAGE_SIZE;
        img.onload = () => resolve(img);
    });
};
export const getInputs = async () => {
    const loadImgs = [];
    const labels = [];
    for (let i = 0; i < 30; i += 1) {
        ['android', 'apple', 'windows'].forEach(label => {
            const src = `http://127.0.0.1:8081/brand/train/${label}-${i}.jpg`;
            const img = loadImg(src);
            loadImgs.push(img);
            labels.push([
                label === 'android' ? 1 : 0,
                label === 'apple' ? 1 : 0,
                label === 'windows' ? 1 : 0, 
            ]);
        });
    }
    const inputs = await Promise.all(loadImgs);
    return {
        inputs,
        labels,
    };
}