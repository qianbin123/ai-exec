export function file2img(file) {
  return new Promise(resolve => {
      const reader = new FileReader();
      // 开始读这个文件
      reader.readAsDataURL(file);
      // 加载完成后触发 onload 
      reader.onload = (e) => {
          const img = document.createElement('img');
          img.src = e.target.result;
          // 这个宽高必须设置成224的，因为它接受到的图片也是224的 
          img.width = 224;
          img.height = 224;
          img.onload = () => resolve(img);
      };
  });
}