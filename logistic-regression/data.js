export function getData(numSamples){
  let point = [];

  // 生成正态分布
  function genGauss(cx, cy, label){
    for(let i = 0; i < numSamples / 2; i++){
      // 已cx为中心生成正态分布点
      let x = normalRandom(cx);
      let y = normalRandom(cy);
      point.push({x, y, label});
    }
  }

  genGauss(2, 2, 1);
  genGauss(-2, -2, 0);
  return point;
}

// 用来生成正太分布的随机数
function normalRandom(mean = 0, variance = 1){
  let v1, v2, s;

  do{
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  }while(s > 1);

  let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}