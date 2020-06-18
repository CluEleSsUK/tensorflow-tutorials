class SquaredSumLayer extends tf.layers.Layer {
  constructor() {
    super({});
  }

  computeOutputShape() {
    return []
  }

  call(input) {
    return input.square().sum()
  }

  getClassName() {
    return "SquaredSum"
  }
}

const t = tf.tensor([-2, 1, 0, 5])
const o = new SquaredSumLayer().apply(t)
o.print()