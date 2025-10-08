const tf = require('@tensorflow/tfjs-node');

try {
    // Define a simple model.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
    model.add(tf.layers.dense({units: 1, activation: 'linear'}));
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

    const xs = tf.randomNormal([100, 10]);
    const ys = tf.randomNormal([100, 1]);

    model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, log) => {
                console.log(`Epoch ${epoch}: loss = ${log.loss}`);
            }
        }
    }).then(() => {
        const input = tf.randomNormal([1, 10]);
        const prediction = model.predict(input);
        prediction.print();
    });
} catch (error) {
    console.error('Error during model creation or training:', error);
}
