let model = null;
let isTraining = false;
let irisDataset = null;
let dataStats = null;

// Function to update status display
function updateStatus(message) {
    document.getElementById('status').textContent = message;
}

// Function to log training progress
function logTraining(message) {
    const logContent = document.getElementById('logContent');
    logContent.innerHTML += message + '<br>';
    logContent.scrollTop = logContent.scrollHeight;
}

// Function to load and parse CSV data
async function loadCSVData() {
    try {
        updateStatus('Loading Iris.csv...');
        
        const response = await fetch('Iris.csv');
        if (!response.ok) {
            throw new Error(`Failed to load Iris.csv: ${response.status} ${response.statusText}`);
        }
        
        const csvText = await response.text();
        return parseCSV(csvText);
        
    } catch (error) {
        console.error('Error loading CSV:', error);
        throw new Error('Could not load Iris.csv. Make sure the file exists in the same directory as your HTML file.');
    }
}

// Function to parse CSV text into array
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const data = [];
    let headers = null;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        const values = line.split(',').map(val => val.trim().replace(/"/g, ''));
        
        if (i === 0) {
            // First line should be headers in your CSV
            headers = values;
            console.log('CSV Headers:', headers);
            continue;
        }
        
        // Parse the data row - your CSV format: Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
        if (values.length >= 6) {
            // Extract species name (remove "Iris-" prefix and convert to lowercase)
            let speciesName = values[5].toLowerCase();
            if (speciesName.startsWith('iris-')) {
                speciesName = speciesName.substring(5); // Remove "iris-" prefix
            }
            
            const row = [
                parseFloat(values[1]), // SepalLengthCm
                parseFloat(values[2]), // SepalWidthCm  
                parseFloat(values[3]), // PetalLengthCm
                parseFloat(values[4]), // PetalWidthCm
                speciesName            // Species (cleaned)
            ];
            
            // Validate the row
            if (!isNaN(row[0]) && !isNaN(row[1]) && !isNaN(row[2]) && !isNaN(row[3]) && row[4]) {
                data.push(row);
            }
        }
    }
    
    if (data.length === 0) {
        throw new Error('No valid data found in CSV file');
    }
    
    console.log(`Parsed ${data.length} samples from CSV`);
    console.log('Sample data:', data.slice(0, 3));
    
    return { data, headers };
}

// Function to calculate dataset statistics
function calculateDataStats(data) {
    const species = {};
    const features = {
        sepalLength: [],
        sepalWidth: [],
        petalLength: [],
        petalWidth: []
    };
    
    data.forEach(row => {
        const speciesName = row[4];
        species[speciesName] = (species[speciesName] || 0) + 1;
        
        features.sepalLength.push(row[0]);
        features.sepalWidth.push(row[1]);
        features.petalLength.push(row[2]);
        features.petalWidth.push(row[3]);
    });
    
    const getStats = (arr) => ({
        min: Math.min(...arr).toFixed(2),
        max: Math.max(...arr).toFixed(2),
        mean: (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2)
    });
    
    return {
        totalSamples: data.length,
        species: species,
        features: {
            sepalLength: getStats(features.sepalLength),
            sepalWidth: getStats(features.sepalWidth),
            petalLength: getStats(features.petalLength),
            petalWidth: getStats(features.petalWidth)
        }
    };
}

// Function to display dataset information
function displayDataInfo(stats) {
    const dataStatsDiv = document.getElementById('dataStats');
    const speciesInfo = Object.entries(stats.species)
        .map(([species, count]) => `${species}: ${count}`)
        .join(', ');
    
    dataStatsDiv.innerHTML = `
        <p><strong>Total Samples:</strong> ${stats.totalSamples}</p>
        <p><strong>Species Distribution:</strong> ${speciesInfo}</p>
        <p><strong>Feature Ranges:</strong></p>
        <ul>
            <li>Sepal Length: ${stats.features.sepalLength.min}-${stats.features.sepalLength.max} cm (avg: ${stats.features.sepalLength.mean})</li>
            <li>Sepal Width: ${stats.features.sepalWidth.min}-${stats.features.sepalWidth.max} cm (avg: ${stats.features.sepalWidth.mean})</li>
            <li>Petal Length: ${stats.features.petalLength.min}-${stats.features.petalLength.max} cm (avg: ${stats.features.petalLength.mean})</li>
            <li>Petal Width: ${stats.features.petalWidth.min}-${stats.features.petalWidth.max} cm (avg: ${stats.features.petalWidth.mean})</li>
        </ul>
    `;
    
    document.getElementById('dataInfo').style.display = 'block';
}

// Function to shuffle array
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// Function to preprocess data for training
function preprocessData(data) {
    // Shuffle the data
    const shuffledData = shuffleArray(data);
    
    // Create species mapping
    const uniqueSpecies = [...new Set(data.map(row => row[4]))].sort();
    const speciesMap = {};
    uniqueSpecies.forEach((species, index) => {
        speciesMap[species] = index;
    });
    
    console.log('Species mapping:', speciesMap);
    console.log('Unique species:', uniqueSpecies);
    
    const features = [];
    const labels = [];
    
    shuffledData.forEach(row => {
        // Extract features (first 4 columns)
        features.push([row[0], row[1], row[2], row[3]]);
        
        // Create one-hot encoding for species
        const speciesIndex = speciesMap[row[4]];
        const oneHot = new Array(uniqueSpecies.length).fill(0);
        oneHot[speciesIndex] = 1;
        labels.push(oneHot);
    });
    
    console.log(`Preprocessed ${features.length} samples`);
    console.log('Sample features:', features.slice(0, 3));
    console.log('Sample labels:', labels.slice(0, 3));
    
    return {
        xs: tf.tensor2d(features),
        ys: tf.tensor2d(labels),
        speciesMap: speciesMap,
        uniqueSpecies: uniqueSpecies
    };
}

// Function to create and train the model
async function createAndTrainModel(xs, ys) {
    // Create the model
    model = tf.sequential({
        layers: [
            tf.layers.dense({
                inputShape: [4],
                units: 16,
                activation: 'relu'
            }),
            tf.layers.dropout({ rate: 0.2 }),
            tf.layers.dense({
                units: 12,
                activation: 'relu'
            }),
            tf.layers.dropout({ rate: 0.2 }),
            tf.layers.dense({
                units: 3,
                activation: 'softmax'
            })
        ]
    });
    
    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    updateStatus('Training model...');
    document.getElementById('trainingLog').style.display = 'block';
    logTraining('Starting training...');
    logTraining(`Training samples: ${xs.shape[0]}`);
    logTraining('Model architecture: 4 -> 16 -> 12 -> 3');
    
    // Train the model
    const history = await model.fit(xs, ys, {
        epochs: 150,
        batchSize: 16,
        validationSplit: 0.2,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (epoch % 20 === 0 || epoch === 149) {
                    const accuracy = (logs.acc * 100).toFixed(1);
                    const valAccuracy = logs.val_acc ? (logs.val_acc * 100).toFixed(1) : 'N/A';
                    logTraining(`Epoch ${epoch + 1}: accuracy = ${accuracy}%, val_accuracy = ${valAccuracy}%, loss = ${logs.loss.toFixed(4)}`);
                }
            }
        }
    });
    
    const finalAccuracy = (history.history.acc[history.history.acc.length - 1] * 100).toFixed(1);
    const finalValAccuracy = history.history.val_acc ? 
        (history.history.val_acc[history.history.val_acc.length - 1] * 100).toFixed(1) : 'N/A';
    
    logTraining(`Training completed! Final accuracy: ${finalAccuracy}%`);
    logTraining(`Final validation accuracy: ${finalValAccuracy}%`);
    
    return history;
}

// Main function to load data and train model
async function loadDataAndTrain() {
    if (isTraining) return;
    
    isTraining = true;
    document.getElementById('trainBtn').disabled = true;
    
    try {
        // Load CSV data
        const csvResult = await loadCSVData();
        irisDataset = csvResult.data;
        
        updateStatus('Data loaded successfully! Analyzing...');
        
        // Calculate and display statistics
        dataStats = calculateDataStats(irisDataset);
        displayDataInfo(dataStats);
        
        // Preprocess data
        updateStatus('Preprocessing data...');
        const preprocessed = preprocessData(irisDataset);
        
        updateStatus('Building neural network...');
        
        // Train model
        await createAndTrainModel(preprocessed.xs, preprocessed.ys);
        
        // Store species information for predictions
        model.speciesNames = preprocessed.uniqueSpecies;
        
        updateStatus('Model trained successfully! 🎉 Ready for predictions.');
        
        // Show prediction section
        document.getElementById('predictionSection').style.display = 'block';
        
        // Clean up tensors
        preprocessed.xs.dispose();
        preprocessed.ys.dispose();
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Error: ' + error.message);
        logTraining('Error: ' + error.message);
    } finally {
        isTraining = false;
        document.getElementById('trainBtn').disabled = false;
    }
}

// Function to make predictions
async function makePrediction() {
    if (!model) {
        alert('Please train the model first!');
        return;
    }
    
    const sepalLength = parseFloat(document.getElementById('sepalLength').value);
    const sepalWidth = parseFloat(document.getElementById('sepalWidth').value);
    const petalLength = parseFloat(document.getElementById('petalLength').value);
    const petalWidth = parseFloat(document.getElementById('petalWidth').value);
    
    if (isNaN(sepalLength) || isNaN(sepalWidth) || isNaN(petalLength) || isNaN(petalWidth)) {
        alert('Please enter valid numbers for all measurements.');
        return;
    }
    
    // Make prediction
    const inputData = tf.tensor2d([[sepalLength, sepalWidth, petalLength, petalWidth]]);
    const prediction = model.predict(inputData);
    const probabilities = await prediction.data();
    const predictedIndex = prediction.argMax(1).dataSync()[0];
    
    const predictedSpecies = model.speciesNames[predictedIndex];
    const confidence = (probabilities[predictedIndex] * 100).toFixed(1);
    
    // Display result
    const resultDiv = document.getElementById('predictionResult');
    
    const probabilityDetails = model.speciesNames.map((species, index) => 
        `${species.charAt(0).toUpperCase() + species.slice(1)}: ${(probabilities[index] * 100).toFixed(1)}%`
    ).join('<br>');
    
    resultDiv.innerHTML = `
        <div class="prediction-result">
            <h3>🌺 Prediction Result</h3>
            <p><strong>Predicted Species:</strong> ${predictedSpecies.charAt(0).toUpperCase() + predictedSpecies.slice(1)}</p>
            <p><strong>Confidence:</strong> ${confidence}%</p>
            <hr style="margin: 15px 0; border: 1px solid rgba(255,255,255,0.3);">
            <p><strong>All Probabilities:</strong></p>
            <p>${probabilityDetails}</p>
        </div>
    `;
    
    // Clean up tensors
    inputData.dispose();
    prediction.dispose();
}

// Function to try a random example
function tryRandomExample() {
    if (!irisDataset) {
        alert('Please load the dataset first!');
        return;
    }
    
    const randomIndex = Math.floor(Math.random() * irisDataset.length);
    const randomSample = irisDataset[randomIndex];
    
    document.getElementById('sepalLength').value = randomSample[0];
    document.getElementById('sepalWidth').value = randomSample[1];
    document.getElementById('petalLength').value = randomSample[2];
    document.getElementById('petalWidth').value = randomSample[3];
    
    // Show what the actual species is in the console
    console.log(`Random example: ${randomSample[4]} (actual species)`);
}