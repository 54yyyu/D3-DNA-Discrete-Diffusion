// D3-DNA API Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize form controls
    initializeFormControls();
    
    // Bind form submission
    document.getElementById('generation-form').addEventListener('submit', handleFormSubmission);
    
    // Bind toggle buttons
    document.getElementById('toggle-json').addEventListener('click', toggleJsonOutput);
    
    // Bind unconditional checkbox
    document.getElementById('unconditional').addEventListener('change', toggleConditioningControls);
    
    // Bind mode selection
    document.getElementById('mode').addEventListener('change', toggleEvaluationControls);
});

function initializeFormControls() {
    // Initialize range sliders with value display
    const rangeInputs = ['num_samples', 'steps'];
    rangeInputs.forEach(inputId => {
        const input = document.getElementById(inputId);
        const valueSpan = document.getElementById(inputId + '_value');
        
        input.addEventListener('input', function() {
            valueSpan.textContent = this.value;
        });
    });
    
    // Set initial conditioning controls state
    toggleConditioningControls();
    toggleEvaluationControls();
}

function toggleConditioningControls() {
    const unconditional = document.getElementById('unconditional').checked;
    const conditioningControls = document.getElementById('conditioning-controls');
    
    if (unconditional) {
        conditioningControls.style.opacity = '0.5';
        conditioningControls.style.pointerEvents = 'none';
    } else {
        conditioningControls.style.opacity = '1';
        conditioningControls.style.pointerEvents = 'auto';
    }
}

function toggleEvaluationControls() {
    const mode = document.getElementById('mode').value;
    
    // Find the evaluation options section
    const paramGroups = document.querySelectorAll('.param-group');
    let evaluationGroup = null;
    
    paramGroups.forEach(group => {
        const heading = group.querySelector('h3');
        if (heading && heading.textContent.includes('Evaluation Options')) {
            evaluationGroup = group;
        }
    });
    
    if (evaluationGroup) {
        if (mode === 'sampling') {
            evaluationGroup.style.opacity = '0.5';
            evaluationGroup.style.pointerEvents = 'none';
        } else {
            evaluationGroup.style.opacity = '1';
            evaluationGroup.style.pointerEvents = 'auto';
        }
    }
}

async function handleFormSubmission(event) {
    event.preventDefault();
    
    // Show loading state
    showLoading();
    
    try {
        // Collect form data
        const formData = collectFormData();
        
        // Make API request
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Request failed with status ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

function collectFormData() {
    const formData = {
        // Basic settings
        num_samples: parseInt(document.getElementById('num_samples').value),
        steps: parseInt(document.getElementById('steps').value),
        dataset: document.getElementById('dataset').value,
        mode: document.getElementById('mode').value,
        
        // Conditioning
        unconditional: document.getElementById('unconditional').checked,
        
        // Evaluation options
        include_oracle: document.getElementById('include_oracle').checked,
        include_sp_mse: document.getElementById('include_sp_mse').checked,
        
        // Visualization options
        include_visualization: document.getElementById('include_visualization').checked,
        save_intermediate_steps: document.getElementById('save_intermediate_steps').checked
    };
    
    // Add optional conditioning values
    if (!formData.unconditional) {
        const devActivity = document.getElementById('dev_activity').value;
        const hkActivity = document.getElementById('hk_activity').value;
        
        if (devActivity) {
            formData.dev_activity = parseFloat(devActivity);
        }
        if (hkActivity) {
            formData.hk_activity = parseFloat(hkActivity);
        }
    }
    
    // Add optional evaluation parameters
    const specificIndices = document.getElementById('specific_indices').value;
    if (specificIndices) {
        formData.specific_indices = specificIndices;
    }
    
    const maxSamples = document.getElementById('max_samples').value;
    if (maxSamples) {
        formData.max_samples = parseInt(maxSamples);
    }
    
    return formData;
}

function displayResults(data) {
    // Show results section
    document.getElementById('results').classList.remove('hidden');
    
    // Display metadata preview
    displayMetadataPreview(data.metadata, data.generation_time, data.sp_mse);
    
    // Display sequences preview
    displaySequencesPreview(data.final_sequences);
    
    // Display steps preview
    displayStepsPreview(data.steps);
    
    // Display oracle predictions if available
    displayOraclePreview(data.steps, data.metadata);
    
    // Display complete JSON
    displayCompleteJson(data);
}

function displayMetadataPreview(metadata, generationTime, spMse) {
    const preview = document.getElementById('metadata-preview');
    
    let html = `
        <div class="metadata-grid">
            <div><strong>Dataset:</strong> ${metadata.dataset}</div>
            <div><strong>Mode:</strong> ${document.getElementById('mode').value}</div>
            <div><strong>Samples:</strong> ${metadata.num_samples}</div>
            <div><strong>Steps:</strong> ${metadata.total_steps}</div>
            <div><strong>Generation Time:</strong> ${generationTime.toFixed(2)}s</div>
            <div><strong>Sequence Length:</strong> ${metadata.sequence_length}</div>
        </div>
    `;
    
    if (spMse !== undefined && spMse !== null) {
        html += `<div class="sp-mse"><strong>SP-MSE:</strong> ${spMse.toFixed(6)}</div>`;
    }
    
    preview.innerHTML = html;
}

function displaySequencesPreview(sequences) {
    const preview = document.getElementById('sequences-preview');
    const maxDisplay = Math.min(3, sequences.length);
    
    let html = '';
    for (let i = 0; i < maxDisplay; i++) {
        const seq = sequences[i];
        const truncated = seq.length > 80 ? seq.substring(0, 80) + '...' : seq;
        html += `<div class="sequence-item">
            <div class="sequence-header">Sequence ${i + 1}:</div>
            <div class="sequence-data">${truncated}</div>
        </div>`;
    }
    
    if (sequences.length > maxDisplay) {
        html += `<div class="more-info">... and ${sequences.length - maxDisplay} more sequences</div>`;
    }
    
    preview.innerHTML = html;
}

function displayStepsPreview(steps) {
    const preview = document.getElementById('steps-preview');
    const maxDisplay = Math.min(2, steps.length);
    
    let html = '';
    for (let i = 0; i < maxDisplay; i++) {
        const step = steps[i];
        const firstSeq = step.sequences[0] ? step.sequences[0].slice(0, 10).join(',') + '...' : 'N/A';
        
        html += `<div class="step-item">
            <div class="step-header">Step ${step.step}:</div>
            <div class="step-data">
                timestep=${step.timestep.toFixed(2)}, 
                noise=${step.noise_level.toFixed(1)}, 
                sequences=[${firstSeq}]
            </div>
        </div>`;
    }
    
    if (steps.length > maxDisplay) {
        html += `<div class="more-info">... and ${steps.length - maxDisplay} more steps</div>`;
    }
    
    preview.innerHTML = html;
}

function displayOraclePreview(steps, metadata) {
    const oracleGroup = document.getElementById('oracle-group');
    const preview = document.getElementById('oracle-preview');
    
    // Check if we have oracle data
    const hasOracleData = steps.some(step => step.oracle_predictions && step.oracle_predictions.length > 0);
    
    if (!hasOracleData) {
        oracleGroup.classList.add('hidden');
        return;
    }
    
    oracleGroup.classList.remove('hidden');
    
    // Find the last step with oracle predictions
    let lastStepWithOracle = null;
    for (let i = steps.length - 1; i >= 0; i--) {
        if (steps[i].oracle_predictions && steps[i].oracle_predictions.length > 0) {
            lastStepWithOracle = steps[i];
            break;
        }
    }
    
    if (!lastStepWithOracle) {
        preview.innerHTML = '<div class="no-data">No oracle predictions available</div>';
        return;
    }
    
    const maxDisplay = Math.min(3, lastStepWithOracle.oracle_predictions.length);
    let html = '';
    
    for (let i = 0; i < maxDisplay; i++) {
        const prediction = lastStepWithOracle.oracle_predictions[i];
        const mse = lastStepWithOracle.oracle_mses && lastStepWithOracle.oracle_mses[i] ? 
                   lastStepWithOracle.oracle_mses[i].toFixed(3) : 'N/A';
        
        html += `<div class="oracle-item">
            <div class="oracle-header">Sample ${i + 1}:</div>
            <div class="oracle-data">
                Dev=${prediction[0].toFixed(2)}, 
                HK=${prediction[1].toFixed(2)} | 
                MSE=${mse}
            </div>
        </div>`;
    }
    
    if (lastStepWithOracle.oracle_predictions.length > maxDisplay) {
        html += `<div class="more-info">... and ${lastStepWithOracle.oracle_predictions.length - maxDisplay} more predictions</div>`;
    }
    
    preview.innerHTML = html;
}

function displayCompleteJson(data) {
    const jsonOutput = document.getElementById('json-output');
    jsonOutput.textContent = JSON.stringify(data, null, 2);
}

function toggleJsonOutput() {
    const jsonOutput = document.getElementById('json-output');
    const toggleBtn = document.getElementById('toggle-json');
    
    if (jsonOutput.classList.contains('hidden')) {
        jsonOutput.classList.remove('hidden');
        toggleBtn.textContent = 'Hide Full JSON';
    } else {
        jsonOutput.classList.add('hidden');
        toggleBtn.textContent = 'Show Full JSON';
    }
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('error-message').classList.add('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('generate-btn').disabled = true;
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('generate-btn').disabled = false;
}

function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = `Error: ${message}`;
    errorDiv.classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
}