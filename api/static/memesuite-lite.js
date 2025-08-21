/**
 * memesuite-lite.js - JavaScript implementation of MEME suite algorithms
 * 
 * A JavaScript port of the Python memesuite-lite library providing FIMO and TOMTOM
 * algorithms for biological sequence analysis.
 * 
 * Original Python library: https://github.com/jmschrei/memesuite-lite
 * 
 * @author Port to JavaScript
 * @version 1.0.0
 */

class MemeSuiteLite {
    constructor() {
        this.version = '1.0.0';
    }

    // =====================================================
    // UTILITY FUNCTIONS
    // =====================================================

    /**
     * Convert a string sequence to one-hot encoding
     * @param {string} sequence - DNA sequence string
     * @param {Array<string>} alphabet - Alphabet characters ['A', 'C', 'G', 'T']
     * @param {Array<string>} ignore - Characters to ignore ['N']
     * @returns {Array<Array<number>>} - One-hot encoded matrix [alphabet_size x sequence_length]
     */
    oneHotEncode(sequence, alphabet = ['A', 'C', 'G', 'T'], ignore = ['N']) {
        // Validate inputs
        for (let char of ignore) {
            if (alphabet.includes(char)) {
                throw new Error(`Character ${char} in alphabet and also in ignored characters`);
            }
        }

        const n = sequence.length;
        const m = alphabet.length;
        const oneHot = Array(m).fill().map(() => Array(n).fill(0));

        // Create mapping
        const charMap = new Map();
        alphabet.forEach((char, idx) => charMap.set(char, idx));
        ignore.forEach(char => charMap.set(char, -1));

        // Fill one-hot encoding
        for (let i = 0; i < n; i++) {
            const char = sequence[i].toUpperCase();
            const idx = charMap.get(char);
            
            if (idx === undefined) {
                throw new Error(`Character ${char} not in alphabet or ignore list`);
            }
            
            if (idx >= 0) {
                oneHot[idx][i] = 1;
            }
        }

        return oneHot;
    }

    /**
     * Convert PWM/one-hot encoding to character sequence
     * @param {Array<Array<number>>} pwm - Position weight matrix [alphabet_size x sequence_length]
     * @param {Array<string>} alphabet - Alphabet characters
     * @param {boolean} force - Force sequence even with ties
     * @returns {string} - Character sequence
     */
    characters(pwm, alphabet = ['A', 'C', 'G', 'T'], force = false) {
        if (pwm.length !== alphabet.length) {
            throw new Error('PWM alphabet size must match provided alphabet');
        }

        const seqLength = pwm[0].length;
        const result = [];

        for (let pos = 0; pos < seqLength; pos++) {
            let maxVal = -Infinity;
            let maxIdx = 0;
            let tieCount = 0;

            // Find maximum value and check for ties
            for (let i = 0; i < alphabet.length; i++) {
                if (pwm[i][pos] > maxVal) {
                    maxVal = pwm[i][pos];
                    maxIdx = i;
                    tieCount = 1;
                } else if (pwm[i][pos] === maxVal) {
                    tieCount++;
                }
            }

            if (tieCount > 1 && !force) {
                throw new Error('Position has multiple characters with same probability');
            }

            result.push(alphabet[maxIdx]);
        }

        return result.join('');
    }

    // =====================================================
    // MEME FILE I/O
    // =====================================================

    /**
     * Read MEME formatted file
     * @param {string} fileContent - Content of MEME file
     * @param {number} maxMotifs - Maximum number of motifs to read
     * @returns {Object} - Dictionary of motif names to PWMs
     */
    readMeme(fileContent, maxMotifs = null) {
        const lines = fileContent.split('\n').map(line => line.trim());
        const motifs = {};
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // Look for motif definition
            if (line.startsWith('MOTIF ')) {
                const motifName = line.replace('MOTIF ', '').trim();
                
                // Find the letter-probability matrix line
                let matrixLine = -1;
                for (let j = i + 1; j < Math.min(i + 10, lines.length); j++) {
                    if (lines[j].startsWith('letter-probability matrix:')) {
                        matrixLine = j;
                        break;
                    }
                }
                
                if (matrixLine !== -1) {
                    const match = lines[matrixLine].match(/w=\s*(\d+)/);
                    if (match) {
                        const width = parseInt(match[1]);
                        const pwmRows = [];
                        
                        // Read PWM data
                        for (let j = matrixLine + 1; j < lines.length && pwmRows.length < width; j++) {
                            const values = lines[j].split(/\s+/).map(parseFloat).filter(x => !isNaN(x));
                            if (values.length === 4) {
                                pwmRows.push(values);
                            }
                        }
                        
                        // If we have the right number of rows, create the motif
                        if (pwmRows.length === width) {
                            // Transpose to get [4 x width] format
                            const pwm = Array(4).fill().map(() => Array(width).fill(0));
                            for (let pos = 0; pos < width; pos++) {
                                for (let base = 0; base < 4; base++) {
                                    pwm[base][pos] = pwmRows[pos][base];
                                }
                            }
                            
                            motifs[motifName] = pwm;
                            
                            if (maxMotifs && Object.keys(motifs).length >= maxMotifs) {
                                break;
                            }
                        }
                    }
                }
            }
        }

        return motifs;
    }

    /**
     * Write MEME formatted file
     * @param {Object} motifs - Dictionary of motif names to PWMs
     * @returns {string} - MEME file content
     */
    writeMeme(motifs) {
        let content = "MEME version 4\n\n";
        content += "ALPHABET= ACGT\n\n";
        content += "strands: + -\n\n";
        content += "Background letter frequencies\n";
        content += "A 0.25 C 0.25 G 0.25 T 0.25\n\n";

        for (const [name, pwm] of Object.entries(motifs)) {
            content += `MOTIF ${name}\n`;
            content += `letter-probability matrix: alength= 4 w= ${pwm[0].length} nsites= 1 E= 0\n`;
            
            for (let pos = 0; pos < pwm[0].length; pos++) {
                const row = [pwm[0][pos], pwm[1][pos], pwm[2][pos], pwm[3][pos]];
                content += row.join(' ') + '\n';
            }
            
            content += "URL BLANK\n\n";
        }

        return content;
    }

    // =====================================================
    // MATHEMATICAL UTILITIES
    // =====================================================

    /**
     * Numerically stable log addition in base 2
     * @param {number} x - First log value
     * @param {number} y - Second log value
     * @returns {number} - log2(2^x + 2^y)
     */
    logAddExp2(x, y) {
        if (x === -Infinity && y === -Infinity) {
            return -Infinity;
        }
        if (x === Infinity || y === Infinity) {
            return Infinity;
        }

        const vmax = Math.max(x, y);
        const vmin = Math.min(x, y);
        return vmax + Math.log2(Math.pow(2, vmin - vmax) + 1);
    }

    /**
     * Calculate binned median approximation
     * @param {Array<number>} values - Array of values
     * @param {number} minVal - Minimum value
     * @param {number} maxVal - Maximum value
     * @param {Array<number>} counts - Count weights
     * @param {number} nBins - Number of bins
     * @returns {number} - Approximate median
     */
    binnedMedian(values, minVal, maxVal, counts, nBins = 1000) {
        const bins = Array(nBins).fill().map(() => [0, 0]); // [count, sum]
        let totalWeight = 0;
        const range = maxVal - minVal;

        // Handle case where all values are the same
        if (range === 0) {
            return minVal;
        }

        for (let i = 0; i < values.length; i++) {
            const binIdx = Math.min(Math.floor((values[i] - minVal) / range * (nBins - 1)), nBins - 1);
            bins[binIdx][0] += counts[i];
            bins[binIdx][1] += values[i] * counts[i];
            totalWeight += counts[i];
        }

        const halfway = totalWeight / 2;
        let cumCount = 0;

        for (let i = 0; i < nBins; i++) {
            cumCount += bins[i][0];
            if (cumCount >= halfway) {
                return bins[i][0] > 0 ? bins[i][1] / bins[i][0] : minVal;
            }
        }

        return maxVal;
    }

    // =====================================================
    // FIMO ALGORITHM
    // =====================================================

    /**
     * Calculate PWM to p-value mapping using dynamic programming
     * @param {Array<Array<number>>} logPwm - Log PWM matrix
     * @param {number} binSize - Score discretization bin size
     * @returns {Object} - {smallest: offset, logPdf: p-value array}
     */
    pwmToMapping(logPwm, binSize) {
        const [numChars, length] = [logPwm.length, logPwm[0].length];
        const logBg = Math.log2(0.25);

        // Convert to integer PWM
        const intLogPwm = logPwm.map(row => row.map(val => Math.round(val / binSize)));

        // Calculate score range
        let smallest = Infinity, largest = -Infinity;
        let minCsum = 0, maxCsum = 0;

        for (let pos = 0; pos < length; pos++) {
            let posMin = Infinity, posMax = -Infinity;
            
            for (let char = 0; char < numChars; char++) {
                posMin = Math.min(posMin, intLogPwm[char][pos]);
                posMax = Math.max(posMax, intLogPwm[char][pos]);
            }
            
            minCsum += posMin;
            maxCsum += posMax;
            smallest = Math.min(smallest, minCsum);
            largest = Math.max(largest, maxCsum);
        }

        largest += length;
        const arraySize = largest - smallest + 1;

        // Dynamic programming to calculate score distribution
        let oldLogPdf = Array(arraySize).fill(-Infinity);
        let newLogPdf = Array(arraySize).fill(-Infinity);

        // Initialize with first position
        for (let char = 0; char < numChars; char++) {
            const idx = intLogPwm[char][0] - smallest;
            oldLogPdf[idx] = this.logAddExp2(oldLogPdf[idx], logBg);
        }

        // Process remaining positions
        for (let pos = 1; pos < length; pos++) {
            newLogPdf.fill(-Infinity);
            
            for (let j = 0; j < arraySize; j++) {
                if (oldLogPdf[j] !== -Infinity) {
                    for (let char = 0; char < numChars; char++) {
                        const newIdx = j + intLogPwm[char][pos];
                        if (newIdx >= 0 && newIdx < arraySize) {
                            newLogPdf[newIdx] = this.logAddExp2(
                                newLogPdf[newIdx], 
                                logBg + oldLogPdf[j]
                            );
                        }
                    }
                }
            }
            
            [oldLogPdf, newLogPdf] = [newLogPdf, oldLogPdf];
        }

        // Convert to cumulative (1 - CDF)
        for (let i = arraySize - 2; i >= 0; i--) {
            oldLogPdf[i] = this.logAddExp2(oldLogPdf[i], oldLogPdf[i + 1]);
        }

        return { smallest, logPdf: oldLogPdf };
    }

    /**
     * FIMO algorithm - Find Individual Motif Instances
     * @param {Object} motifs - Dictionary of motif names to PWMs
     * @param {Array<string>} sequences - Array of DNA sequences
     * @param {Object} options - Algorithm options
     * @returns {Array<Object>} - Array of hit objects for each motif
     */
    fimo(motifs, sequences, options = {}) {
        const {
            alphabet = ['A', 'C', 'G', 'T'],
            binSize = 0.1,
            eps = 0.0001,
            threshold = 0.0001,
            reverseComplement = true
        } = options;

        const logThreshold = Math.log2(threshold);
        const results = [];

        // Process each motif
        for (const [motifName, pwm] of Object.entries(motifs)) {
            const motifHits = [];
            
            // Convert PWM to log space
            const logPwm = pwm.map(row => 
                row.map(val => Math.log2(val + eps) - Math.log2(0.25))
            );

            // Calculate p-value mapping
            const { smallest, logPdf } = this.pwmToMapping(logPwm, binSize);
            
            // Find score threshold
            let scoreThreshold = Infinity;
            for (let i = 0; i < logPdf.length; i++) {
                if (logPdf[i] < logThreshold) {
                    scoreThreshold = (i + smallest) * binSize;
                    break;
                }
            }

            // Scan sequences
            for (let seqIdx = 0; seqIdx < sequences.length; seqIdx++) {
                const sequence = sequences[seqIdx].toUpperCase();
                const oneHot = this.oneHotEncode(sequence, alphabet);
                
                // Scan forward strand
                this._scanSequence(oneHot, logPwm, scoreThreshold, binSize, 
                    smallest, logPdf, motifName, seqIdx, '+', motifHits);
                
                // Scan reverse complement
                if (reverseComplement) {
                    const rcOneHot = this._reverseComplementOneHot(oneHot);
                    const rcLogPwm = logPwm.slice().reverse().map(row => row.slice().reverse());
                    this._scanSequence(rcOneHot, rcLogPwm, scoreThreshold, binSize, 
                        smallest, logPdf, motifName, seqIdx, '-', motifHits);
                }
            }
            
            results.push({
                motif_name: motifName,
                hits: motifHits
            });
        }

        return results;
    }

    /**
     * Helper function to scan a sequence with a motif
     * @private
     */
    _scanSequence(oneHot, logPwm, scoreThreshold, binSize, smallest, logPdf, 
                  motifName, seqIdx, strand, hits) {
        const seqLength = oneHot[0].length;
        const motifLength = logPwm[0].length;

        for (let pos = 0; pos <= seqLength - motifLength; pos++) {
            let score = 0;
            
            // Calculate score at this position
            for (let motifPos = 0; motifPos < motifLength; motifPos++) {
                for (let char = 0; char < 4; char++) {
                    if (oneHot[char][pos + motifPos] === 1) {
                        score += logPwm[char][motifPos];
                        break;
                    }
                }
            }

            // Check if score exceeds threshold
            if (score > scoreThreshold) {
                const scoreIdx = Math.floor(score / binSize) - smallest;
                const pValue = scoreIdx < logPdf.length ? Math.pow(2, logPdf[scoreIdx]) : 0;
                
                hits.push({
                    sequence_idx: seqIdx,
                    start: pos,
                    end: pos + motifLength,
                    strand: strand,
                    score: score,
                    p_value: pValue
                });
            }
        }
    }

    /**
     * Create reverse complement of one-hot encoded sequence
     * @private
     */
    _reverseComplementOneHot(oneHot) {
        const rc = oneHot.map(row => row.slice().reverse());
        return [rc[3], rc[2], rc[1], rc[0]]; // T, G, C, A
    }

    // =====================================================
    // TOMTOM ALGORITHM
    // =====================================================

    /**
     * TOMTOM algorithm - motif similarity with p-values
     * @param {Array<Array<Array<number>>>} queries - Query PWMs
     * @param {Array<Array<Array<number>>>} targets - Target PWMs  
     * @param {Object} options - Algorithm options
     * @returns {Object} - Results object with p-values, scores, etc.
     */
    tomtom(queries, targets, options = {}) {
        const {
            nScoreBins = 100,
            nMedianBins = 1000,
            nCache = 100,
            reverseComplement = true
        } = options;

        const nQueries = queries.length;
        const nTargets = targets.length;
        
        // Prepare target set (include reverse complements if needed)
        let allTargets = [...targets];
        if (reverseComplement) {
            const rcTargets = targets.map(pwm => 
                pwm.slice().reverse().map(row => row.slice().reverse())
            );
            allTargets = [...allTargets, ...rcTargets];
        }

        // Initialize result matrices - results are per original target (best of fwd/rc)
        const pValues = Array(nQueries).fill().map(() => Array(nTargets).fill(1));
        const scores = Array(nQueries).fill().map(() => Array(nTargets).fill(0));
        const offsets = Array(nQueries).fill().map(() => Array(nTargets).fill(0));
        const overlaps = Array(nQueries).fill().map(() => Array(nTargets).fill(0));
        const strands = Array(nQueries).fill().map(() => Array(nTargets).fill(0));

        // Process each query
        for (let qIdx = 0; qIdx < nQueries; qIdx++) {
            const query = queries[qIdx];
            
            // Calculate similarity scores against all targets
            for (let tIdx = 0; tIdx < nTargets; tIdx++) {
                let bestAlignment = { pValue: 1, score: 0, offset: 0, overlap: 0 };
                let bestStrand = 0;
                
                // Compare against forward target
                const fwdTarget = targets[tIdx];
                const fwdAlignment = this._calculateAlignment(query, fwdTarget, nScoreBins, nMedianBins);
                
                if (fwdAlignment.pValue < bestAlignment.pValue) {
                    bestAlignment = fwdAlignment;
                    bestStrand = 0;
                }
                
                // Compare against reverse complement if enabled
                if (reverseComplement) {
                    const rcTarget = allTargets[tIdx + nTargets];
                    const rcAlignment = this._calculateAlignment(query, rcTarget, nScoreBins, nMedianBins);
                    
                    if (rcAlignment.pValue < bestAlignment.pValue) {
                        bestAlignment = rcAlignment;
                        bestStrand = 1;
                    }
                }
                
                // Store best result
                pValues[qIdx][tIdx] = bestAlignment.pValue;
                scores[qIdx][tIdx] = bestAlignment.score;
                offsets[qIdx][tIdx] = bestAlignment.offset;
                overlaps[qIdx][tIdx] = bestAlignment.overlap;
                strands[qIdx][tIdx] = bestStrand;
            }
        }

        return {
            pValues,
            scores, 
            offsets,
            overlaps,
            strands
        };
    }

    /**
     * Calculate best alignment between query and target motifs
     * @private
     */
    _calculateAlignment(query, target, nScoreBins, nMedianBins) {
        const queryLength = query[0].length;
        const targetLength = target[0].length;
        
        // Calculate distance matrix
        const distances = [];
        for (let tPos = 0; tPos < targetLength; tPos++) {
            const rowDistances = [];
            for (let qPos = 0; qPos < queryLength; qPos++) {
                let dist = 0;
                for (let char = 0; char < 4; char++) {
                    dist += (query[char][qPos] - target[char][tPos]) ** 2;
                }
                rowDistances.push(-Math.sqrt(dist));
            }
            distances.push(rowDistances);
        }

        // Subtract median from each query position
        for (let qPos = 0; qPos < queryLength; qPos++) {
            const values = distances.map(row => row[qPos]);
            const minVal = Math.min(...values);
            const maxVal = Math.max(...values);
            const counts = Array(values.length).fill(1);
            const median = this.binnedMedian(values, minVal, maxVal, counts, nMedianBins);
            
            for (let tPos = 0; tPos < targetLength; tPos++) {
                distances[tPos][qPos] -= median;
            }
        }

        // Find best alignment
        let bestScore = -Infinity;
        let bestOffset = 0;
        let bestOverlap = 0;

        // Try all possible alignments
        for (let offset = -(queryLength - 1); offset < targetLength; offset++) {
            let score = 0;
            let overlap = 0;
            
            for (let qPos = 0; qPos < queryLength; qPos++) {
                const tPos = qPos + offset;
                if (tPos >= 0 && tPos < targetLength) {
                    score += distances[tPos][qPos];
                    overlap++;
                }
            }
            
            if (score > bestScore) {
                bestScore = score;
                bestOffset = offset;
                bestOverlap = overlap;
            }
        }

        // Convert score to p-value (simplified)
        const pValue = Math.exp(-Math.abs(bestScore) / 100); // Placeholder calculation

        return {
            score: bestScore,
            offset: bestOffset,
            overlap: bestOverlap,
            pValue: Math.max(1e-15, pValue)
        };
    }

    // =====================================================
    // PUBLIC API METHODS
    // =====================================================

    /**
     * Run FIMO analysis on sequences
     * @param {string} memeContent - MEME file content
     * @param {Array<string>} sequences - DNA sequences
     * @param {Object} options - Options
     * @returns {Array<Object>} - FIMO results
     */
    runFimo(memeContent, sequences, options = {}) {
        const motifs = this.readMeme(memeContent);
        return this.fimo(motifs, sequences, options);
    }

    /**
     * Run TOMTOM analysis between motif sets
     * @param {string} queryMemeContent - Query MEME file
     * @param {string} targetMemeContent - Target MEME file  
     * @param {Object} options - Options
     * @returns {Object} - TOMTOM results
     */
    runTomtom(queryMemeContent, targetMemeContent, options = {}) {
        const queryMotifs = this.readMeme(queryMemeContent);
        const targetMotifs = this.readMeme(targetMemeContent);
        
        const queryPwms = Object.values(queryMotifs);
        const targetPwms = Object.values(targetMotifs);
        
        return this.tomtom(queryPwms, targetPwms, options);
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MemeSuiteLite;
} else if (typeof window !== 'undefined') {
    window.MemeSuiteLite = MemeSuiteLite;
}