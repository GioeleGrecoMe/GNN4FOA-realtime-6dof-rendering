// GNN4FOA v2
// Authors: Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci
// Image and Sound Processing Lab (ISPL), Politecnico di Milano
// Copyright (c) 2026 Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci

#include "DiracGNNEngine.h"
#include "GNNWeights.h"

#include <algorithm>
#include <cmath>
#include <limits>

#if GNN4FOA_HAS_VDSP
 #include <Accelerate/Accelerate.h>
#endif

namespace
{
constexpr float kEps = 1.0e-8f;
constexpr float kPi = 3.14159265358979323846f;

inline bool finitePositive(float v) noexcept
{
    return std::isfinite(v) && v > 0.0f;
}

// Dense row-major GEMM C = A * B where B is the pre-transposed learned
// weight matrix. On macOS we deliberately use vDSP_mmul rather than CBLAS:
// vDSP_mmul is independent of LAPACK/BLAS symbol-versioning and is available
// on the Accelerate runtime available on the oldest supported macOS target.
using RowMatrixXfLocal = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void multiplyPreparedWeights(const RowMatrixXfLocal& a,
                             int rows,
                             int inner,
                             const RowMatrixXfLocal& b,
                             RowMatrixXfLocal& c,
                             float* packedScratch,
                             size_t packedCapacity) noexcept
{
    const int outputCols = (int) b.cols();
    jassert(b.rows() == inner);
    jassert(rows <= a.rows() && inner <= a.cols());
    jassert(rows <= c.rows() && outputCols <= c.cols());

    // All GNN workspaces are row-major. The second multiplication of every
    // MLP is naturally tightly packed (48 columns). For first-layer inputs,
    // some shared buffers are wider than the active input dimension; vDSP has
    // no leading-dimension parameter, so pack those rows into a fixed
    // thread-local scratch buffer. This allocates only on first use of the
    // dedicated worker thread, never in the audio callback.
    const float* aPtr = a.data();
    if (a.cols() != inner)
    {
        const size_t needed = (size_t) rows * (size_t) inner;
        jassert(packedScratch != nullptr && packedCapacity >= needed);
        if (packedScratch == nullptr || packedCapacity < needed)
            return;
        for (int r = 0; r < rows; ++r)
            std::copy_n(a.data() + (size_t) r * (size_t) a.cols(),
                        inner,
                        packedScratch + (size_t) r * (size_t) inner);
        aPtr = packedScratch;
    }

#if GNN4FOA_HAS_VDSP
    vDSP_mmul(aPtr, 1,
              b.data(), 1,
              c.data(), 1,
              (vDSP_Length) rows,
              (vDSP_Length) outputCols,
              (vDSP_Length) inner);
#else
    for (int r = 0; r < rows; ++r)
        for (int out = 0; out < outputCols; ++out)
        {
            float acc = 0.0f;
            for (int k = 0; k < inner; ++k)
                acc += aPtr[(size_t) r * (size_t) inner + (size_t) k] * b(k, out);
            c(r, out) = acc;
        }
#endif
}
}

DiracGNNEngine::DiracGNNEngine()
    : forwardFFT(9),
      inverseFFT(9),
      window(512, juce::dsp::WindowingFunction<float>::hann, false, 0.0f)
{
    numLayers = GNN_NUM_LAYERS;

    // GNNWeights.h is MATLAB column-major data. Eigen defaults to column-major,
    // therefore the maps preserve the exported layout exactly. Bias arrays are
    // vectors even though the generated *_cols metadata is one.
    outLayer.W1T = Eigen::Map<const Eigen::MatrixXf>(out_W1_data, out_W1_rows, out_W1_cols).transpose();
    outLayer.b1 = Eigen::Map<const Eigen::RowVectorXf>(out_b1_data, out_b1_rows * out_b1_cols);
    outLayer.W2T = Eigen::Map<const Eigen::MatrixXf>(out_W2_data, out_W2_rows, out_W2_cols).transpose();
    outLayer.b2 = Eigen::Map<const Eigen::RowVectorXf>(out_b2_data, out_b2_rows * out_b2_cols);

    msgLayers.resize((size_t) numLayers);
    updLayers.resize((size_t) numLayers);

    if (numLayers >= 1)
    {
        msgLayers[0].W1T = Eigen::Map<const Eigen::MatrixXf>(msg_1_W1_data, msg_1_W1_rows, msg_1_W1_cols).transpose();
        msgLayers[0].b1 = Eigen::Map<const Eigen::RowVectorXf>(msg_1_b1_data, msg_1_b1_rows * msg_1_b1_cols);
        msgLayers[0].W2T = Eigen::Map<const Eigen::MatrixXf>(msg_1_W2_data, msg_1_W2_rows, msg_1_W2_cols).transpose();
        msgLayers[0].b2 = Eigen::Map<const Eigen::RowVectorXf>(msg_1_b2_data, msg_1_b2_rows * msg_1_b2_cols);

        updLayers[0].W1T = Eigen::Map<const Eigen::MatrixXf>(upd_1_W1_data, upd_1_W1_rows, upd_1_W1_cols).transpose();
        updLayers[0].b1 = Eigen::Map<const Eigen::RowVectorXf>(upd_1_b1_data, upd_1_b1_rows * upd_1_b1_cols);
        updLayers[0].W2T = Eigen::Map<const Eigen::MatrixXf>(upd_1_W2_data, upd_1_W2_rows, upd_1_W2_cols).transpose();
        updLayers[0].b2 = Eigen::Map<const Eigen::RowVectorXf>(upd_1_b2_data, upd_1_b2_rows * upd_1_b2_cols);
    }

    if (numLayers >= 2)
    {
        msgLayers[1].W1T = Eigen::Map<const Eigen::MatrixXf>(msg_2_W1_data, msg_2_W1_rows, msg_2_W1_cols).transpose();
        msgLayers[1].b1 = Eigen::Map<const Eigen::RowVectorXf>(msg_2_b1_data, msg_2_b1_rows * msg_2_b1_cols);
        msgLayers[1].W2T = Eigen::Map<const Eigen::MatrixXf>(msg_2_W2_data, msg_2_W2_rows, msg_2_W2_cols).transpose();
        msgLayers[1].b2 = Eigen::Map<const Eigen::RowVectorXf>(msg_2_b2_data, msg_2_b2_rows * msg_2_b2_cols);

        updLayers[1].W1T = Eigen::Map<const Eigen::MatrixXf>(upd_2_W1_data, upd_2_W1_rows, upd_2_W1_cols).transpose();
        updLayers[1].b1 = Eigen::Map<const Eigen::RowVectorXf>(upd_2_b1_data, upd_2_b1_rows * upd_2_b1_cols);
        updLayers[1].W2T = Eigen::Map<const Eigen::MatrixXf>(upd_2_W2_data, upd_2_W2_rows, upd_2_W2_cols).transpose();
        updLayers[1].b2 = Eigen::Map<const Eigen::RowVectorXf>(upd_2_b2_data, upd_2_b2_rows * upd_2_b2_cols);
    }

    auto validateLayer = [](const MLP& layer)
    {
        jassert(layer.W1T.cols() == layer.b1.cols());
        jassert(layer.W2T.cols() == layer.b2.cols());
        jassert(layer.W2T.rows() == layer.W1T.cols());
    };

    validateLayer(outLayer);
    for (int l = 0; l < numLayers; ++l)
    {
        validateLayer(msgLayers[(size_t) l]);
        validateLayer(updLayers[(size_t) l]);
    }

    initialFeatureDim = (msgLayers.front().W1T.rows() - 4) / 2;
    maxHiddenDim = outLayer.W1T.cols();
    maxOutInputDim = outLayer.W1T.rows();

    for (int l = 0; l < numLayers; ++l)
    {
        maxHiddenDim = std::max(maxHiddenDim, (int) msgLayers[(size_t) l].W2T.cols());
        maxHiddenDim = std::max(maxHiddenDim, (int) updLayers[(size_t) l].W2T.cols());
        maxMessageInputDim = std::max(maxMessageInputDim, (int) msgLayers[(size_t) l].W1T.rows());
        maxUpdateInputDim = std::max(maxUpdateInputDim, (int) updLayers[(size_t) l].W1T.rows());
    }

    // Network architecture: 5 node features, 4 edge features, two 48-D hidden
    // states and a 4-D edge in the readout -> 48 + 48 + 4 = 100.
    jassert(initialFeatureDim == 5);
    jassert(maxOutInputDim == 100);
}

DiracGNNEngine::~DiracGNNEngine() = default;

void DiracGNNEngine::prepare(double sampleRate, int)
{
    fs = sampleRate;
    jassert(std::abs(fs - kModelSampleRate) < 1.0);
    numBins = nfft / 2 + 1;

    const int ringSize = nfft * 4;
    inputRingBuffer.setSize(kNumInputChannels, ringSize, false, true, false);
    outputRingBuffer.setSize(kNumOutputChannels, ringSize, false, true, false);
    outputNormRingBuffer.setSize(1, ringSize, false, true, false);

    fftWorkspace.assign((size_t) nfft * 2u, 0.0f);
    windowCoefficients.assign((size_t) winLen, 1.0f);
    window.multiplyWithWindowingTable(windowCoefficients.data(), winLen);

    spectra.assign((size_t) kNumInputChannels * (size_t) numBins, {});
    querySpectra.assign((size_t) kNumOutputChannels * (size_t) numBins, {});
    previousResidualSpectrum.assign((size_t) kNumOutputChannels * (size_t) numBins, {});
    previousResidualValid.assign((size_t) numBins, 0u);
    useGNNForBin.assign((size_t) numBins, 0u);

    E_obs = Eigen::MatrixXf::Zero(numBins, kNumMics);
    psi_obs = Eigen::MatrixXf::Ones(numBins, kNumMics);
    u_obs_X = Eigen::MatrixXf::Zero(numBins, kNumMics);
    u_obs_Y = Eigen::MatrixXf::Zero(numBins, kNumMics);
    u_obs_Z = Eigen::MatrixXf::Zero(numBins, kNumMics);

    covWW = Eigen::MatrixXf::Zero(numBins, kNumMics);
    covYY = Eigen::MatrixXf::Zero(numBins, kNumMics);
    covZZ = Eigen::MatrixXf::Zero(numBins, kNumMics);
    covXX = Eigen::MatrixXf::Zero(numBins, kNumMics);
    covWX = Eigen::MatrixXcf::Zero(numBins, kNumMics);
    covWY = Eigen::MatrixXcf::Zero(numBins, kNumMics);
    covWZ = Eigen::MatrixXcf::Zero(numBins, kNumMics);

    for (auto& h : hiddenA)
        h = RowMatrixXf::Zero(numBins, maxHiddenDim);
    for (auto& h : hiddenB)
        h = RowMatrixXf::Zero(numBins, maxHiddenDim);

    aggregateScratch = RowMatrixXf::Zero(numBins, maxHiddenDim);
    edgeFeatureScratch = RowMatrixXf::Zero(numBins, 4);
    messageInputScratch = RowMatrixXf::Zero(numBins, maxMessageInputDim);
    updateInputScratch = RowMatrixXf::Zero(numBins, maxUpdateInputDim);
    outInputScratch = RowMatrixXf::Zero(numBins, maxOutInputDim);
    mlpHiddenScratch = RowMatrixXf::Zero(numBins, maxHiddenDim);
    mlpOutputScratch = RowMatrixXf::Zero(numBins, maxHiddenDim);

    constexpr int directedEdges = kNumNodes * (kNumNodes - 1);
    messageBatchInput = RowMatrixXf::Zero(numBins * directedEdges, maxMessageInputDim);
    messageBatchHidden = RowMatrixXf::Zero(numBins * directedEdges, maxHiddenDim);
    messageBatchOutput = RowMatrixXf::Zero(numBins * directedEdges, maxHiddenDim);
    updateBatchInput = RowMatrixXf::Zero(numBins * kNumNodes, maxUpdateInputDim);
    updateBatchHidden = RowMatrixXf::Zero(numBins * kNumNodes, maxHiddenDim);
    updateBatchOutput = RowMatrixXf::Zero(numBins * kNumNodes, maxHiddenDim);
    outBatchInput = RowMatrixXf::Zero(numBins * kNumMics, maxOutInputDim);
    outBatchHidden = RowMatrixXf::Zero(numBins * kNumMics, maxHiddenDim);
    outBatchOutput = RowMatrixXf::Zero(numBins * kNumMics, 1);

    const size_t maxGemmRows = (size_t) numBins * (size_t) directedEdges;
    const size_t maxGemmInput = (size_t) std::max({ maxMessageInputDim, maxUpdateInputDim, maxOutInputDim, maxHiddenDim });
    gemmPackScratch.assign(maxGemmRows * maxGemmInput, 0.0f);

    weightsRaw = Eigen::MatrixXf::Zero(numBins, kNumMics);
    weightsPreConstraint = Eigen::MatrixXf::Zero(numBins, kNumMics);
    weightsGNNConstrained = Eigen::MatrixXf::Zero(numBins, kNumMics);
    weightsCandidate = Eigen::MatrixXf::Zero(numBins, kNumMics);
    weightsDelaunay = Eigen::MatrixXf::Zero(numBins, kNumMics);
    weightsUsed = Eigen::MatrixXf::Zero(numBins, kNumMics);
    lastValidGNNWeights = Eigen::MatrixXf::Zero(numBins, kNumMics);
    gnnBlendFrom = Eigen::MatrixXf::Zero(numBins, kNumMics);
    gnnBlendTo = Eigen::MatrixXf::Zero(numBins, kNumMics);
    lastInferenceE = Eigen::MatrixXf::Zero(numBins, kNumMics);
    lastInferencePsi = Eigen::MatrixXf::Ones(numBins, kNumMics);
    lastInferenceUX = Eigen::MatrixXf::Zero(numBins, kNumMics);
    lastInferenceUY = Eigen::MatrixXf::Zero(numBins, kNumMics);
    lastInferenceUZ = Eigen::MatrixXf::Zero(numBins, kNumMics);

    for (auto& history : weightHistory)
        history = Eigen::MatrixXf::Zero(numBins, kNumMics);
    for (auto& valid : weightHistoryValid)
        valid.fill(0u);

    reset();
}

void DiracGNNEngine::reset() noexcept
{
    inputRingBuffer.clear();
    outputRingBuffer.clear();
    outputNormRingBuffer.clear();

    inputWritePointer = 0;
    outputReadPointer = 0;
    samplesReceived = 0;

    if (E_obs.size() != 0)
    {
        E_obs.setZero();
        psi_obs.setOnes();
        u_obs_X.setZero();
        u_obs_Y.setZero();
        u_obs_Z.setZero();
        covWW.setZero();
        covYY.setZero();
        covZZ.setZero();
        covXX.setZero();
        covWX.setZero();
        covWY.setZero();
        covWZ.setZero();
        lastValidGNNWeights.setZero();
        gnnBlendFrom.setZero();
        gnnBlendTo.setZero();
        lastInferenceE.setZero();
        lastInferencePsi.setOnes();
        lastInferenceUX.setZero();
        lastInferenceUY.setZero();
        lastInferenceUZ.setZero();
    }

    validGNNBinMask.fill(0u);
    previousUseGNNForBin.fill(0u);
    lastInferenceFeatureValid.fill(0u);
    reliableBinIndices.fill(0);
    coverageBinIndices.fill(0);
    gnnGateMix.fill(0.0f);
    std::fill(previousResidualSpectrum.begin(), previousResidualSpectrum.end(), std::complex<float>{});
    std::fill(previousResidualValid.begin(), previousResidualValid.end(), 0u);
    std::fill(useGNNForBin.begin(), useGNNForBin.end(), 0u);

    resetWeightHistory();
    lastInterpolationMode = -1;
    hasLastValidGNNWeights = false;
    currentAnchor = -1;
    previousAnchor = -1;
    anchorCrossfadeRemaining = 0;
    lastMicAzimuth.fill(0.0f);
    micAzimuthStateInitialised = false;
    invalidGNNFrames.store(0, std::memory_order_relaxed);
    reusedGNNFrames.store(0, std::memory_order_relaxed);
    lastMaxGNNWeight.store(0.0f, std::memory_order_relaxed);
    lastGNNStage.store(0, std::memory_order_relaxed);
    lastGNNForwardMs.store(0.0, std::memory_order_relaxed);
    gnnForwardEwmaMs.store(0.0, std::memory_order_relaxed);
    hasRuntimeGNNForwardTiming = false;
    gnnInferenceStride.store(4, std::memory_order_relaxed);
    lastGNNActiveBinCount.store(0, std::memory_order_relaxed);
    gnnInferenceCount.store(0, std::memory_order_relaxed);
    gnnCadenceHeldFrames.store(0, std::memory_order_relaxed);
    gnnPressureDeferredFrames.store(0, std::memory_order_relaxed);
    gnnGeometryForcedInferences.store(0, std::memory_order_relaxed);
    gnnAcousticForcedInferences.store(0, std::memory_order_relaxed);
    gnnGateEntries.store(0, std::memory_order_relaxed);
    gnnGateExits.store(0, std::memory_order_relaxed);
    gnnCoverageOnlyInferences.store(0, std::memory_order_relaxed);
    gnnFullInferences.store(0, std::memory_order_relaxed);
    lastGNNInferredBinCount.store(0, std::memory_order_relaxed);
    lastGNNCoverageBinCount.store(0, std::memory_order_relaxed);
    lastPredictedGNNForwardMs.store(0.0, std::memory_order_relaxed);
    gnnForwardPerBinEwmaMs.store(0.0, std::memory_order_relaxed);
    gnnForwardPerBinPeakMs.store(0.0, std::memory_order_relaxed);
    lastAcousticNovelty.store(0.0f, std::memory_order_relaxed);
    gnnInferenceCountdown = 0;
    gnnBlendRemaining = 0;
    weightVizCadenceCountdown = 0;
    weightVizHasPublished = false;
    lastActiveMicCount = kNumMics;
    schedulerGeometryInitialised = false;
    lastSchedulerMicX.fill(0.0f);
    lastSchedulerMicY.fill(0.0f);
    lastSchedulerMicZ.fill(0.0f);
    lastSchedulerMicAzimuth.fill(0.0f);
    lastSchedulerListenerX = lastSchedulerListenerY = lastSchedulerListenerZ = 0.0f;
    weightVizVersion.store(0u, std::memory_order_relaxed);
    weightVizFrameCounter.store(0u, std::memory_order_relaxed);
    weightVizMeanRawSum.store(0.0f, std::memory_order_relaxed);
    weightVizMeanAppliedSum.store(0.0f, std::memory_order_relaxed);
    weightVizMaxRawWeight.store(0.0f, std::memory_order_relaxed);
    weightVizMeanScale.store(1.0f, std::memory_order_relaxed);
    weightVizMaxScale.store(1.0f, std::memory_order_relaxed);
    weightVizLowClampFraction.store(weightVizEmaLowClampFraction, std::memory_order_relaxed);
    weightVizHighClampFraction.store(weightVizEmaHighClampFraction, std::memory_order_relaxed);
    weightVizGNNBinFraction.store(0.0f, std::memory_order_relaxed);
    weightVizGnnMode.store(false, std::memory_order_relaxed);
    for (int m = 0; m < kNumMics; ++m)
    {
        weightVizRawMean[(size_t) m].store(0.0f, std::memory_order_relaxed);
        weightVizGNNRelativeMean[(size_t) m].store(0.0f, std::memory_order_relaxed);
        weightVizDelaunayRelativeMean[(size_t) m].store(0.0f, std::memory_order_relaxed);
        weightVizRelativeMean[(size_t) m].store(0.0f, std::memory_order_relaxed);
        weightVizRelativeLow[(size_t) m].store(0.0f, std::memory_order_relaxed);
        weightVizRelativeMid[(size_t) m].store(0.0f, std::memory_order_relaxed);
        weightVizRelativeHigh[(size_t) m].store(0.0f, std::memory_order_relaxed);
        weightVizEmaRawMean[(size_t) m] = 0.0f;
        weightVizEmaGNNRelativeMean[(size_t) m] = 0.0f;
        weightVizEmaDelaunayRelativeMean[(size_t) m] = 0.0f;
        weightVizEmaRelativeMean[(size_t) m] = 0.0f;
        weightVizEmaRelativeLow[(size_t) m] = 0.0f;
        weightVizEmaRelativeMid[(size_t) m] = 0.0f;
        weightVizEmaRelativeHigh[(size_t) m] = 0.0f;
    }
    weightVizEmaMeanRawSum = 0.0f;
    weightVizEmaMeanAppliedSum = 0.0f;
    weightVizEmaMeanScale = 1.0f;
    weightVizEmaMaxScale = 1.0f;
    weightVizEmaLowClampFraction = 0.0f;
    weightVizEmaHighClampFraction = 0.0f;
    weightVizEmaGNNBinFraction = 0.0f;
    weightVizEmaInitialised = false;
}

void DiracGNNEngine::process(const juce::AudioBuffer<float>& inputBuffer,
                             juce::AudioBuffer<float>& outputBuffer,
                             float listenerX,
                             float listenerY,
                             float listenerZ,
                             const std::array<float, kNumMics>& micX,
                             const std::array<float, kNumMics>& micY,
                             const std::array<float, kNumMics>& micZ,
                             const std::array<float, kNumMics>& micAzimuth,
                             float diracSmoothing,
                             float energyThresholdDb,
                             int interpolationMode,
                             int temporalWindow,
                             int activeMicCount,
                             float outputReservoirFill01,
                             float outputReservoirAvailableMs)
{
    if (interpolationMode == 1)
        lastGNNStage.store(1, std::memory_order_relaxed); // STFT frame entered
    const int ringSize = inputRingBuffer.getNumSamples();
    if (ringSize <= 0 || outputRingBuffer.getNumSamples() != ringSize)
        return;

    const int numSamples = inputBuffer.getNumSamples();
    const int availableInputChannels = inputBuffer.getNumChannels();
    const int availableOutputChannels = outputBuffer.getNumChannels();

    diracSmoothing = juce::jlimit(0.0f, 0.9999f, diracSmoothing);
    energyThresholdDb = juce::jlimit(-120.0f, 0.0f, energyThresholdDb);
    interpolationMode = juce::jlimit(0, 1, interpolationMode);
    temporalWindow = juce::jlimit(1, kMaxTemporalWindow, temporalWindow);
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);

    if (activeMicCount != lastActiveMicCount)
    {
        // A host bus-layout change changes the graph itself. Do not carry
        // temporal learned weights or anchor state across two different graphs.
        resetWeightHistory();
        hasLastValidGNNWeights = false;
        lastValidGNNWeights.setZero();
        weightsRaw.setZero();
        weightsPreConstraint.setZero();
        weightsGNNConstrained.setZero();
        currentAnchor = -1;
        previousAnchor = -1;
        anchorCrossfadeRemaining = 0;
        gnnInferenceCountdown = 0;
        const double conservativeFullMs = predictGNNForwardMs(numBins, numBins, true, activeMicCount);
        gnnForwardEwmaMs.store(conservativeFullMs, std::memory_order_relaxed);
        gnnInferenceStride.store(chooseGNNInferenceStride(conservativeFullMs, outputReservoirFill01),
                                 std::memory_order_relaxed);
        hasRuntimeGNNForwardTiming = false;
        validGNNBinMask.fill(0u);
        previousUseGNNForBin.fill(0u);
        lastInferenceFeatureValid.fill(0u);
        gnnGateMix.fill(0.0f);
        gnnBlendRemaining = 0;
        schedulerGeometryInitialised = false;
        lastActiveMicCount = activeMicCount;
    }

    for (int i = 0; i < numSamples; ++i)
    {
        for (int ch = 0; ch < activeMicCount * 4; ++ch)
        {
            const float sample = (ch < availableInputChannels) ? inputBuffer.getSample(ch, i) : 0.0f;
            inputRingBuffer.setSample(ch, inputWritePointer, sample);
        }

        ++samplesReceived;
        if (samplesReceived >= winLen && ((samplesReceived - winLen) % hop) == 0)
        {
            processSTFTFrame(listenerX,
                             listenerY,
                             listenerZ,
                             micX,
                             micY,
                             micZ,
                             micAzimuth,
                             diracSmoothing,
                             energyThresholdDb,
                             interpolationMode,
                             temporalWindow,
                             activeMicCount,
                             outputReservoirFill01,
                             outputReservoirAvailableMs);
        }

        const float norm = outputNormRingBuffer.getSample(0, outputReadPointer);
        const float invNorm = (norm > kEps) ? (1.0f / norm) : 0.0f;

        for (int ch = 0; ch < kNumOutputChannels; ++ch)
        {
            const float outSample = outputRingBuffer.getSample(ch, outputReadPointer) * invNorm;
            if (ch < availableOutputChannels)
                outputBuffer.setSample(ch, i, outSample);
            outputRingBuffer.setSample(ch, outputReadPointer, 0.0f);
        }

        outputNormRingBuffer.setSample(0, outputReadPointer, 0.0f);
        inputWritePointer = (inputWritePointer + 1) % ringSize;
        outputReadPointer = (outputReadPointer + 1) % ringSize;
    }
}

void DiracGNNEngine::processSTFTFrame(
    float listenerX,
    float listenerY,
    float listenerZ,
    const std::array<float, kNumMics>& micX,
    const std::array<float, kNumMics>& micY,
    const std::array<float, kNumMics>& micZ,
    const std::array<float, kNumMics>& micAzimuth,
    float diracSmoothing,
    float energyThresholdDb,
    int interpolationMode,
    int temporalWindow,
    int activeMicCount,
    float outputReservoirFill01,
    float outputReservoirAvailableMs)
{
    const int ringSize = inputRingBuffer.getNumSamples();
    const int readIdx = (inputWritePointer - (winLen - 1) + ringSize) % ringSize;

    // 1. STFT: Hann 512, hop 128, FFT 512, matching the paper.
    for (int ch = 0; ch < activeMicCount * 4; ++ch)
    {
        std::fill(fftWorkspace.begin(), fftWorkspace.end(), 0.0f);
        for (int i = 0; i < winLen; ++i)
        {
            const int idx = (readIdx + i) % ringSize;
            fftWorkspace[(size_t) i] = inputRingBuffer.getSample(ch, idx) * windowCoefficients[(size_t) i];
        }

        forwardFFT.performRealOnlyForwardTransform(fftWorkspace.data(), true);
        const size_t spectrumOffset = (size_t) ch * (size_t) numBins;
        for (int k = 0; k < numBins; ++k)
            spectra[spectrumOffset + (size_t) k] = { fftWorkspace[(size_t) (2 * k)],
                                                     fftWorkspace[(size_t) (2 * k + 1)] };
    }

    if (interpolationMode == 1)
        lastGNNStage.store(2, std::memory_order_relaxed); // STFT complete

    // If microphone yaw metadata changes while running, the recursively
    // smoothed W-X/W-Y covariance cross-terms still encode the previous
    // orientation. Re-express those historical intensity components by the
    // yaw delta before injecting the current frame. The XY power sum used by
    // energy is rotation-invariant, so covXX+covYY needs no correction.
    bool yawMetadataChanged = false;
    if (!micAzimuthStateInitialised)
    {
        lastMicAzimuth = micAzimuth;
        micAzimuthStateInitialised = true;
    }
    else
    {
        constexpr float twoPi = 6.28318530717958647692f;
        for (int m = 0; m < activeMicCount; ++m)
        {
            const float newYaw = micAzimuth[(size_t) m];
            const float delta = std::remainder(newYaw - lastMicAzimuth[(size_t) m], twoPi);
            if (std::abs(delta) <= 1.0e-6f)
                continue;

            const float c = std::cos(delta);
            const float sn = std::sin(delta);
            for (int k = 0; k < numBins; ++k)
            {
                const auto oldWX = covWX(k, m);
                const auto oldWY = covWY(k, m);
                covWX(k, m) = c * oldWX - sn * oldWY;
                covWY(k, m) = sn * oldWX + c * oldWY;
            }
            lastMicAzimuth[(size_t) m] = newYaw;
            yawMetadataChanged = true;
        }
    }

    if (yawMetadataChanged)
    {
        // Do not temporally average GNN weights or diffuse residuals that were
        // computed under the old microphone coordinate interpretation.
        resetWeightHistory();
        hasLastValidGNNWeights = false;
        lastValidGNNWeights.setZero();
        validGNNBinMask.fill(0u);
        gnnBlendRemaining = 0;
        gnnInferenceCountdown = 0;
        schedulerGeometryInitialised = false;
        std::fill(previousResidualValid.begin(), previousResidualValid.end(), 0u);
    }

    // Convert every known microphone from its local FOA orientation into the
    // common global Cartesian frame before DirAC analysis, GNN edge features,
    // anchor selection/synthesis, or Delaunay/GNN comparison.  The stored yaw
    // is in radians with the standard XY convention: yaw=0 means local +X is
    // global +X, positive yaw rotates local +X toward global +Y (CCW viewed
    // from +Z).  FOA channel order is [W, Y, Z, X], so only X/Y rotate.
    //
    //   Xg = cos(yaw) Xl - sin(yaw) Yl
    //   Yg = sin(yaw) Xl + cos(yaw) Yl
    //
    // Rotating in the TF domain is exactly equivalent to rotating the two
    // time-domain FOA components, while WOLA naturally smooths GUI changes
    // across overlapping frames.
    for (int m = 0; m < activeMicCount; ++m)
    {
        const float yaw = micAzimuth[(size_t) m];
        if (std::abs(yaw) <= 1.0e-12f)
            continue;

        const float c = std::cos(yaw);
        const float sn = std::sin(yaw);
        const int baseCh = m * 4;
        for (int k = 0; k < numBins; ++k)
        {
            const size_t yIndex = (size_t) (baseCh + 1) * (size_t) numBins + (size_t) k;
            const size_t xIndex = (size_t) (baseCh + 3) * (size_t) numBins + (size_t) k;
            const auto yLocal = spectra[yIndex];
            const auto xLocal = spectra[xIndex];
            spectra[xIndex] = c * xLocal - sn * yLocal;
            spectra[yIndex] = sn * xLocal + c * yLocal;
        }
    }

    // 2. DirAC analysis from the smoothed FOA covariance entries required by
    // Eq. (2): E = 1/2 Tr(C), i = Re([C14,C12,C13]^T), psi and u from i.
    const float keep = diracSmoothing;
    const float inject = 1.0f - keep;
    float framePeakEnergy = kEps;

    for (int m = 0; m < activeMicCount; ++m)
    {
        const int baseCh = m * 4; // [W, Y, Z, X]
        for (int k = 0; k < numBins; ++k)
        {
            const auto W = spectra[(size_t) (baseCh + 0) * (size_t) numBins + (size_t) k];
            const auto Y = spectra[(size_t) (baseCh + 1) * (size_t) numBins + (size_t) k];
            const auto Z = spectra[(size_t) (baseCh + 2) * (size_t) numBins + (size_t) k];
            const auto X = spectra[(size_t) (baseCh + 3) * (size_t) numBins + (size_t) k];

            covWW(k, m) = keep * covWW(k, m) + inject * std::norm(W);
            covYY(k, m) = keep * covYY(k, m) + inject * std::norm(Y);
            covZZ(k, m) = keep * covZZ(k, m) + inject * std::norm(Z);
            covXX(k, m) = keep * covXX(k, m) + inject * std::norm(X);
            covWX(k, m) = keep * covWX(k, m) + inject * (W * std::conj(X));
            covWY(k, m) = keep * covWY(k, m) + inject * (W * std::conj(Y));
            covWZ(k, m) = keep * covWZ(k, m) + inject * (W * std::conj(Z));

            const float E = std::max(0.5f * (covWW(k, m) + covYY(k, m) + covZZ(k, m) + covXX(k, m)), kEps);
            const float iX = std::real(covWX(k, m));
            const float iY = std::real(covWY(k, m));
            const float iZ = std::real(covWZ(k, m));
            const float normI = std::sqrt(iX * iX + iY * iY + iZ * iZ);

            E_obs(k, m) = E;
            psi_obs(k, m) = juce::jlimit(0.0f, 1.0f, 1.0f - normI / (E + kEps));
            u_obs_X(k, m) = iX / (normI + kEps);
            u_obs_Y(k, m) = iY / (normI + kEps);
            u_obs_Z(k, m) = iZ / (normI + kEps);
            framePeakEnergy = std::max(framePeakEnergy, E);
        }
    }

    for (int m = activeMicCount; m < kNumMics; ++m)
    {
        E_obs.col(m).setZero();
        psi_obs.col(m).setOnes();
        u_obs_X.col(m).setZero();
        u_obs_Y.col(m).setZero();
        u_obs_Z.col(m).setZero();
    }

    if (interpolationMode == 1)
        lastGNNStage.store(3, std::memory_order_relaxed); // DirAC analysis complete

    if (lastInterpolationMode != interpolationMode)
    {
        resetWeightHistory();
        lastInterpolationMode = interpolationMode;
    }

    // 3. Interpolation. Delaunay weights are always available because in GNN
    // mode they are intentionally used for LOW-ENERGY TF bins. High-energy
    // bins use learned weights, possibly held/interpolated from the most recent
    // successful learned inference. A scheduling hold is never a Delaunay
    // failure fallback.
    computeDelaunayWeights(listenerX, listenerY, micX, micY, weightsDelaunay, activeMicCount);
    std::fill(useGNNForBin.begin(), useGNNForBin.end(), 0u);

    bool inferredThisFrame = false;
    if (interpolationMode == 0)
    {
        weightsRaw = weightsDelaunay;
        weightsPreConstraint = weightsDelaunay;
        weightsGNNConstrained.setZero();
        weightsUsed = weightsDelaunay;
        gnnGateMix.fill(0.0f);
        previousUseGNNForBin.fill(0u);
        gnnInferenceCountdown = 0;
        gnnBlendRemaining = 0;
    }
    else
    {
        // Classify the hybrid TF gate with 3 dB hysteresis. A bin must reach
        // the configured threshold to ENTER the learned region, but once learned
        // it may fall 3 dB below that threshold before EXIT. This prevents
        // frame-to-frame GNN/Delaunay flapping on marginal bins.
        int reliableBinCount = 0;
        int coverageBinCount = 0;
        const float thresholdLinear = std::pow(10.0f, 0.1f * energyThresholdDb);
        const float thresholdEnergy = (framePeakEnergy + kEps) * thresholdLinear;
        for (int k = 0; k < numBins; ++k)
        {
            float binPeakEnergy = kEps;
            for (int m = 0; m < activeMicCount; ++m)
                binPeakEnergy = std::max(binPeakEnergy, E_obs(k, m));

            const bool wasGNN = previousUseGNNForBin[(size_t) k] != 0u;
            const bool useGNN = gateWantsGNN(wasGNN, binPeakEnergy, thresholdEnergy);
            useGNNForBin[(size_t) k] = useGNN ? 1u : 0u;

            if (useGNN && !wasGNN)
            {
                // Never revive a learned row from an older active island. The
                // gate stays on the Delaunay side until this row has been
                // recomputed, then the bidirectional 16-ms gate mix fades in.
                validGNNBinMask[(size_t) k] = 0u;
                lastInferenceFeatureValid[(size_t) k] = 0u;
                invalidateTemporalHistoryForBin(k);
                gnnGateEntries.fetch_add(1, std::memory_order_relaxed);
            }
            else if (!useGNN && wasGNN)
            {
                // Keep the last learned row valid while gnnGateMix fades OUT.
                // It is invalidated if/when the bin later enters again.
                gnnGateExits.fetch_add(1, std::memory_order_relaxed);
            }

            if (useGNN)
            {
                reliableBinIndices[(size_t) reliableBinCount++] = k;
                if (validGNNBinMask[(size_t) k] == 0u)
                    coverageBinIndices[(size_t) coverageBinCount++] = k;
            }
            previousUseGNNForBin[(size_t) k] = useGNN ? 1u : 0u;
        }
        lastGNNActiveBinCount.store(reliableBinCount, std::memory_order_relaxed);
        lastGNNCoverageBinCount.store(coverageBinCount, std::memory_order_relaxed);

        // Single-node interpolation remains exactly weight 1, which is inside
        // the allowed learned gain-sum interval [0.5, 2.0]. Do not spend neural CPU on a
        // graph whose learned result is analytically known.
        if (activeMicCount == 1)
        {
            weightsGNNConstrained.setZero();
            weightsGNNConstrained.col(0).setOnes();
            lastValidGNNWeights = weightsGNNConstrained;
            validGNNBinMask.fill(1u);
            hasLastValidGNNWeights = true;
            gnnInferenceStride.store(16, std::memory_order_relaxed);
            gnnInferenceCountdown = 15;
            lastGNNForwardMs.store(0.0, std::memory_order_relaxed);
            lastGNNStage.store(63, std::memory_order_relaxed); // exact one-node learned result
        }
        else if (reliableBinCount == 0)
        {
            if (hasLastValidGNNWeights)
                weightsGNNConstrained = lastValidGNNWeights;
            else
                weightsGNNConstrained.setZero();
            lastGNNStage.store(62, std::memory_order_relaxed); // all bins intentionally Delaunay
        }
        else
        {
            // Cadence is based on a prediction for the CURRENT active batch,
            // not on the duration of whichever (possibly tiny) batch happened
            // to run last. This prevents a 1-bin coverage pass from convincing
            // the scheduler that a 200-bin pass is equally cheap.
            const double predictedFullBatchMs = predictGNNForwardMs(reliableBinCount, reliableBinCount, false, activeMicCount);
            const int desiredStride = chooseGNNInferenceStride(predictedFullBatchMs, outputReservoirFill01);
            gnnInferenceStride.store(desiredStride, std::memory_order_relaxed);

            // Geometry changes refresh sooner than a static scene, but
            // continuous dragging must not defeat the CPU controller on the
            // 2010 Xeon. At most double the learned update rate: stride 4 -> 2,
            // 8 -> 4, 16 -> 8, etc. The preload reservoir remains authoritative.
            const bool geometryChanged = geometryRequiresImmediateGNN(listenerX, listenerY, listenerZ,
                                                                       micX, micY, micZ, micAzimuth,
                                                                       activeMicCount);
            if (geometryChanged && hasLastValidGNNWeights)
            {
                const int geometryStride = std::max(1, desiredStride / 2);
                gnnInferenceCountdown = std::min(gnnInferenceCountdown, geometryStride - 1);
            }

            // The learned model consumes acoustic descriptors, not only geometry.
            // Scheduling uses CPU/reservoir state together with acoustic novelty,
            // allowing E/psi/u to change substantially while weights were held.
            // This O(N*bins) detector is tiny relative to one GNN forward and
            // accelerates the cadence in AUDIO time when the actual learned
            // inputs move. It never bypasses the reservoir admission guard.
            float acousticNovelty = 0.0f;
            const bool acousticChanged = hasLastValidGNNWeights
                                      && acousticFeaturesChanged(reliableBinIndices.data(),
                                                                 reliableBinCount,
                                                                 activeMicCount,
                                                                 acousticNovelty);
            lastAcousticNovelty.store(acousticNovelty, std::memory_order_relaxed);
            if (acousticChanged)
            {
                int acousticStride = std::max(1, desiredStride / 2);
                if (acousticNovelty >= 0.18f && outputReservoirFill01 >= 0.80f)
                    acousticStride = std::max(1, desiredStride / 4);
                gnnInferenceCountdown = std::min(gnnInferenceCountdown, acousticStride - 1);
            }

            // Coverage and cadence are deliberately separate. A newly entered
            // bin must receive a fresh learned row, but it must NOT force a full
            // forward over every already-covered bin. Full-batch inference is
            // reserved for normal cadence / geometry / acoustic updates.
            const bool fullInferenceDue = !hasLastValidGNNWeights || gnnInferenceCountdown <= 0;
            const bool coverageDue = coverageBinCount > 0;
            const bool coverageOnly = shouldUseCoverageOnly(hasLastValidGNNWeights,
                                                            gnnInferenceCountdown,
                                                            coverageBinCount);
            const int* inferenceBinIndices = coverageOnly ? coverageBinIndices.data()
                                                           : reliableBinIndices.data();
            const int inferenceBinCount = coverageOnly ? coverageBinCount : reliableBinCount;
            const bool inferenceDue = fullInferenceDue || coverageDue;

            // Continuity-first admission uses the batch that is ACTUALLY about
            // to run. The per-bin throughput estimate is seeded by the startup
            // full benchmark and refined only by meaningful runtime batches.
            const double predictedForwardMs = predictGNNForwardMs(inferenceBinCount, reliableBinCount, true, activeMicCount);
            lastPredictedGNNForwardMs.store(predictedForwardMs, std::memory_order_relaxed);
            const float requiredReserveMs = (float) juce::jlimit(56.0, 496.0,
                1.60 * predictedForwardMs + 48.0);
            const bool reservoirUnsafe = outputReservoirFill01 < 0.34f
                                      || outputReservoirAvailableMs < requiredReserveMs;
            const bool pressureDeferred = inferenceDue && hasLastValidGNNWeights && reservoirUnsafe;
            const bool mustInfer = inferenceDue && !pressureDeferred;

            bool forwardValid = true;
            bool haveCurrentGNN = hasLastValidGNNWeights;
            if (mustInfer)
            {
                inferredThisFrame = true;
                lastGNNStage.store(9, std::memory_order_relaxed);
                const double gnnT0 = juce::Time::getMillisecondCounterHiRes();
                forwardValid = forwardGNN(listenerX, listenerY, listenerZ,
                                          micX, micY, micZ, weightsRaw, activeMicCount,
                                          inferenceBinIndices, inferenceBinCount);
                const double forwardMs = juce::Time::getMillisecondCounterHiRes() - gnnT0;
                lastGNNForwardMs.store(forwardMs, std::memory_order_relaxed);
                lastGNNInferredBinCount.store(inferenceBinCount, std::memory_order_relaxed);
                const double oldEwma = gnnForwardEwmaMs.load(std::memory_order_relaxed);
                const double newEwma = !hasRuntimeGNNForwardTiming || oldEwma <= 0.0
                                     ? forwardMs
                                     : (0.80 * oldEwma + 0.20 * forwardMs);
                hasRuntimeGNNForwardTiming = true;
                gnnForwardEwmaMs.store(newEwma, std::memory_order_relaxed);

                // Refine throughput only from batches large enough that fixed
                // call overhead does not dominate. Tiny coverage passes still
                // benefit from the startup/full-batch estimate instead of
                // poisoning future admission predictions.
                if (inferenceBinCount >= 8 || gnnForwardPerBinEwmaMs.load(std::memory_order_relaxed) <= 0.0)
                {
                    const double observedPerBin = forwardMs / (double) std::max(4, inferenceBinCount);
                    const double oldPerBin = gnnForwardPerBinEwmaMs.load(std::memory_order_relaxed);
                    const double newPerBin = oldPerBin > 0.0
                                           ? 0.80 * oldPerBin + 0.20 * observedPerBin
                                           : observedPerBin;
                    gnnForwardPerBinEwmaMs.store(newPerBin, std::memory_order_relaxed);
                    gnnTimingReferenceMicCount.store(activeMicCount, std::memory_order_relaxed);
                    const double oldPeak = gnnForwardPerBinPeakMs.load(std::memory_order_relaxed);
                    const double decayedPeak = 0.98 * oldPeak;
                    gnnForwardPerBinPeakMs.store(std::max(observedPerBin, decayedPeak),
                                                 std::memory_order_relaxed);
                }
                gnnInferenceCount.fetch_add(1, std::memory_order_relaxed);
                if (coverageOnly)
                    gnnCoverageOnlyInferences.fetch_add(1, std::memory_order_relaxed);
                else
                    gnnFullInferences.fetch_add(1, std::memory_order_relaxed);
                if (geometryChanged)
                    gnnGeometryForcedInferences.fetch_add(1, std::memory_order_relaxed);

                if (forwardValid)
                {
                    // Temporal averaging is maintained only for bins actually
                    // inferred on this pass. Below-threshold bins never inject
                    // zeros into the learned history.
                    applyTemporalWeightWindowForBins(weightsRaw,
                                                     inferenceBinIndices, inferenceBinCount,
                                                     temporalWindow, weightsPreConstraint);
                    weightsCandidate = lastValidGNNWeights;
                    for (int r = 0; r < inferenceBinCount; ++r)
                    {
                        const int k = inferenceBinIndices[(size_t) r];
                        weightsCandidate.row(k) = weightsPreConstraint.row(k);
                    }
                    haveCurrentGNN = constrainGNNWeightSumsForBins(weightsCandidate, activeMicCount,
                                                                    inferenceBinIndices, inferenceBinCount);
                    if (haveCurrentGNN)
                    {
                        const bool hadPrevious = hasLastValidGNNWeights;
                        if (hadPrevious)
                        {
                            gnnBlendFrom = weightsGNNConstrained;
                            // Existing learned rows de-click from their previous
                            // GNN value. Newly covered bins have no learned-side
                            // predecessor, so start that side directly at the new
                            // candidate; the dedicated gnnGateMix performs the
                            // Delaunay<->GNN parameter crossfade with correct
                            // renderer semantics.
                            for (int r = 0; r < inferenceBinCount; ++r)
                            {
                                const int k = inferenceBinIndices[(size_t) r];
                                if (validGNNBinMask[(size_t) k] == 0u)
                                    gnnBlendFrom.row(k) = weightsCandidate.row(k);
                            }
                            gnnBlendTo = weightsCandidate;
                            // Keep a short fixed de-click blend. Tying blend length
                            // to inference stride
                            // added 32..256 ms of extra tracking lag and made the
                            // learned field chase stale acoustic features.
                            gnnBlendTotalFrames = kGNNBlendFrames;
                            gnnBlendRemaining = gnnBlendTotalFrames;
                        }

                        for (int r = 0; r < inferenceBinCount; ++r)
                        {
                            const int k = inferenceBinIndices[(size_t) r];
                            lastValidGNNWeights.row(k) = weightsCandidate.row(k);
                            validGNNBinMask[(size_t) k] = 1u;
                        }
                        captureAcousticFeatureSnapshot(inferenceBinIndices,
                                                       inferenceBinCount, activeMicCount);
                        if (acousticChanged)
                            gnnAcousticForcedInferences.fetch_add(1, std::memory_order_relaxed);
                        hasLastValidGNNWeights = true;
                        if (!hadPrevious)
                        {
                            weightsGNNConstrained = lastValidGNNWeights;
                            gnnBlendRemaining = 0;
                        }
                    }
                }

                if (!forwardValid || !haveCurrentGNN)
                {
                    invalidGNNFrames.fetch_add(1, std::memory_order_relaxed);
                    if (hasLastValidGNNWeights)
                    {
                        weightsGNNConstrained = lastValidGNNWeights;
                        reusedGNNFrames.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                        weightsGNNConstrained.setZero();
                }

                const double postPredictedFullMs = predictGNNForwardMs(reliableBinCount, reliableBinCount, false, activeMicCount);
                const int postStride = chooseGNNInferenceStride(postPredictedFullMs, outputReservoirFill01);
                gnnInferenceStride.store(postStride, std::memory_order_relaxed);
                // A tiny coverage repair must not postpone the next scheduled
                // full learned update. It consumes this audio frame but leaves
                // the established cadence otherwise intact.
                gnnInferenceCountdown = coverageOnly
                                      ? std::max(0, gnnInferenceCountdown - 1)
                                      : postStride - 1;
            }
            else
            {
                if (hasLastValidGNNWeights)
                {
                    if (gnnBlendRemaining > 0)
                        updateGNNBlend(weightsGNNConstrained);
                    else
                        weightsGNNConstrained = lastValidGNNWeights;
                    gnnCadenceHeldFrames.fetch_add(1, std::memory_order_relaxed);
                }
                else
                    weightsGNNConstrained.setZero();

                if (pressureDeferred)
                {
                    gnnPressureDeferredFrames.fetch_add(1, std::memory_order_relaxed);
                    lastGNNStage.store(64, std::memory_order_relaxed); // reservoir-protected hold
                }
                else
                    lastGNNStage.store(61, std::memory_order_relaxed); // normal multirate hold
                gnnInferenceCountdown = std::max(0, gnnInferenceCountdown - 1);
            }

            if (inferredThisFrame && gnnBlendRemaining > 0)
                updateGNNBlend(weightsGNNConstrained);
        }

        for (int k = 0; k < numBins; ++k)
        {
            const bool learnedReady = validGNNBinMask[(size_t) k] != 0u;
            const bool targetGNN = useGNNForBin[(size_t) k] != 0u && learnedReady;
            gnnGateMix[(size_t) k] = advanceGateMix(gnnGateMix[(size_t) k], targetGNN);
            const float mix = gnnGateMix[(size_t) k];
            if (learnedReady && mix > 0.0f)
                weightsUsed.row(k) = (1.0f - mix) * weightsDelaunay.row(k)
                                   + mix * weightsGNNConstrained.row(k);
            else
                weightsUsed.row(k) = weightsDelaunay.row(k);
        }
    }

    // Publish a lightweight live weight display while decoupling its cost from the
    // learned cadence. Even if the GNN runs at 62.5 Hz, the display is published
    // at ~10 Hz (plus the very first frame) so extra atomics/averaging cannot
    // steal a material share of the old Xeon's compute budget.
    if (!weightVizHasPublished || inferredThisFrame || weightVizCadenceCountdown <= 0)
    {
        publishWeightVisualization(weightsPreConstraint,
                                   weightsGNNConstrained,
                                   weightsDelaunay,
                                   weightsUsed,
                                   interpolationMode == 1);
        weightVizHasPublished = true;
        weightVizCadenceCountdown = 11; // 125 / 12 ~= 10.4 Hz
    }
    else
        --weightVizCadenceCountdown;
    selectAnchor(listenerX, listenerY, listenerZ, micX, micY, micZ, activeMicCount, interpolationMode == 0);

    constexpr float sqrt3 = 1.7320508075688772f;

    for (int k = 0; k < numBins; ++k)
    {
        float eHat = 0.0f;
        float psiHat = 1.0f;
        float ux = 0.0f;
        float uy = 0.0f;
        float uz = 0.0f;

        const bool learnedRowAvailable = interpolationMode == 1
                                      && validGNNBinMask[(size_t) k] != 0u;
        const float learnedMix = learnedRowAvailable ? gnnGateMix[(size_t) k] : 0.0f;
        const auto delaunayParams = interpolatePaperDelaunayParams(k, weightsDelaunay, activeMicCount);

        if (learnedMix <= 0.0f)
        {
            eHat = delaunayParams.energy;
            psiHat = delaunayParams.diffuseness;
            ux = delaunayParams.ux; uy = delaunayParams.uy; uz = delaunayParams.uz;
        }
        else
        {
            // Keep the established learned operator at mix=1, but transition
            // between it and the paper Delaunay operator in normalized-intensity
            // space. This makes BOTH gate directions continuous even though the
            // two interpolation methods intentionally use different semantics.
            const auto learnedParams = interpolateLearnedParams(k, activeMicCount);
            const auto p = blendSpatialParams(delaunayParams, learnedParams, learnedMix);
            eHat = p.energy;
            psiHat = p.diffuseness;
            ux = p.ux; uy = p.uy; uz = p.uz;
        }

        if (!std::isfinite(eHat) || eHat <= kEps)
        {
            for (int c = 0; c < kNumOutputChannels; ++c)
                querySpectra[(size_t) c * (size_t) numBins + (size_t) k] = {};
            continue;
        }

        const float eDir = std::max(0.0f, eHat * (1.0f - psiHat));
        const float eDiff = std::max(0.0f, eHat * psiHat);
        const float shTarget[4] = { 1.0f, sqrt3 * uy, sqrt3 * uz, sqrt3 * ux };
        float shTargetNorm2 = 0.0f;
        for (float value : shTarget)
            shTargetNorm2 += value * value;

        std::complex<float> anchor[4];
        for (int c = 0; c < 4; ++c)
            anchor[c] = getTransitionAnchorSpectrum(c, k);

        const float magAnchorW = std::abs(anchor[0]);
        const std::complex<float> phaseAnchorW = (magAnchorW > kEps)
                                                ? anchor[0] / magAnchorW
                                                : std::complex<float>(1.0f, 0.0f);

        // Decompose the anchor into a local direct estimate plus its measured
        // residual. The residual carries coherent measured phase through time;
        // it replaces the old frame-wise random diffuse synthesis.
        const int a = juce::jlimit(0, activeMicCount - 1, currentAnchor);
        float auX = u_obs_X(k, a);
        float auY = u_obs_Y(k, a);
        float auZ = u_obs_Z(k, a);
        float aPsi = psi_obs(k, a);
        float aE = E_obs(k, a);

        if (previousAnchor >= 0 && anchorCrossfadeRemaining > 0)
        {
            const int transitionFrames = std::max(1, anchorCrossfadeTotalFrames);
    const float alpha = 1.0f - (float) anchorCrossfadeRemaining / (float) transitionFrames;
            const int p = previousAnchor;
            auX = (1.0f - alpha) * u_obs_X(k, p) + alpha * auX;
            auY = (1.0f - alpha) * u_obs_Y(k, p) + alpha * auY;
            auZ = (1.0f - alpha) * u_obs_Z(k, p) + alpha * auZ;
            const float an = std::sqrt(auX * auX + auY * auY + auZ * auZ);
            auX /= (an + kEps); auY /= (an + kEps); auZ /= (an + kEps);
            aPsi = (1.0f - alpha) * psi_obs(k, p) + alpha * aPsi;
            aE = (1.0f - alpha) * E_obs(k, p) + alpha * aE;
        }

        const float shAnchor[4] = { 1.0f, sqrt3 * auY, sqrt3 * auZ, sqrt3 * auX };
        float shAnchorNorm2 = 0.0f;
        for (float value : shAnchor)
            shAnchorNorm2 += value * value;
        const float anchorDirectEnergy = std::max(0.0f, aE * (1.0f - aPsi));
        const float anchorDirectGain = std::sqrt(2.0f * anchorDirectEnergy
                                               / std::max(shAnchorNorm2, kEps));
        std::complex<float> residual[4];
        float residualEnergy = 0.0f;
        for (int c = 0; c < 4; ++c)
        {
            residual[c] = anchor[c] - shAnchor[c] * anchorDirectGain * phaseAnchorW;
            residualEnergy += 0.5f * std::norm(residual[c]);
        }

        if (residualEnergy > 1.0e-10f)
        {
            for (int c = 0; c < 4; ++c)
                previousResidualSpectrum[(size_t) c * (size_t) numBins + (size_t) k] = residual[c];
            previousResidualValid[(size_t) k] = 1u;
        }
        else if (previousResidualValid[(size_t) k] != 0u)
        {
            residualEnergy = 0.0f;
            for (int c = 0; c < 4; ++c)
            {
                residual[c] = previousResidualSpectrum[(size_t) c * (size_t) numBins + (size_t) k];
                residualEnergy += 0.5f * std::norm(residual[c]);
            }
        }
        else
        {
            // Deterministic, frame-coherent startup basis. This is used only
            // before a measured residual exists and never randomises per frame.
            residual[0] = phaseAnchorW;
            residual[1] = phaseAnchorW * std::complex<float>(0.0f, 1.0f);
            residual[2] = -phaseAnchorW;
            residual[3] = phaseAnchorW * std::complex<float>(0.0f, -1.0f);
            residualEnergy = 2.0f;
        }

        // Remove any component of the measured residual that is collinear with
        // the target direct steering vector. This prevents coherent cross-terms
        // between the direct and residual branches from modulating the target
        // energy while preserving the measured residual phase structure.
        auto orthogonalizeResidualToTarget = [&]() noexcept
        {
            std::complex<float> projectionNumerator {};
            float projectionDenominator = 0.0f;
            for (int c = 0; c < 4; ++c)
            {
                projectionNumerator += shTarget[c] * residual[c];
                projectionDenominator += shTarget[c] * shTarget[c];
            }

            const auto projection = projectionNumerator / std::max(projectionDenominator, kEps);
            float energy = 0.0f;
            for (int c = 0; c < 4; ++c)
            {
                residual[c] -= shTarget[c] * projection;
                energy += 0.5f * std::norm(residual[c]);
            }
            return energy;
        };

        residualEnergy = orthogonalizeResidualToTarget();
        if (residualEnergy <= 1.0e-10f && eDiff > kEps)
        {
            // Degenerate case: the available residual happened to align with
            // the target direct steering vector. Use a deterministic coherent
            // basis and project it into the target-orthogonal subspace. This is
            // a startup/degeneracy guard only; measured residuals remain the
            // normal synthesis path.
            residual[0] = phaseAnchorW;
            residual[1] = phaseAnchorW * std::complex<float>(0.0f, 1.0f);
            residual[2] = -phaseAnchorW;
            residual[3] = phaseAnchorW * std::complex<float>(0.0f, -1.0f);
            residualEnergy = orthogonalizeResidualToTarget();
        }

        const float targetDirectGain = std::sqrt(2.0f * eDir / std::max(shTargetNorm2, kEps));
        const float residualGain = (eDiff > kEps && residualEnergy > kEps)
                                   ? std::sqrt(eDiff / residualEnergy)
                                   : 0.0f;
        for (int c = 0; c < 4; ++c)
        {
            const auto directPart = shTarget[c] * targetDirectGain * phaseAnchorW;
            const auto rendered = directPart + residualGain * residual[c];
            querySpectra[(size_t) c * (size_t) numBins + (size_t) k] = rendered;
        }
    }

    if (anchorCrossfadeRemaining > 0)
    {
        --anchorCrossfadeRemaining;
        if (anchorCrossfadeRemaining == 0)
            previousAnchor = -1;
    }

    // DC and Nyquist are real for a real inverse FFT.
    for (int c = 0; c < kNumOutputChannels; ++c)
    {
        auto& dc = querySpectra[(size_t) c * (size_t) numBins];
        auto& ny = querySpectra[(size_t) c * (size_t) numBins + (size_t) (numBins - 1)];
        dc = { dc.real(), 0.0f };
        ny = { ny.real(), 0.0f };
    }

    // 4. Causal WOLA, normalized by the sum of squared Hann windows.
    const int writeIdx = outputReadPointer;
    for (int i = 0; i < winLen; ++i)
    {
        const int idx = (writeIdx + i) % outputRingBuffer.getNumSamples();
        const float w = windowCoefficients[(size_t) i];
        outputNormRingBuffer.addSample(0, idx, w * w);
    }

    for (int c = 0; c < kNumOutputChannels; ++c)
    {
        std::fill(fftWorkspace.begin(), fftWorkspace.end(), 0.0f);
        const size_t spectrumOffset = (size_t) c * (size_t) numBins;
        for (int k = 0; k < numBins; ++k)
        {
            const auto value = querySpectra[spectrumOffset + (size_t) k];
            fftWorkspace[(size_t) (2 * k)] = value.real();
            fftWorkspace[(size_t) (2 * k + 1)] = value.imag();
        }

        inverseFFT.performRealOnlyInverseTransform(fftWorkspace.data());
        for (int i = 0; i < winLen; ++i)
        {
            const int idx = (writeIdx + i) % outputRingBuffer.getNumSamples();
            outputRingBuffer.addSample(c, idx, fftWorkspace[(size_t) i] * windowCoefficients[(size_t) i]);
        }
    }
}

void DiracGNNEngine::mlpApplyPrepared(const RowMatrixXf& inputStorage,
                                      int inputColumns,
                                      const MLP& layer,
                                      bool useSoftplus,
                                      RowMatrixXf& outputStorage) noexcept
{
    jassert(inputColumns == layer.W1T.rows());
    jassert(inputStorage.rows() == numBins);
    const int hiddenCols = (int) layer.W1T.cols();
    const int outputCols = (int) layer.W2T.cols();

    auto hidden = mlpHiddenScratch.leftCols(hiddenCols);
    multiplyPreparedWeights(inputStorage, numBins, inputColumns, layer.W1T, mlpHiddenScratch, gemmPackScratch.data(), gemmPackScratch.size());
    for (int r = 0; r < hidden.rows(); ++r)
        for (int c = 0; c < hiddenCols; ++c)
        {
            const float v = hidden(r, c) + layer.b1(c);
            hidden(r, c) = (v >= 0.0f) ? v : (0.1f * v);
        }

    auto output = outputStorage.leftCols(outputCols);
    multiplyPreparedWeights(mlpHiddenScratch, numBins, hiddenCols, layer.W2T, outputStorage, gemmPackScratch.data(), gemmPackScratch.size());
    for (int r = 0; r < output.rows(); ++r)
        for (int c = 0; c < outputCols; ++c)
        {
            float v = output(r, c) + layer.b2(c);
            if (useSoftplus)
                v = std::log1p(std::exp(-std::abs(v))) + std::max(v, 0.0f);
            output(r, c) = v;
        }
}


void DiracGNNEngine::mlpApplyBatch(const RowMatrixXf& inputStorage,
                                   int rows,
                                   int inputColumns,
                                   const MLP& layer,
                                   bool useSoftplus,
                                   RowMatrixXf& hiddenStorage,
                                   RowMatrixXf& outputStorage) noexcept
{
    jassert(inputColumns == layer.W1T.rows());
    jassert(rows <= inputStorage.rows());
    const int hiddenCols = (int) layer.W1T.cols();
    const int outputCols = (int) layer.W2T.cols();
    jassert(rows <= hiddenStorage.rows() && hiddenCols <= hiddenStorage.cols());
    jassert(rows <= outputStorage.rows() && outputCols <= outputStorage.cols());

    auto hidden = hiddenStorage.topRows(rows).leftCols(hiddenCols);
    multiplyPreparedWeights(inputStorage, rows, inputColumns, layer.W1T, hiddenStorage, gemmPackScratch.data(), gemmPackScratch.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < hiddenCols; ++c)
        {
            const float v = hidden(r, c) + layer.b1(c);
            hidden(r, c) = (v >= 0.0f) ? v : (0.1f * v);
        }

    auto output = outputStorage.topRows(rows).leftCols(outputCols);
    multiplyPreparedWeights(hiddenStorage, rows, hiddenCols, layer.W2T, outputStorage, gemmPackScratch.data(), gemmPackScratch.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < outputCols; ++c)
        {
            float v = output(r, c) + layer.b2(c);
            if (useSoftplus)
                v = std::log1p(std::exp(-std::abs(v))) + std::max(v, 0.0f);
            output(r, c) = v;
        }
}

double DiracGNNEngine::benchmarkGNNForwardMs(
    float qPosX, float qPosY, float qPosZ,
    const std::array<float, kNumMics>& micX,
    const std::array<float, kNumMics>& micY,
    const std::array<float, kNumMics>& micZ,
    int activeMicCount) noexcept
{
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    if (activeMicCount <= 1)
        return 0.0;

    std::array<int, 257> bins {};
    for (int k = 0; k < numBins; ++k)
        bins[(size_t) k] = k;

    // One warm-up removes first-use cache/lazy-binding effects. Then keep the
    // slower of two full forwards; using the faster pass could
    // under-size the continuity reservoir on the 2010 Xeon and make an ordinary
    // runtime fluctuation audible as an output-FIFO starvation.
    (void) forwardGNN(qPosX, qPosY, qPosZ, micX, micY, micZ,
                      weightsRaw, activeMicCount, bins.data(), numBins);

    double conservativeMs = 0.0;
    for (int pass = 0; pass < 2; ++pass)
    {
        const double t0 = juce::Time::getMillisecondCounterHiRes();
        const bool ok = forwardGNN(qPosX, qPosY, qPosZ, micX, micY, micZ,
                                   weightsRaw, activeMicCount, bins.data(), numBins);
        const double elapsed = juce::Time::getMillisecondCounterHiRes() - t0;
        if (ok && std::isfinite(elapsed))
            conservativeMs = std::max(conservativeMs, elapsed);
    }
    return std::isfinite(conservativeMs) ? std::max(0.0, conservativeMs) : 0.0;
}

bool DiracGNNEngine::gateWantsGNN(bool wasGNN,
                                   float binPeakEnergy,
                                   float enterThresholdEnergy) noexcept
{
    const float threshold = wasGNN
                          ? enterThresholdEnergy * kGNNGateExitEnergyRatio
                          : enterThresholdEnergy;
    return (binPeakEnergy + kEps) >= threshold;
}

float DiracGNNEngine::advanceGateMix(float currentMix, bool targetGNN) noexcept
{
    currentMix = juce::jlimit(0.0f, 1.0f, currentMix);
    const float step = 1.0f / (float) kGNNGateBlendFrames;
    return targetGNN ? std::min(1.0f, currentMix + step)
                     : std::max(0.0f, currentMix - step);
}

bool DiracGNNEngine::shouldUseCoverageOnly(bool hasLastValidWeights,
                                               int inferenceCountdown,
                                               int coverageBinCount) noexcept
{
    return hasLastValidWeights && coverageBinCount > 0 && inferenceCountdown > 0;
}

double DiracGNNEngine::predictGNNForwardMs(int requestedBinCount, int activeBinCount, bool conservative, int activeMicCount) const noexcept
{
    activeBinCount = juce::jlimit(1, numBins, activeBinCount);
    requestedBinCount = juce::jlimit(0, activeBinCount, requestedBinCount);
    if (requestedBinCount <= 0)
        return 0.0;

    double perBin = gnnForwardPerBinEwmaMs.load(std::memory_order_relaxed);
    if (conservative)
        perBin = std::max(perBin, gnnForwardPerBinPeakMs.load(std::memory_order_relaxed));
    if (!(perBin > 0.0) || !std::isfinite(perBin))
    {
        const double last = gnnForwardEwmaMs.load(std::memory_order_relaxed);
        const int lastBins = std::max(1, lastGNNInferredBinCount.load(std::memory_order_relaxed));
        if (last > 0.0 && std::isfinite(last))
            perBin = last / (double) std::max(4, lastBins);
    }
    if (!(perBin > 0.0) || !std::isfinite(perBin))
        return 0.0;

    // Scale the observed per-bin cost when N changes. The two message-passing
    // layers are dominated by complete-graph directed edges N*(N+1). This is
    // deliberately conservative for N>5, where the checkpoint is OOD and the
    // old Mac must never discover the ~quadratic cost via an audible underrun.
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    const int referenceMics = juce::jlimit(1, kNumMics,
        gnnTimingReferenceMicCount.load(std::memory_order_relaxed));
    const double currentWork = (double) activeMicCount * (double) (activeMicCount + 1);
    const double referenceWork = (double) referenceMics * (double) (referenceMics + 1);
    const double graphRatio = currentWork / std::max(1.0, referenceWork);

    // Four-bin floor covers fixed call/packing overhead for tiny coverage
    // repairs. The 25% margin plus 1 ms fixed reserve is intentionally
    // conservative because an underestimate is audible while an overestimate
    // merely defers learned work and keeps the current field.
    const int effectiveBins = std::max(4, requestedBinCount);
    const double predicted = 1.25 * perBin * graphRatio * (double) effectiveBins + 1.0;
    return std::isfinite(predicted) ? std::max(0.0, predicted) : 0.0;
}

int DiracGNNEngine::chooseGNNInferenceStride(double forwardEwmaMs, float reservoirFill01) const noexcept
{
    // Deliberately tiny controller: a handful of comparisons once per STFT
    // frame. No spectral novelty scan, allocations or locks. The neural
    // forward itself is orders of magnitude more expensive on the target Xeon.
    int stride = 4; // conservative 31.25 Hz startup on unknown/old hardware
    if (forwardEwmaMs > 0.0)
    {
        // Target an average learned-compute share around 2.5 ms per 8-ms hop.
        // Power-of-two cadences avoid jitter and make diagnostics intuitive.
        if      (forwardEwmaMs <= 2.5)  stride = 1;  // 125 Hz when genuinely cheap
        else if (forwardEwmaMs <= 5.0)  stride = 2;  // 62.5 Hz
        else if (forwardEwmaMs <= 10.0) stride = 4;  // 31.25 Hz
        else if (forwardEwmaMs <= 20.0) stride = 8;  // 15.625 Hz
        else if (forwardEwmaMs <= 40.0) stride = 16; // 7.8125 Hz
        else if (forwardEwmaMs <= 80.0) stride = 32; // 3.90625 Hz
        else                            stride = 64; // 1.953125 Hz emergency survival cadence
    }

    // Reservoir pressure can only slow inference, never accelerate it.
    if      (reservoirFill01 < 0.20f) stride = std::max(stride, 64);
    else if (reservoirFill01 < 0.35f) stride = std::max(stride, 32);
    else if (reservoirFill01 < 0.50f) stride = std::max(stride, 16);
    else if (reservoirFill01 < 0.65f) stride = std::max(stride, 8);
    else if (reservoirFill01 < 0.75f) stride = std::max(stride, 4);
    return stride;
}

bool DiracGNNEngine::geometryRequiresImmediateGNN(
    float listenerX, float listenerY, float listenerZ,
    const std::array<float, kNumMics>& micX,
    const std::array<float, kNumMics>& micY,
    const std::array<float, kNumMics>& micZ,
    const std::array<float, kNumMics>& micAzimuth,
    int activeMicCount) noexcept
{
    constexpr float posEps = 1.0e-4f;
    constexpr float yawEps = 1.0e-5f;
    bool changed = !schedulerGeometryInitialised
                || std::abs(listenerX - lastSchedulerListenerX) > posEps
                || std::abs(listenerY - lastSchedulerListenerY) > posEps
                || std::abs(listenerZ - lastSchedulerListenerZ) > posEps;
    for (int m = 0; m < activeMicCount && !changed; ++m)
        changed = std::abs(micX[(size_t) m] - lastSchedulerMicX[(size_t) m]) > posEps
               || std::abs(micY[(size_t) m] - lastSchedulerMicY[(size_t) m]) > posEps
               || std::abs(micZ[(size_t) m] - lastSchedulerMicZ[(size_t) m]) > posEps
               || std::abs(micAzimuth[(size_t) m] - lastSchedulerMicAzimuth[(size_t) m]) > yawEps;

    // Update the cheap observation baseline every STFT frame. Continuous GUI
    // dragging can therefore request at most the capped 62.5-Hz response set by
    // the one-hop countdown shortening in processSTFTFrame().
    lastSchedulerListenerX = listenerX;
    lastSchedulerListenerY = listenerY;
    lastSchedulerListenerZ = listenerZ;
    for (int m = 0; m < activeMicCount; ++m)
    {
        lastSchedulerMicX[(size_t) m] = micX[(size_t) m];
        lastSchedulerMicY[(size_t) m] = micY[(size_t) m];
        lastSchedulerMicZ[(size_t) m] = micZ[(size_t) m];
        lastSchedulerMicAzimuth[(size_t) m] = micAzimuth[(size_t) m];
    }
    schedulerGeometryInitialised = true;
    return changed;
}

void DiracGNNEngine::updateGNNBlend(Eigen::MatrixXf& outputWeights) noexcept
{
    if (gnnBlendRemaining <= 0)
    {
        outputWeights = lastValidGNNWeights;
        return;
    }
    const int total = juce::jmax(1, gnnBlendTotalFrames);
    const int completed = total - gnnBlendRemaining + 1;
    const float alpha = juce::jlimit(0.0f, 1.0f, (float) completed / (float) total);
    outputWeights = (1.0f - alpha) * gnnBlendFrom + alpha * gnnBlendTo;
    --gnnBlendRemaining;
    if (gnnBlendRemaining <= 0)
        outputWeights = lastValidGNNWeights;
}

bool DiracGNNEngine::forwardGNN(
    float qPosX,
    float qPosY,
    float qPosZ,
    const std::array<float, kNumMics>& micX,
    const std::array<float, kNumMics>& micY,
    const std::array<float, kNumMics>& micZ,
    Eigen::MatrixXf& outputWeights,
    int activeMicCount,
    const int* binIndices,
    int binCount)
{
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    binCount = juce::jlimit(0, numBins, binCount);
    if (binCount <= 0 || binIndices == nullptr)
        return true;

    const int queryIndex = activeMicCount;
    const int activeNodeCount = activeMicCount + 1;
    const int directedEdges = activeNodeCount * (activeNodeCount - 1);

    lastGNNStage.store(10, std::memory_order_relaxed);
    float posX[kNumNodes] {};
    float posY[kNumNodes] {};
    float posZ[kNumNodes] {};
    for (int i = 0; i < activeMicCount; ++i)
    {
        posX[i] = micX[(size_t) i];
        posY[i] = micY[(size_t) i];
        posZ[i] = micZ[(size_t) i];
    }
    posX[queryIndex] = qPosX;
    posY[queryIndex] = qPosY;
    posZ[queryIndex] = qPosZ;

    for (int i = 0; i < activeNodeCount; ++i)
        hiddenA[(size_t) i].topRows(binCount).setZero();

    for (int i = 0; i < activeMicCount; ++i)
        for (int r = 0; r < binCount; ++r)
        {
            const int f = binIndices[(size_t) r];
            hiddenA[(size_t) i](r, 0) = std::log10(E_obs(f, i) + kEps);
            hiddenA[(size_t) i](r, 1) = psi_obs(f, i);
            hiddenA[(size_t) i](r, 2) = (float) f / (float) (numBins - 1);
            hiddenA[(size_t) i](r, 3) = 1.0f;
            hiddenA[(size_t) i](r, 4) = 0.0f;
        }

    for (int r = 0; r < binCount; ++r)
    {
        const int f = binIndices[(size_t) r];
        hiddenA[(size_t) queryIndex](r, 2) = (float) f / (float) (numBins - 1);
        hiddenA[(size_t) queryIndex](r, 3) = 0.0f;
        hiddenA[(size_t) queryIndex](r, 4) = 1.0f;
    }

    int currentFeatureDim = initialFeatureDim;
    for (int l = 0; l < numLayers; ++l)
    {
        const auto& msgLayer = msgLayers[(size_t) l];
        const auto& updLayer = updLayers[(size_t) l];
        const int msgInputCols = (int) msgLayer.W1T.rows();
        const int msgOutputCols = (int) msgLayer.W2T.cols();
        const int updateInputCols = (int) updLayer.W1T.rows();
        const int nextFeatureDim = (int) updLayer.W2T.cols();

        int edgeIndex = 0;
        for (int i = 0; i < activeNodeCount; ++i)
            for (int j = 0; j < activeNodeCount; ++j)
            {
                if (i == j) continue;
                const float dx = posX[j] - posX[i];
                const float dy = posY[j] - posY[i];
                const float dz = posZ[j] - posZ[i];
                const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                const float invDist = 1.0f / (dist + kEps);
                const float rx = dx * invDist, ry = dy * invDist, rz = dz * invDist;
                const int row0 = edgeIndex * binCount;

                messageBatchInput.block(row0, 0, binCount, currentFeatureDim)
                    = hiddenA[(size_t) i].topRows(binCount).leftCols(currentFeatureDim);
                messageBatchInput.block(row0, currentFeatureDim, binCount, currentFeatureDim)
                    = hiddenA[(size_t) j].topRows(binCount).leftCols(currentFeatureDim);

                for (int r = 0; r < binCount; ++r)
                {
                    const int f = binIndices[(size_t) r];
                    const float uiX = (i < activeMicCount) ? u_obs_X(f, i) : 0.0f;
                    const float uiY = (i < activeMicCount) ? u_obs_Y(f, i) : 0.0f;
                    const float uiZ = (i < activeMicCount) ? u_obs_Z(f, i) : 0.0f;
                    const float ujX = (j < activeMicCount) ? u_obs_X(f, j) : 0.0f;
                    const float ujY = (j < activeMicCount) ? u_obs_Y(f, j) : 0.0f;
                    const float ujZ = (j < activeMicCount) ? u_obs_Z(f, j) : 0.0f;
                    const int row = row0 + r;
                    messageBatchInput(row, 2 * currentFeatureDim + 0) = dist;
                    messageBatchInput(row, 2 * currentFeatureDim + 1) = uiX * rx + uiY * ry + uiZ * rz;
                    messageBatchInput(row, 2 * currentFeatureDim + 2) = ujX * rx + ujY * ry + ujZ * rz;
                    messageBatchInput(row, 2 * currentFeatureDim + 3) = uiX * ujX + uiY * ujY + uiZ * ujZ;
                }
                ++edgeIndex;
            }

        const int messageRows = directedEdges * binCount;
        lastGNNStage.store(20 + l * 10, std::memory_order_relaxed);
        mlpApplyBatch(messageBatchInput, messageRows, msgInputCols, msgLayer, false,
                      messageBatchHidden, messageBatchOutput);

        const int allUpdateRows = activeNodeCount * binCount;
        updateBatchInput.topRows(allUpdateRows).leftCols(updateInputCols).setZero();
        for (int i = 0; i < activeNodeCount; ++i)
            updateBatchInput.block(i * binCount, 0, binCount, currentFeatureDim)
                = hiddenA[(size_t) i].topRows(binCount).leftCols(currentFeatureDim);

        edgeIndex = 0;
        for (int i = 0; i < activeNodeCount; ++i)
            for (int j = 0; j < activeNodeCount; ++j)
            {
                if (i == j) continue;
                updateBatchInput.block(i * binCount, currentFeatureDim, binCount, msgOutputCols)
                    += messageBatchOutput.block(edgeIndex * binCount, 0, binCount, msgOutputCols);
                ++edgeIndex;
            }

        if (activeMicCount != kTrainingMicCount)
        {
            // The checkpoint was trained with five measured nodes. Preserve the
            // expected aggregate-message degree for both N<5 and N>5. N!=5 is
            // therefore technically supported but remains an OOD extrapolation.
            const float trainingDegreeScale = (float) kTrainingMicCount / (float) activeMicCount;
            for (int i = 0; i < activeNodeCount; ++i)
                updateBatchInput.block(i * binCount, currentFeatureDim, binCount, msgOutputCols)
                    *= trainingDegreeScale;
        }

        lastGNNStage.store(25 + l * 10, std::memory_order_relaxed);
        mlpApplyBatch(updateBatchInput, allUpdateRows, updateInputCols, updLayer, false,
                      updateBatchHidden, updateBatchOutput);

        for (int i = 0; i < activeNodeCount; ++i)
            hiddenB[(size_t) i].topRows(binCount).leftCols(nextFeatureDim)
                = updateBatchOutput.block(i * binCount, 0, binCount, nextFeatureDim);
        for (int i = 0; i < activeNodeCount; ++i)
            hiddenA[(size_t) i].swap(hiddenB[(size_t) i]);
        currentFeatureDim = nextFeatureDim;
    }

    for (int m = 0; m < activeMicCount; ++m)
    {
        const int row0 = m * binCount;
        const float dx = qPosX - micX[(size_t) m];
        const float dy = qPosY - micY[(size_t) m];
        const float dz = qPosZ - micZ[(size_t) m];
        const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float invDist = 1.0f / (dist + kEps);
        const float rx = dx * invDist, ry = dy * invDist, rz = dz * invDist;

        outBatchInput.block(row0, 0, binCount, currentFeatureDim)
            = hiddenA[(size_t) m].topRows(binCount).leftCols(currentFeatureDim);
        outBatchInput.block(row0, currentFeatureDim, binCount, currentFeatureDim)
            = hiddenA[(size_t) queryIndex].topRows(binCount).leftCols(currentFeatureDim);
        for (int r = 0; r < binCount; ++r)
        {
            const int f = binIndices[(size_t) r];
            const float ux = u_obs_X(f, m), uy = u_obs_Y(f, m), uz = u_obs_Z(f, m);
            const int row = row0 + r;
            outBatchInput(row, 2 * currentFeatureDim + 0) = dist;
            outBatchInput(row, 2 * currentFeatureDim + 1) = ux * rx + uy * ry + uz * rz;
            outBatchInput(row, 2 * currentFeatureDim + 2) = 0.0f;
            outBatchInput(row, 2 * currentFeatureDim + 3) = 0.0f;
        }
    }

    const int outRows = activeMicCount * binCount;
    lastGNNStage.store(50, std::memory_order_relaxed);
    mlpApplyBatch(outBatchInput, outRows, maxOutInputDim, outLayer, true,
                  outBatchHidden, outBatchOutput);

    float maxWeight = 0.0f;
    bool valid = true;
    for (int m = 0; m < activeMicCount; ++m)
        for (int r = 0; r < binCount; ++r)
        {
            const int f = binIndices[(size_t) r];
            const float w = outBatchOutput(m * binCount + r, 0);
            outputWeights(f, m) = w;
            if (!finitePositive(w)) valid = false;
            else maxWeight = std::max(maxWeight, w);
        }

    lastMaxGNNWeight.store(maxWeight, std::memory_order_relaxed);
    lastGNNStage.store(valid ? 60 : -60, std::memory_order_relaxed);
    return valid;
}

namespace
{
constexpr double kGNNWeightSumTiny = 1.0e-30;
constexpr double kGNNMinAppliedWeightSum = 0.5;
constexpr double kGNNMaxAppliedWeightSum = 2.0;
}

bool DiracGNNEngine::constrainGNNWeightSumsForBins(Eigen::MatrixXf& weights,
                                                    int activeMicCount,
                                                    const int* binIndices,
                                                    int binCount) noexcept
{
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    bool allValid = true;
    for (int r = 0; r < binCount; ++r)
    {
        const int f = binIndices[(size_t) r];
        double rawSum = 0.0;
        bool rowValid = true;
        for (int m = 0; m < activeMicCount; ++m)
        {
            const float w = weights(f, m);
            if (!std::isfinite(w) || w < 0.0f)
            {
                rowValid = false;
                break;
            }
            rawSum += (double) w;
        }

        // The checkpoint can legitimately emit very small positive Softplus
        // values. Their RELATIVE distribution is meaningful, while the global
        // learned gain is allowed to vary only inside [0.5, 2.0]. Therefore we
        // preserve all node ratios and clamp only the row sum.
        if (!rowValid || !(rawSum > kGNNWeightSumTiny) || !std::isfinite(rawSum))
        {
            weights.row(f).setZero();
            allValid = false;
            continue;
        }

        const double targetSum = juce::jlimit(kGNNMinAppliedWeightSum,
                                              kGNNMaxAppliedWeightSum,
                                              rawSum);
        const float scale = (float) (targetSum / rawSum);
        for (int m = 0; m < activeMicCount; ++m)
            weights(f, m) *= scale;
        for (int m = activeMicCount; m < kNumMics; ++m)
            weights(f, m) = 0.0f;
    }
    return allValid;
}

bool DiracGNNEngine::constrainGNNWeightSums(Eigen::MatrixXf& weights, int activeMicCount) noexcept
{
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    bool allValid = true;
    for (int f = 0; f < numBins; ++f)
    {
        double rawSum = 0.0;
        bool rowValid = true;
        for (int m = 0; m < activeMicCount; ++m)
        {
            const float w = weights(f, m);
            if (!std::isfinite(w) || w < 0.0f)
            {
                rowValid = false;
                break;
            }
            rawSum += (double) w;
        }

        if (!rowValid || !(rawSum > kGNNWeightSumTiny) || !std::isfinite(rawSum))
        {
            weights.row(f).setZero();
            allValid = false;
            continue;
        }

        const double targetSum = juce::jlimit(kGNNMinAppliedWeightSum,
                                              kGNNMaxAppliedWeightSum,
                                              rawSum);
        const float scale = (float) (targetSum / rawSum);
        for (int m = 0; m < activeMicCount; ++m)
            weights(f, m) *= scale;
        for (int m = activeMicCount; m < kNumMics; ++m)
            weights(f, m) = 0.0f;
    }
    return allValid;
}

DiracGNNEngine::WeightVisualizationSnapshot
DiracGNNEngine::getWeightVisualizationSnapshot() const noexcept
{
    WeightVisualizationSnapshot out;
    for (int tries = 0; tries < 4; ++tries)
    {
        const uint32_t before = weightVizVersion.load(std::memory_order_acquire);
        if ((before & 1u) != 0u) continue;
        for (int m = 0; m < kNumMics; ++m)
        {
            out.rawMean[(size_t) m] = weightVizRawMean[(size_t) m].load(std::memory_order_relaxed);
            out.gnnRelativeMean[(size_t) m] = weightVizGNNRelativeMean[(size_t) m].load(std::memory_order_relaxed);
            out.delaunayRelativeMean[(size_t) m] = weightVizDelaunayRelativeMean[(size_t) m].load(std::memory_order_relaxed);
            out.relativeMean[(size_t) m] = weightVizRelativeMean[(size_t) m].load(std::memory_order_relaxed);
            out.relativeLow[(size_t) m] = weightVizRelativeLow[(size_t) m].load(std::memory_order_relaxed);
            out.relativeMid[(size_t) m] = weightVizRelativeMid[(size_t) m].load(std::memory_order_relaxed);
            out.relativeHigh[(size_t) m] = weightVizRelativeHigh[(size_t) m].load(std::memory_order_relaxed);
        }
        out.meanRawSum = weightVizMeanRawSum.load(std::memory_order_relaxed);
        out.meanAppliedSum = weightVizMeanAppliedSum.load(std::memory_order_relaxed);
        out.maxRawWeight = weightVizMaxRawWeight.load(std::memory_order_relaxed);
        out.meanScale = weightVizMeanScale.load(std::memory_order_relaxed);
        out.maxScale = weightVizMaxScale.load(std::memory_order_relaxed);
        out.lowClampFraction = weightVizLowClampFraction.load(std::memory_order_relaxed);
        out.highClampFraction = weightVizHighClampFraction.load(std::memory_order_relaxed);
        out.gnnBinFraction = weightVizGNNBinFraction.load(std::memory_order_relaxed);
        out.frameCounter = weightVizFrameCounter.load(std::memory_order_relaxed);
        out.gnnMode = weightVizGnnMode.load(std::memory_order_relaxed);
        const uint32_t after = weightVizVersion.load(std::memory_order_acquire);
        if (before == after && (after & 1u) == 0u) return out;
    }
    return out;
}

void DiracGNNEngine::publishWeightVisualization(const Eigen::MatrixXf& rawWeights,
                                                const Eigen::MatrixXf& gnnWeights,
                                                const Eigen::MatrixXf& delaunayWeights,
                                                const Eigen::MatrixXf& appliedWeights,
                                                bool gnnMode) noexcept
{
    std::array<float, kNumMics> raw {}, gnnRel {}, delRel {}, appliedRel {}, low {}, mid {}, high {};
    int nLow = 0, nMid = 0, nHigh = 0;
    int gnnBins = 0, gnnClampMeasuredBins = 0, lowClampBins = 0, highClampBins = 0;
    double gnnMixTotal = 0.0;
    float meanRawSum = 0.0f, meanAppliedSum = 0.0f, maxRaw = 0.0f;
    double scaleSum = 0.0, scaleMax = 1.0;

    for (int f = 0; f < numBins; ++f)
    {
        float rawSum = 0.0f, gnnSum = 0.0f, delSum = 0.0f, appliedSum = 0.0f;
        for (int m = 0; m < kNumMics; ++m)
        {
            const float rw = std::max(0.0f, rawWeights(f, m));
            const float gw = std::max(0.0f, gnnWeights(f, m));
            const float dw = std::max(0.0f, delaunayWeights(f, m));
            const float aw = std::max(0.0f, appliedWeights(f, m));
            raw[(size_t) m] += rw;
            rawSum += rw;
            gnnSum += gw;
            delSum += dw;
            appliedSum += aw;
            maxRaw = std::max(maxRaw, rw);
        }

        meanRawSum += rawSum;
        meanAppliedSum += appliedSum;
        const bool gnnActive = gnnMode && useGNNForBin[(size_t) f] != 0u;
        if (gnnActive)
            ++gnnBins;
        if (gnnMode)
            gnnMixTotal += (double) juce::jlimit(0.0f, 1.0f, gnnGateMix[(size_t) f]);

        if (gnnActive && rawSum > 1.0e-30f && std::isfinite(rawSum))
        {
            ++gnnClampMeasuredBins;
            if (rawSum < 0.5f) ++lowClampBins;
            if (rawSum > 2.0f) ++highClampBins;
            const double scale = (double) gnnSum / (double) rawSum;
            scaleSum += scale;
            scaleMax = std::max(scaleMax, scale);
        }

        const bool isLow = f <= 32;                // 0-1 kHz @ 16 kHz / 512 FFT
        const bool isMid = f > 32 && f <= 128;    // 1-4 kHz
        if (isLow) ++nLow; else if (isMid) ++nMid; else ++nHigh;

        for (int m = 0; m < kNumMics; ++m)
        {
            if (gnnSum > kEps)
                gnnRel[(size_t) m] += std::max(0.0f, gnnWeights(f, m)) / gnnSum;
            if (delSum > kEps)
                delRel[(size_t) m] += std::max(0.0f, delaunayWeights(f, m)) / delSum;
            if (appliedSum > kEps)
            {
                const float r = std::max(0.0f, appliedWeights(f, m)) / appliedSum;
                appliedRel[(size_t) m] += r;
                if (isLow) low[(size_t) m] += r;
                else if (isMid) mid[(size_t) m] += r;
                else high[(size_t) m] += r;
            }
        }
    }

    const float invAll = 1.0f / (float) numBins;
    const float invLow = nLow ? 1.0f / (float) nLow : 0.0f;
    const float invMid = nMid ? 1.0f / (float) nMid : 0.0f;
    const float invHigh = nHigh ? 1.0f / (float) nHigh : 0.0f;
    for (int m = 0; m < kNumMics; ++m)
    {
        raw[(size_t) m] *= invAll;
        gnnRel[(size_t) m] *= invAll;
        delRel[(size_t) m] *= invAll;
        appliedRel[(size_t) m] *= invAll;
        low[(size_t) m] *= invLow;
        mid[(size_t) m] *= invMid;
        high[(size_t) m] *= invHigh;
    }
    meanRawSum *= invAll;
    meanAppliedSum *= invAll;
    const float meanScale = gnnClampMeasuredBins > 0
                          ? (float) std::min(scaleSum / (double) gnnClampMeasuredBins, 1.0e12)
                          : 1.0f;
    const float maxScale = (float) std::min(scaleMax, 1.0e12);
    const float lowClampFraction = gnnClampMeasuredBins > 0
                                 ? (float) lowClampBins / (float) gnnClampMeasuredBins : 0.0f;
    const float highClampFraction = gnnClampMeasuredBins > 0
                                  ? (float) highClampBins / (float) gnnClampMeasuredBins : 0.0f;
    const float gnnFraction = gnnMode ? (float) (gnnMixTotal * (double) invAll) : 0.0f;

    constexpr float alpha = 0.10f;
    if (!weightVizEmaInitialised)
    {
        weightVizEmaRawMean = raw;
        weightVizEmaGNNRelativeMean = gnnRel;
        weightVizEmaDelaunayRelativeMean = delRel;
        weightVizEmaRelativeMean = appliedRel;
        weightVizEmaRelativeLow = low;
        weightVizEmaRelativeMid = mid;
        weightVizEmaRelativeHigh = high;
        weightVizEmaMeanRawSum = meanRawSum;
        weightVizEmaMeanAppliedSum = meanAppliedSum;
        weightVizEmaMeanScale = meanScale;
        weightVizEmaMaxScale = maxScale;
        weightVizEmaLowClampFraction = lowClampFraction;
        weightVizEmaHighClampFraction = highClampFraction;
        weightVizEmaGNNBinFraction = gnnFraction;
        weightVizEmaInitialised = true;
    }
    else
    {
        for (int m = 0; m < kNumMics; ++m)
        {
            const size_t i = (size_t) m;
            weightVizEmaRawMean[i] += alpha * (raw[i] - weightVizEmaRawMean[i]);
            weightVizEmaGNNRelativeMean[i] += alpha * (gnnRel[i] - weightVizEmaGNNRelativeMean[i]);
            weightVizEmaDelaunayRelativeMean[i] += alpha * (delRel[i] - weightVizEmaDelaunayRelativeMean[i]);
            weightVizEmaRelativeMean[i] += alpha * (appliedRel[i] - weightVizEmaRelativeMean[i]);
            weightVizEmaRelativeLow[i] += alpha * (low[i] - weightVizEmaRelativeLow[i]);
            weightVizEmaRelativeMid[i] += alpha * (mid[i] - weightVizEmaRelativeMid[i]);
            weightVizEmaRelativeHigh[i] += alpha * (high[i] - weightVizEmaRelativeHigh[i]);
        }
        weightVizEmaMeanRawSum += alpha * (meanRawSum - weightVizEmaMeanRawSum);
        weightVizEmaMeanAppliedSum += alpha * (meanAppliedSum - weightVizEmaMeanAppliedSum);
        weightVizEmaMeanScale += alpha * (meanScale - weightVizEmaMeanScale);
        weightVizEmaMaxScale += alpha * (maxScale - weightVizEmaMaxScale);
        weightVizEmaLowClampFraction += alpha * (lowClampFraction - weightVizEmaLowClampFraction);
        weightVizEmaHighClampFraction += alpha * (highClampFraction - weightVizEmaHighClampFraction);
        weightVizEmaGNNBinFraction += alpha * (gnnFraction - weightVizEmaGNNBinFraction);
    }

    weightVizVersion.fetch_add(1u, std::memory_order_acq_rel);
    for (int m = 0; m < kNumMics; ++m)
    {
        const size_t i = (size_t) m;
        weightVizRawMean[i].store(weightVizEmaRawMean[i], std::memory_order_relaxed);
        weightVizGNNRelativeMean[i].store(weightVizEmaGNNRelativeMean[i], std::memory_order_relaxed);
        weightVizDelaunayRelativeMean[i].store(weightVizEmaDelaunayRelativeMean[i], std::memory_order_relaxed);
        weightVizRelativeMean[i].store(weightVizEmaRelativeMean[i], std::memory_order_relaxed);
        weightVizRelativeLow[i].store(weightVizEmaRelativeLow[i], std::memory_order_relaxed);
        weightVizRelativeMid[i].store(weightVizEmaRelativeMid[i], std::memory_order_relaxed);
        weightVizRelativeHigh[i].store(weightVizEmaRelativeHigh[i], std::memory_order_relaxed);
    }
    weightVizMeanRawSum.store(weightVizEmaMeanRawSum, std::memory_order_relaxed);
    weightVizMeanAppliedSum.store(weightVizEmaMeanAppliedSum, std::memory_order_relaxed);
    weightVizMaxRawWeight.store(maxRaw, std::memory_order_relaxed);
    weightVizMeanScale.store(weightVizEmaMeanScale, std::memory_order_relaxed);
    weightVizMaxScale.store(weightVizEmaMaxScale, std::memory_order_relaxed);
    weightVizLowClampFraction.store(weightVizEmaLowClampFraction, std::memory_order_relaxed);
    weightVizHighClampFraction.store(weightVizEmaHighClampFraction, std::memory_order_relaxed);
    weightVizGNNBinFraction.store(weightVizEmaGNNBinFraction, std::memory_order_relaxed);
    weightVizGnnMode.store(gnnMode, std::memory_order_relaxed);
    weightVizFrameCounter.fetch_add(1u, std::memory_order_relaxed);
    weightVizVersion.fetch_add(1u, std::memory_order_release);
}

void DiracGNNEngine::selectAnchor(float listenerX,
                                  float listenerY,
                                  float listenerZ,
                                  const std::array<float, kNumMics>& micX,
                                  const std::array<float, kNumMics>& micY,
                                  const std::array<float, kNumMics>& micZ,
                                  int activeMicCount,
                                  bool exactNearestPrototype) noexcept
{
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    int nearest = 0;
    float nearestD2 = std::numeric_limits<float>::max();
    float currentD2 = std::numeric_limits<float>::max();

    for (int m = 0; m < activeMicCount; ++m)
    {
        const float dx = micX[(size_t) m] - listenerX;
        const float dy = micY[(size_t) m] - listenerY;
        const float dz = micZ[(size_t) m] - listenerZ;
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < nearestD2)
        {
            nearestD2 = d2;
            nearest = m;
        }
        if (m == currentAnchor)
            currentD2 = d2;
    }

    if (currentAnchor < 0)
    {
        currentAnchor = nearest;
        return;
    }

    // Pure Delaunay mode follows the paper literally: the prototype is the
    // closest microphone, with no spatial hysteresis. The learned path keeps
    // the learned-path anchor policy so the geometric baseline does not
    // silently change GNN audio during A/B comparison.
    const bool shouldSwitch = exactNearestPrototype
                            ? (nearest != currentAnchor)
                            : (nearest != currentAnchor
                               && nearestD2 < kLearnedAnchorSwitchRatio * kLearnedAnchorSwitchRatio * currentD2);
    if (shouldSwitch)
    {
        previousAnchor = currentAnchor;
        currentAnchor = nearest;
        anchorCrossfadeTotalFrames = exactNearestPrototype
                                   ? kPaperAnchorCrossfadeFrames
                                   : kLearnedAnchorCrossfadeFrames;
        anchorCrossfadeRemaining = anchorCrossfadeTotalFrames;
    }
}

std::complex<float> DiracGNNEngine::getTransitionAnchorSpectrum(int c, int bin) const noexcept
{
    const int currentBase = juce::jlimit(0, kNumMics - 1, currentAnchor) * 4;
    const auto current = spectra[(size_t) (currentBase + c) * (size_t) numBins + (size_t) bin];

    if (previousAnchor < 0 || anchorCrossfadeRemaining <= 0)
        return current;

    const int previousBase = previousAnchor * 4;
    const auto previous = spectra[(size_t) (previousBase + c) * (size_t) numBins + (size_t) bin];
    const auto previousW = spectra[(size_t) previousBase * (size_t) numBins + (size_t) bin];
    const auto currentW = spectra[(size_t) currentBase * (size_t) numBins + (size_t) bin];

    std::complex<float> phaseAlign(1.0f, 0.0f);
    const float oldMag = std::abs(previousW);
    const float newMag = std::abs(currentW);
    if (oldMag > kEps && newMag > kEps)
        phaseAlign = (previousW / oldMag) * std::conj(currentW / newMag);

    const int transitionFrames = std::max(1, anchorCrossfadeTotalFrames);
    const float alpha = 1.0f - (float) anchorCrossfadeRemaining / (float) transitionFrames;
    // At the start the new anchor is phase-aligned to the old W channel. The
    // phase correction is then relaxed continuously to unity, so the frame
    // after the crossfade does not contain a hidden phase discontinuity.
    const float alignAngle = std::atan2(phaseAlign.imag(), phaseAlign.real());
    const auto relaxedAlignment = std::polar(1.0f, alignAngle * (1.0f - alpha));
    return (1.0f - alpha) * previous + alpha * current * relaxedAlignment;
}

bool DiracGNNEngine::barycentric(const Vec2& p,
                                 const Vec2& a,
                                 const Vec2& b,
                                 const Vec2& c,
                                 std::array<float, 3>& w) noexcept
{
    const float denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if (std::abs(denom) < 1.0e-7f)
        return false;
    w[0] = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denom;
    w[1] = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denom;
    w[2] = 1.0f - w[0] - w[1];
    return true;
}

int DiracGNNEngine::buildDelaunayTriangles(
    const std::array<Vec2, kNumMics>& points,
    int activeMicCount,
    std::array<std::array<int, 3>, 32>& output) noexcept
{
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    output.fill({ -1, -1, -1 });
    if (activeMicCount < 3)
        return 0;

    // Deterministic Bowyer-Watson triangulation. A naive empty-circumcircle
    // enumerated every empty-circumcircle triplet independently. That is not a
    // triangulation in cocircular cases (e.g. a square): both diagonals can be
    // accepted simultaneously, producing overlapping triangles and spatial
    // discontinuities. Sorting the insertion order geometrically and treating
    // on-circle points as part of the cavity yields one consistent Delaunay
    // triangulation independent of the microphone numbering.
    struct Triangle { int a = -1, b = -1, c = -1; };
    struct Edge { int a = -1, b = -1, count = 0; };

    constexpr int kMaxVertices = kNumMics + 3;
    constexpr int kMaxTriangles = 64;
    constexpr int kMaxEdges = 192;
    std::array<Vec2, kMaxVertices> verts {};
    for (int i = 0; i < activeMicCount; ++i)
        verts[(size_t) i] = points[(size_t) i];

    float minX = points[0].x, maxX = points[0].x;
    float minY = points[0].y, maxY = points[0].y;
    for (int i = 1; i < activeMicCount; ++i)
    {
        minX = std::min(minX, points[(size_t) i].x);
        maxX = std::max(maxX, points[(size_t) i].x);
        minY = std::min(minY, points[(size_t) i].y);
        maxY = std::max(maxY, points[(size_t) i].y);
    }
    const float dx = maxX - minX;
    const float dy = maxY - minY;
    const float span = std::max(std::max(dx, dy), 1.0f);
    const float cx = 0.5f * (minX + maxX);
    const float cy = 0.5f * (minY + maxY);
    const int s0 = activeMicCount;
    const int s1 = activeMicCount + 1;
    const int s2 = activeMicCount + 2;
    verts[(size_t) s0] = { cx - 32.0f * span, cy - 2.0f * span };
    verts[(size_t) s1] = { cx,                cy + 32.0f * span };
    verts[(size_t) s2] = { cx + 32.0f * span, cy - 2.0f * span };

    auto orient = [&](int ia, int ib, int ic) noexcept
    {
        const auto& a = verts[(size_t) ia];
        const auto& b = verts[(size_t) ib];
        const auto& c = verts[(size_t) ic];
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    };

    auto circumcircleContains = [&](const Triangle& t, int ip) noexcept
    {
        const auto& a = verts[(size_t) t.a];
        const auto& b = verts[(size_t) t.b];
        const auto& c = verts[(size_t) t.c];
        const auto& p = verts[(size_t) ip];
        const double d = 2.0 * (double(a.x) * (double(b.y) - double(c.y))
                              + double(b.x) * (double(c.y) - double(a.y))
                              + double(c.x) * (double(a.y) - double(b.y)));
        if (std::abs(d) < 1.0e-12)
            return false;
        const double a2 = double(a.x) * a.x + double(a.y) * a.y;
        const double b2 = double(b.x) * b.x + double(b.y) * b.y;
        const double c2 = double(c.x) * c.x + double(c.y) * c.y;
        const double ux = (a2 * (double(b.y) - double(c.y))
                         + b2 * (double(c.y) - double(a.y))
                         + c2 * (double(a.y) - double(b.y))) / d;
        const double uy = (a2 * (double(c.x) - double(b.x))
                         + b2 * (double(a.x) - double(c.x))
                         + c2 * (double(b.x) - double(a.x))) / d;
        const double r2 = (ux - a.x) * (ux - a.x) + (uy - a.y) * (uy - a.y);
        const double pd2 = (ux - p.x) * (ux - p.x) + (uy - p.y) * (uy - p.y);
        const double tol = 1.0e-10 * std::max(1.0, r2);
        return pd2 <= r2 + tol;
    };

    std::array<int, kNumMics> order {};
    for (int i = 0; i < activeMicCount; ++i)
        order[(size_t) i] = i;
    // Fixed-size selection sort avoids heap allocation in the model worker.
    for (int i = 0; i < activeMicCount - 1; ++i)
    {
        int best = i;
        for (int j = i + 1; j < activeMicCount; ++j)
        {
            const auto& a = points[(size_t) order[(size_t) j]];
            const auto& b = points[(size_t) order[(size_t) best]];
            if (a.x < b.x || (a.x == b.x && (a.y < b.y || (a.y == b.y && order[(size_t) j] < order[(size_t) best]))))
                best = j;
        }
        std::swap(order[(size_t) i], order[(size_t) best]);
    }

    std::array<Triangle, kMaxTriangles> tris {};
    int triCount = 1;
    tris[0] = { s0, s2, s1 }; // CCW for the super triangle above.

    for (int oi = 0; oi < activeMicCount; ++oi)
    {
        const int ip = order[(size_t) oi];
        std::array<bool, kMaxTriangles> bad {};
        std::array<Edge, kMaxEdges> edges {};
        int edgeCount = 0;

        auto addEdge = [&](int a, int b) noexcept
        {
            if (a > b) std::swap(a, b);
            for (int e = 0; e < edgeCount; ++e)
            {
                if (edges[(size_t) e].a == a && edges[(size_t) e].b == b)
                {
                    ++edges[(size_t) e].count;
                    return;
                }
            }
            if (edgeCount < kMaxEdges)
                edges[(size_t) edgeCount++] = { a, b, 1 };
        };

        for (int t = 0; t < triCount; ++t)
        {
            if (!circumcircleContains(tris[(size_t) t], ip))
                continue;
            bad[(size_t) t] = true;
            addEdge(tris[(size_t) t].a, tris[(size_t) t].b);
            addEdge(tris[(size_t) t].b, tris[(size_t) t].c);
            addEdge(tris[(size_t) t].c, tris[(size_t) t].a);
        }

        int keep = 0;
        for (int t = 0; t < triCount; ++t)
            if (!bad[(size_t) t])
                tris[(size_t) keep++] = tris[(size_t) t];
        triCount = keep;

        for (int e = 0; e < edgeCount; ++e)
        {
            if (edges[(size_t) e].count != 1 || triCount >= kMaxTriangles)
                continue;
            Triangle nt { edges[(size_t) e].a, edges[(size_t) e].b, ip };
            const float o = orient(nt.a, nt.b, nt.c);
            if (std::abs(o) < 1.0e-9f)
                continue;
            if (o < 0.0f)
                std::swap(nt.a, nt.b);
            tris[(size_t) triCount++] = nt;
        }
    }

    int outCount = 0;
    for (int t = 0; t < triCount && outCount < (int) output.size(); ++t)
    {
        const auto& tr = tris[(size_t) t];
        if (tr.a >= activeMicCount || tr.b >= activeMicCount || tr.c >= activeMicCount)
            continue;
        output[(size_t) outCount++] = { tr.a, tr.b, tr.c };
    }
    return outCount;
}

DiracGNNEngine::Vec2 DiracGNNEngine::closestPointOnSegment(const Vec2& p,
                                                              const Vec2& a,
                                                              const Vec2& b,
                                                              float& t) noexcept
{
    const float abx = b.x - a.x;
    const float aby = b.y - a.y;
    const float denom = abx * abx + aby * aby;
    if (denom <= 1.0e-12f)
    {
        t = 0.0f;
        return a;
    }
    t = ((p.x - a.x) * abx + (p.y - a.y) * aby) / denom;
    t = juce::jlimit(0.0f, 1.0f, t);
    return { a.x + t * abx, a.y + t * aby };
}

void DiracGNNEngine::computeDelaunayWeights(
    float qPosX,
    float qPosY,
    const std::array<float, kNumMics>& micX,
    const std::array<float, kNumMics>& micY,
    Eigen::MatrixXf& outputWeights,
    int activeMicCount) const noexcept
{
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    std::array<Vec2, kNumMics> points {};
    for (int i = 0; i < activeMicCount; ++i)
        points[(size_t) i] = { micX[(size_t) i], micY[(size_t) i] };

    const Vec2 q { qPosX, qPosY };
    std::array<float, kNumMics> selected {};
    bool found = false;

    if (activeMicCount == 1)
    {
        selected[0] = 1.0f;
        found = true;
    }
    else if (activeMicCount == 2)
    {
        // The paper's interpolation is triangular and therefore formally starts
        // at M>=3. For the supported two-node plug-in mode, use the exact 1-D
        // analogue: barycentric interpolation on the segment, clamped to its
        // closest point outside the segment. No all-node IDW fallback exists.
        float t = 0.0f;
        closestPointOnSegment(q, points[0], points[1], t);
        selected[0] = 1.0f - t;
        selected[1] = t;
        found = true;
    }
    else
    {
        std::array<std::array<int, 3>, 32> triangles {};
        const int triangleCount = buildDelaunayTriangles(points, activeMicCount, triangles);

        // WASPAA 2023, Eq. (8): use exactly the triangle of the Delaunay
        // triangulation that contains the listener. Every microphone outside
        // this local triplet has weight exactly zero.
        for (int ti = 0; ti < triangleCount && !found; ++ti)
        {
            const int a = triangles[(size_t) ti][0];
            const int b = triangles[(size_t) ti][1];
            const int c = triangles[(size_t) ti][2];
            std::array<float, 3> w {};
            if (!barycentric(q, points[(size_t) a], points[(size_t) b], points[(size_t) c], w))
                continue;
            constexpr float tol = -1.0e-5f;
            if (w[0] < tol || w[1] < tol || w[2] < tol)
                continue;

            selected.fill(0.0f);
            selected[(size_t) a] = std::max(0.0f, w[0]);
            selected[(size_t) b] = std::max(0.0f, w[1]);
            selected[(size_t) c] = std::max(0.0f, w[2]);
            float sum = selected[(size_t) a] + selected[(size_t) b] + selected[(size_t) c];
            if (sum > kEps)
            {
                for (int m = 0; m < activeMicCount; ++m)
                    selected[(size_t) m] /= sum;
                found = true;
            }
        }

        if (!found && triangleCount > 0)
        {
            // Outside the recording region the paper evaluates the closest
            // point on the edge of the closest triplet. Since the Delaunay
            // triangles tile the convex hull, this is the projection onto the
            // triangulated region. No inverse-distance extrapolation is used.
            float bestD2 = std::numeric_limits<float>::max();
            std::array<float, kNumMics> bestWeights {};
            bool haveBoundary = false;

            for (int ti = 0; ti < triangleCount; ++ti)
            {
                const int a = triangles[(size_t) ti][0];
                const int b = triangles[(size_t) ti][1];
                const int c = triangles[(size_t) ti][2];
                const int ids[3] = { a, b, c };
                for (int e = 0; e < 3; ++e)
                {
                    const int i0 = ids[e];
                    const int i1 = ids[(e + 1) % 3];
                    float t = 0.0f;
                    const Vec2 cp = closestPointOnSegment(q, points[(size_t) i0], points[(size_t) i1], t);
                    const float ex = cp.x - q.x;
                    const float ey = cp.y - q.y;
                    const float d2 = ex * ex + ey * ey;
                    if (d2 >= bestD2 - 1.0e-9f)
                        continue;

                    std::array<float, 3> w {};
                    if (!barycentric(cp, points[(size_t) a], points[(size_t) b], points[(size_t) c], w))
                        continue;
                    std::array<float, kNumMics> candidate {};
                    candidate[(size_t) a] = std::max(0.0f, w[0]);
                    candidate[(size_t) b] = std::max(0.0f, w[1]);
                    candidate[(size_t) c] = std::max(0.0f, w[2]);
                    float sum = 0.0f;
                    for (int m = 0; m < activeMicCount; ++m)
                        sum += candidate[(size_t) m];
                    if (sum <= kEps)
                        continue;
                    for (int m = 0; m < activeMicCount; ++m)
                        candidate[(size_t) m] /= sum;

                    bestD2 = d2;
                    bestWeights = candidate;
                    haveBoundary = true;
                }
            }

            if (haveBoundary)
            {
                selected = bestWeights;
                found = true;
            }
        }
    }

    if (!found)
    {
        // Degenerate geometry only (e.g. >=3 collinear/duplicated points): use
        // the closest microphone pair as a 1-D clamped segment. This keeps the
        // baseline local and sparse instead of silently reverting to all-node
        // inverse-distance interpolation.
        float bestD2 = std::numeric_limits<float>::max();
        float bestLen2 = std::numeric_limits<float>::max();
        int bestA = 0, bestB = std::min(1, activeMicCount - 1);
        float bestT = 0.0f;
        for (int a = 0; a < activeMicCount; ++a)
            for (int b = a + 1; b < activeMicCount; ++b)
            {
                float t = 0.0f;
                const Vec2 cp = closestPointOnSegment(q, points[(size_t) a], points[(size_t) b], t);
                const float dx = cp.x - q.x;
                const float dy = cp.y - q.y;
                const float d2 = dx * dx + dy * dy;
                const float sx = points[(size_t) b].x - points[(size_t) a].x;
                const float sy = points[(size_t) b].y - points[(size_t) a].y;
                const float len2 = sx * sx + sy * sy;
                if (d2 < bestD2 - 1.0e-9f || (std::abs(d2 - bestD2) <= 1.0e-9f && len2 < bestLen2))
                {
                    bestD2 = d2;
                    bestLen2 = len2;
                    bestA = a;
                    bestB = b;
                    bestT = t;
                }
            }
        selected.fill(0.0f);
        selected[(size_t) bestA] = 1.0f - bestT;
        selected[(size_t) bestB] = bestT;
    }

    outputWeights.setZero();
    for (int f = 0; f < numBins; ++f)
        for (int m = 0; m < activeMicCount; ++m)
            outputWeights(f, m) = selected[(size_t) m];
}

DiracGNNEngine::SpatialParams DiracGNNEngine::interpolateLearnedParams(
    int bin, int activeMicCount) const noexcept
{
    SpatialParams out;
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    if (bin < 0 || bin >= numBins)
        return out;

    float sumW = 0.0f;
    float psiWeighted = 0.0f;
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    for (int m = 0; m < activeMicCount; ++m)
    {
        const float w = weightsGNNConstrained(bin, m);
        const float em = E_obs(bin, m);
        const float psi = psi_obs(bin, m);
        out.energy += w * em;
        sumW += w;
        psiWeighted += w * psi;
        const float direct = w * em * (1.0f - psi);
        vx += direct * u_obs_X(bin, m);
        vy += direct * u_obs_Y(bin, m);
        vz += direct * u_obs_Z(bin, m);
    }
    if (sumW > kEps)
    {
        out.diffuseness = juce::jlimit(0.0f, 1.0f, psiWeighted / sumW);
        const float normV = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (normV > kEps)
        {
            out.ux = vx / normV;
            out.uy = vy / normV;
            out.uz = vz / normV;
        }
    }
    return out;
}

DiracGNNEngine::SpatialParams DiracGNNEngine::blendSpatialParams(
    const SpatialParams& delaunay, const SpatialParams& learned, float learnedMix) noexcept
{
    const float a = juce::jlimit(0.0f, 1.0f, learnedMix);
    SpatialParams out;
    out.energy = (1.0f - a) * delaunay.energy + a * learned.energy;

    // Interpolate normalized intensity v=(1-psi)u. This is continuous in both
    // direction and diffuseness and reduces exactly to each original renderer
    // at a=0 and a=1.
    const float dv = 1.0f - juce::jlimit(0.0f, 1.0f, delaunay.diffuseness);
    const float lv = 1.0f - juce::jlimit(0.0f, 1.0f, learned.diffuseness);
    const float vx = (1.0f - a) * dv * delaunay.ux + a * lv * learned.ux;
    const float vy = (1.0f - a) * dv * delaunay.uy + a * lv * learned.uy;
    const float vz = (1.0f - a) * dv * delaunay.uz + a * lv * learned.uz;
    const float norm = std::sqrt(vx * vx + vy * vy + vz * vz);
    const float clampedNorm = juce::jlimit(0.0f, 1.0f, norm);
    out.diffuseness = 1.0f - clampedNorm;
    if (norm > kEps)
    {
        out.ux = vx / norm;
        out.uy = vy / norm;
        out.uz = vz / norm;
    }
    return out;
}

DiracGNNEngine::SpatialParams DiracGNNEngine::interpolatePaperDelaunayParams(
    int bin,
    const Eigen::MatrixXf& weights,
    int activeMicCount) const noexcept
{
    SpatialParams out {};
    activeMicCount = juce::jlimit(1, kNumMics, activeMicCount);
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    float sumW = 0.0f;

    for (int m = 0; m < activeMicCount; ++m)
    {
        const float g = std::max(0.0f, weights(bin, m));
        const float em = std::max(0.0f, E_obs(bin, m));
        const float oneMinusPsi = juce::jlimit(0.0f, 1.0f, 1.0f - psi_obs(bin, m));
        out.energy += g * em; // Eq. (9)
        nx += g * oneMinusPsi * u_obs_X(bin, m); // Eq. (10): i/E
        ny += g * oneMinusPsi * u_obs_Y(bin, m);
        nz += g * oneMinusPsi * u_obs_Z(bin, m);
        sumW += g;
    }

    if (sumW <= kEps)
        return out;

    // Delaunay weights are unit-sum, but renormalize defensively for numerical
    // edge cases before applying the exact spatial-parameter equations.
    out.energy /= sumW;
    nx /= sumW;
    ny /= sumW;
    nz /= sumW;

    const float norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    out.diffuseness = juce::jlimit(0.0f, 1.0f, 1.0f - norm); // Eq. (11)
    if (norm > kEps)
    {
        out.ux = nx / norm;
        out.uy = ny / norm;
        out.uz = nz / norm;
    }
    return out;
}

void DiracGNNEngine::applyTemporalWeightWindowForBins(const Eigen::MatrixXf& currentWeights,
                                                       const int* binIndices,
                                                       int binCount,
                                                       int temporalWindow,
                                                       Eigen::MatrixXf& outputWeights) noexcept
{
    temporalWindow = juce::jlimit(1, kMaxTemporalWindow, temporalWindow);
    if (binCount <= 0 || binIndices == nullptr)
        return;

    const int slot = weightHistoryWriteIndex;
    weightHistoryValid[(size_t) slot].fill(0u);
    for (int r = 0; r < binCount; ++r)
    {
        const int f = binIndices[(size_t) r];
        weightHistory[(size_t) slot].row(f) = currentWeights.row(f);
        weightHistoryValid[(size_t) slot][(size_t) f] = 1u;
    }
    weightHistoryWriteIndex = (weightHistoryWriteIndex + 1) % kMaxTemporalWindow;
    weightHistoryCount = std::min(weightHistoryCount + 1, kMaxTemporalWindow);

    // Only touch bins from the current inference. History entries from frames
    // where a bin was below threshold are ignored rather than treated as zero.
    for (int r = 0; r < binCount; ++r)
    {
        const int f = binIndices[(size_t) r];
        outputWeights.row(f).setZero();
        int used = 0;
        const int maxLookback = std::min(weightHistoryCount, temporalWindow);
        for (int n = 0; n < maxLookback; ++n)
        {
            int idx = weightHistoryWriteIndex - 1 - n;
            if (idx < 0) idx += kMaxTemporalWindow;
            if (weightHistoryValid[(size_t) idx][(size_t) f] == 0u)
                continue;
            outputWeights.row(f) += weightHistory[(size_t) idx].row(f);
            ++used;
        }
        if (used > 1)
            outputWeights.row(f) *= 1.0f / (float) used;
        else if (used == 0)
            outputWeights.row(f) = currentWeights.row(f);
    }
}

void DiracGNNEngine::invalidateTemporalHistoryForBin(int bin) noexcept
{
    if (bin < 0 || bin >= numBins)
        return;
    for (auto& valid : weightHistoryValid)
        valid[(size_t) bin] = 0u;
}

bool DiracGNNEngine::acousticFeaturesChanged(const int* binIndices,
                                              int binCount,
                                              int activeMicCount,
                                              float& noveltyOut) const noexcept
{
    noveltyOut = 0.0f;
    if (binIndices == nullptr || binCount <= 0 || activeMicCount <= 0)
        return false;

    double sum = 0.0;
    float maxLocal = 0.0f;
    int used = 0;
    for (int r = 0; r < binCount; ++r)
    {
        const int f = binIndices[(size_t) r];
        if (f < 0 || f >= numBins || lastInferenceFeatureValid[(size_t) f] == 0u)
            continue;
        for (int m = 0; m < activeMicCount; ++m)
        {
            const float eNow = std::max(E_obs(f, m), kEps);
            const float eOld = std::max(lastInferenceE(f, m), kEps);
            const float dE = std::abs(eNow - eOld) / (eNow + eOld + kEps);
            const float dPsi = std::abs(psi_obs(f, m) - lastInferencePsi(f, m));
            const float dot = juce::jlimit(-1.0f, 1.0f,
                u_obs_X(f, m) * lastInferenceUX(f, m)
              + u_obs_Y(f, m) * lastInferenceUY(f, m)
              + u_obs_Z(f, m) * lastInferenceUZ(f, m));
            const float directionReliability = 1.0f - juce::jlimit(0.0f, 1.0f,
                0.5f * (psi_obs(f, m) + lastInferencePsi(f, m)));
            const float dDir = 0.5f * (1.0f - dot) * directionReliability;
            const float local = 0.45f * dE + 0.20f * dPsi + 0.35f * dDir;
            sum += local;
            maxLocal = std::max(maxLocal, local);
            ++used;
        }
    }

    if (used <= 0)
        return false;
    const float mean = (float) (sum / (double) used);
    noveltyOut = std::max(mean, 0.5f * maxLocal);
    return mean >= 0.055f || maxLocal >= 0.22f;
}

void DiracGNNEngine::captureAcousticFeatureSnapshot(const int* binIndices,
                                                     int binCount,
                                                     int activeMicCount) noexcept
{
    if (binIndices == nullptr || binCount <= 0)
        return;
    for (int r = 0; r < binCount; ++r)
    {
        const int f = binIndices[(size_t) r];
        if (f < 0 || f >= numBins)
            continue;
        for (int m = 0; m < activeMicCount; ++m)
        {
            lastInferenceE(f, m) = E_obs(f, m);
            lastInferencePsi(f, m) = psi_obs(f, m);
            lastInferenceUX(f, m) = u_obs_X(f, m);
            lastInferenceUY(f, m) = u_obs_Y(f, m);
            lastInferenceUZ(f, m) = u_obs_Z(f, m);
        }
        lastInferenceFeatureValid[(size_t) f] = 1u;
    }
}

void DiracGNNEngine::resetWeightHistory() noexcept
{
    weightHistoryWriteIndex = 0;
    weightHistoryCount = 0;
    for (auto& h : weightHistory)
        if (h.size() != 0)
            h.setZero();
    for (auto& valid : weightHistoryValid)
        valid.fill(0u);
}
