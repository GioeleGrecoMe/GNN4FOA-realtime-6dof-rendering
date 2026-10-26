// GNN4FOA v2
// Authors: Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci
// Image and Sound Processing Lab (ISPL), Politecnico di Milano
// Copyright (c) 2026 Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci

#pragma once

// Backend selection is intentionally overrideable. On Apple hosts we use the
// long-standing Accelerate/vDSP matrix multiply when the SDK provides it.
// Tests and emergency compatibility builds can define
// GNN4FOA_FORCE_PORTABLE_GEMM=1 to exercise the scalar fallback instead.
#ifndef GNN4FOA_HAS_VDSP
 #if defined(__APPLE__) && !defined(GNN4FOA_FORCE_PORTABLE_GEMM)
  #if defined(__has_include)
   #if __has_include(<Accelerate/Accelerate.h>)
    #define GNN4FOA_HAS_VDSP 1
   #else
    #define GNN4FOA_HAS_VDSP 0
   #endif
  #else
   #define GNN4FOA_HAS_VDSP 1
  #endif
 #else
  #define GNN4FOA_HAS_VDSP 0
 #endif
#endif

// The GNN intentionally does not enable Eigen BLAS specialisations and does
// not call CBLAS/LAPACK. macOS matrix products use the long-standing vDSP_mmul
// path with prepacked row-major matrices, avoiding BLAS symbol-versioning on
// systems such as macOS Mojave.
#include <JuceHeader.h>
#include "Eigen/Dense"
#include <array>
#include <atomic>
#include <complex>
#include <cstdint>
#include <vector>

class DiracGNNEngine
{
public:
    static constexpr int kTrainingMicCount = 5; // checkpoint training cardinality
    static constexpr int kNumMics = 10; // maximum supported FOA arrays
    static constexpr int kNumNodes = kNumMics + 1; // maximum graph nodes = arrays + query
    static constexpr int kNumInputChannels = 4 * kNumMics; // fixed physical input capacity
    static constexpr int kNumOutputChannels = 4;
    static constexpr int kMaxTemporalWindow = 10;
    static constexpr double kModelSampleRate = 16000.0;

    DiracGNNEngine();
    ~DiracGNNEngine();

    // The learned model is defined at 16 kHz. The outer processor performs any
    // host-rate conversion and always calls prepare() with kModelSampleRate.
    void prepare(double sampleRate, int samplesPerBlock);
    void reset() noexcept;

    int getLatencySamples() const noexcept { return winLen - 1; }

    void process(const juce::AudioBuffer<float>& inputBuffer,
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
                 float outputReservoirAvailableMs);

    uint64_t getInvalidGNNFrameCount() const noexcept
    {
        return invalidGNNFrames.load(std::memory_order_relaxed);
    }

    uint64_t getReusedGNNFrameCount() const noexcept
    {
        return reusedGNNFrames.load(std::memory_order_relaxed);
    }

    float getLastMaxGNNWeight() const noexcept
    {
        return lastMaxGNNWeight.load(std::memory_order_relaxed);
    }

    int getLastGNNStage() const noexcept
    {
        return lastGNNStage.load(std::memory_order_relaxed);
    }

    double getLastGNNForwardMs() const noexcept
    {
        return lastGNNForwardMs.load(std::memory_order_relaxed);
    }

    // Pre-play calibration. Runs the real learned forward on the selected
    // backend using preallocated workspaces, then the caller resets the engine
    // before audio starts. This lets slower targets choose cadence/preload
    // from measured rather than assumed compute speed.
    double benchmarkGNNForwardMs(float qPosX,
                                 float qPosY,
                                 float qPosZ,
                                 const std::array<float, kNumMics>& micX,
                                 const std::array<float, kNumMics>& micY,
                                 const std::array<float, kNumMics>& micZ,
                                 int activeMicCount) noexcept;

    void seedGNNForwardTimingEstimate(double milliseconds, int activeMicCount = kTrainingMicCount) noexcept
    {
        const double ms = std::max(0.0, milliseconds);
        gnnForwardEwmaMs.store(ms, std::memory_order_relaxed);
        const double perBin = ms > 0.0 ? ms / 257.0 : 0.0;
        gnnForwardPerBinEwmaMs.store(perBin, std::memory_order_relaxed);
        gnnForwardPerBinPeakMs.store(perBin, std::memory_order_relaxed);
        const int clampedMicCount = activeMicCount < 1 ? 1 : (activeMicCount > kNumMics ? kNumMics : activeMicCount);
        gnnTimingReferenceMicCount.store(clampedMicCount, std::memory_order_relaxed);
        lastActiveMicCount = clampedMicCount;
        hasRuntimeGNNForwardTiming = false;
        gnnInferenceStride.store(chooseGNNInferenceStride(ms, 1.0f), std::memory_order_relaxed);
        gnnInferenceCountdown = 0; // first real learned field is still acquired immediately
    }

    int getGNNInferenceStride() const noexcept
    {
        return gnnInferenceStride.load(std::memory_order_relaxed);
    }

    double getGNNForwardEwmaMs() const noexcept
    {
        return gnnForwardEwmaMs.load(std::memory_order_relaxed);
    }

    int getLastGNNActiveBinCount() const noexcept
    {
        return lastGNNActiveBinCount.load(std::memory_order_relaxed);
    }

    uint64_t getGNNInferenceCount() const noexcept
    {
        return gnnInferenceCount.load(std::memory_order_relaxed);
    }

    uint64_t getGNNCadenceHeldFrameCount() const noexcept
    {
        return gnnCadenceHeldFrames.load(std::memory_order_relaxed);
    }

    uint64_t getGNNPressureDeferredFrameCount() const noexcept
    {
        return gnnPressureDeferredFrames.load(std::memory_order_relaxed);
    }

    uint64_t getGNNGeometryForcedInferenceCount() const noexcept
    {
        return gnnGeometryForcedInferences.load(std::memory_order_relaxed);
    }

    uint64_t getGNNAcousticForcedInferenceCount() const noexcept
    {
        return gnnAcousticForcedInferences.load(std::memory_order_relaxed);
    }

    uint64_t getGNNGateEnterCount() const noexcept
    {
        return gnnGateEntries.load(std::memory_order_relaxed);
    }

    uint64_t getGNNGateExitCount() const noexcept
    {
        return gnnGateExits.load(std::memory_order_relaxed);
    }

    uint64_t getGNNCoverageInferenceCount() const noexcept
    {
        return gnnCoverageOnlyInferences.load(std::memory_order_relaxed);
    }

    uint64_t getGNNFullInferenceCount() const noexcept
    {
        return gnnFullInferences.load(std::memory_order_relaxed);
    }

    int getLastGNNInferredBinCount() const noexcept
    {
        return lastGNNInferredBinCount.load(std::memory_order_relaxed);
    }

    int getLastGNNCoverageBinCount() const noexcept
    {
        return lastGNNCoverageBinCount.load(std::memory_order_relaxed);
    }

    double getLastPredictedGNNForwardMs() const noexcept
    {
        return lastPredictedGNNForwardMs.load(std::memory_order_relaxed);
    }

    double getGNNForwardPerBinEwmaMs() const noexcept
    {
        return gnnForwardPerBinEwmaMs.load(std::memory_order_relaxed);
    }

    float getLastAcousticNovelty() const noexcept
    {
        return lastAcousticNovelty.load(std::memory_order_relaxed);
    }

    static constexpr const char* getMatrixBackendName() noexcept
    {
       #if GNN4FOA_HAS_VDSP
        return "vDSP_mmul";
       #else
        return "portable";
       #endif
    }

    struct WeightVisualizationSnapshot
    {
        std::array<float, kNumMics> rawMean {};             // raw Softplus GNN weight
        std::array<float, kNumMics> gnnRelativeMean {};     // display-only GNN proportions (sum 1)
        std::array<float, kNumMics> delaunayRelativeMean {};
        std::array<float, kNumMics> relativeMean {};        // actually applied hybrid weights
        std::array<float, kNumMics> relativeLow {};
        std::array<float, kNumMics> relativeMid {};
        std::array<float, kNumMics> relativeHigh {};
        float meanRawSum = 0.0f;
        float meanAppliedSum = 0.0f;
        float maxRawWeight = 0.0f;
        float meanScale = 1.0f;
        float maxScale = 1.0f;
        float lowClampFraction = 0.0f;  // GNN-active bins whose raw sum was raised to 0.5
        float highClampFraction = 0.0f; // GNN-active bins whose raw sum was reduced to 2.0
        float gnnBinFraction = 0.0f;   // fraction of TF bins above threshold using GNN
        uint64_t frameCounter = 0;
        bool gnnMode = false;
    };

    WeightVisualizationSnapshot getWeightVisualizationSnapshot() const noexcept;

private:
    using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    struct MLP
    {
        // Stored already transposed for C = A * W^T.  Row-major contiguous
        // matrices let the macOS backend call the long-standing vDSP_mmul
        // interface directly, avoiding BLAS/LAPACK symbol-versioning entirely.
        RowMatrixXf W1T; // input x hidden
        Eigen::RowVectorXf b1;
        RowMatrixXf W2T; // hidden x output
        Eigen::RowVectorXf b2;
    };

    struct Vec2
    {
        float x = 0.0f;
        float y = 0.0f;
    };

    struct SpatialParams
    {
        float energy = 0.0f;
        float diffuseness = 1.0f;
        float ux = 0.0f;
        float uy = 0.0f;
        float uz = 0.0f;
    };

    double fs = kModelSampleRate;
    const int winLen = 512;
    const int hop = 128;
    const int nfft = 512;
    int numBins = 257;

    juce::dsp::FFT forwardFFT;
    juce::dsp::FFT inverseFFT;
    juce::dsp::WindowingFunction<float> window;

    juce::AudioBuffer<float> inputRingBuffer;
    juce::AudioBuffer<float> outputRingBuffer;
    juce::AudioBuffer<float> outputNormRingBuffer;
    int inputWritePointer = 0;
    int outputReadPointer = 0;
    int64_t samplesReceived = 0;

    std::vector<float> fftWorkspace;
    std::vector<float> windowCoefficients;
    std::vector<std::complex<float>> spectra;
    std::vector<std::complex<float>> querySpectra;
    std::vector<std::complex<float>> previousResidualSpectrum;
    std::vector<uint8_t> previousResidualValid;
    std::vector<uint8_t> useGNNForBin; // 1: GNN weights, 0: Delaunay weights
    std::array<uint8_t, 257> previousUseGNNForBin {};
    std::array<int, 257> reliableBinIndices {}; // all bins currently inside learned gate
    std::array<int, 257> coverageBinIndices {}; // gate-entered bins with no fresh learned row
    std::array<uint8_t, 257> validGNNBinMask {};
    std::array<float, 257> gnnGateMix {}; // 0=Delaunay, 1=GNN; de-clicked in both directions
    std::array<uint8_t, 257> lastInferenceFeatureValid {};

    Eigen::MatrixXf E_obs;
    Eigen::MatrixXf psi_obs;
    Eigen::MatrixXf u_obs_X;
    Eigen::MatrixXf u_obs_Y;
    Eigen::MatrixXf u_obs_Z;

    // Smoothed entries needed from the FOA spatial covariance matrix C.
    Eigen::MatrixXf covWW;
    Eigen::MatrixXf covYY;
    Eigen::MatrixXf covZZ;
    Eigen::MatrixXf covXX;
    Eigen::MatrixXcf covWX;
    Eigen::MatrixXcf covWY;
    Eigen::MatrixXcf covWZ;

    MLP outLayer;
    std::vector<MLP> msgLayers;
    std::vector<MLP> updLayers;
    int numLayers = 0;

    int maxHiddenDim = 0;
    int maxMessageInputDim = 0;
    int maxUpdateInputDim = 0;
    int maxOutInputDim = 0;
    int initialFeatureDim = 0;

    std::array<RowMatrixXf, kNumNodes> hiddenA;
    std::array<RowMatrixXf, kNumNodes> hiddenB;
    RowMatrixXf aggregateScratch;
    RowMatrixXf edgeFeatureScratch;
    RowMatrixXf messageInputScratch;
    RowMatrixXf updateInputScratch;
    RowMatrixXf outInputScratch;
    RowMatrixXf mlpHiddenScratch;
    RowMatrixXf mlpOutputScratch;

    // Batched message/update/readout workspaces. Batching the 30 directed graph
    // edges into large GEMMs is important on older Intel Macs, where many tiny
    // Eigen products otherwise miss the 8 ms model-frame deadline.
    RowMatrixXf messageBatchInput;
    RowMatrixXf messageBatchHidden;
    RowMatrixXf messageBatchOutput;
    RowMatrixXf updateBatchInput;
    RowMatrixXf updateBatchHidden;
    RowMatrixXf updateBatchOutput;
    RowMatrixXf outBatchInput;
    RowMatrixXf outBatchHidden;
    RowMatrixXf outBatchOutput;
    std::vector<float> gemmPackScratch; // preallocated in prepare(); never allocates in worker

    Eigen::MatrixXf weightsRaw;
    Eigen::MatrixXf weightsPreConstraint;
    Eigen::MatrixXf weightsGNNConstrained;
    Eigen::MatrixXf weightsCandidate;
    Eigen::MatrixXf weightsDelaunay;
    Eigen::MatrixXf weightsUsed;
    Eigen::MatrixXf lastValidGNNWeights;
    Eigen::MatrixXf gnnBlendFrom;
    Eigen::MatrixXf gnnBlendTo;
    Eigen::MatrixXf lastInferenceE;
    Eigen::MatrixXf lastInferencePsi;
    Eigen::MatrixXf lastInferenceUX;
    Eigen::MatrixXf lastInferenceUY;
    Eigen::MatrixXf lastInferenceUZ;
    std::array<Eigen::MatrixXf, kMaxTemporalWindow> weightHistory;
    std::array<std::array<uint8_t, 257>, kMaxTemporalWindow> weightHistoryValid {};
    int weightHistoryWriteIndex = 0;
    int weightHistoryCount = 0;
    int gnnBlendRemaining = 0;
    int gnnBlendTotalFrames = 2;
    static constexpr int kGNNBlendFrames = 2; // learned-weight update blend, 16 ms at 125 STFT frames/s
    static constexpr int kGNNGateBlendFrames = 2; // Delaunay<->GNN parameter transition, both directions
    static constexpr float kGNNGateExitEnergyRatio = 0.5011872336f; // -3 dB energy hysteresis
    int lastInterpolationMode = -1;
    bool hasLastValidGNNWeights = false;

    int currentAnchor = -1;
    int previousAnchor = -1;
    int anchorCrossfadeRemaining = 0;
    int anchorCrossfadeTotalFrames = 0;
    static constexpr int kPaperAnchorCrossfadeFrames = 2;
    static constexpr int kLearnedAnchorCrossfadeFrames = 8;
    static constexpr float kLearnedAnchorSwitchRatio = 0.85f;

    std::atomic<uint64_t> invalidGNNFrames { 0 };
    std::atomic<uint64_t> reusedGNNFrames { 0 };
    std::atomic<float> lastMaxGNNWeight { 0.0f };
    std::atomic<int> lastGNNStage { 0 };
    std::atomic<double> lastGNNForwardMs { 0.0 };
    std::atomic<double> gnnForwardEwmaMs { 0.0 };
    bool hasRuntimeGNNForwardTiming = false;
    std::atomic<int> gnnInferenceStride { 4 };
    std::atomic<int> lastGNNActiveBinCount { 0 };
    std::atomic<uint64_t> gnnInferenceCount { 0 };
    std::atomic<uint64_t> gnnCadenceHeldFrames { 0 };
    std::atomic<uint64_t> gnnPressureDeferredFrames { 0 };
    std::atomic<uint64_t> gnnGeometryForcedInferences { 0 };
    std::atomic<uint64_t> gnnAcousticForcedInferences { 0 };
    std::atomic<uint64_t> gnnGateEntries { 0 };
    std::atomic<uint64_t> gnnGateExits { 0 };
    std::atomic<uint64_t> gnnCoverageOnlyInferences { 0 };
    std::atomic<uint64_t> gnnFullInferences { 0 };
    std::atomic<int> lastGNNInferredBinCount { 0 };
    std::atomic<int> lastGNNCoverageBinCount { 0 };
    std::atomic<double> lastPredictedGNNForwardMs { 0.0 };
    std::atomic<double> gnnForwardPerBinEwmaMs { 0.0 };
    std::atomic<double> gnnForwardPerBinPeakMs { 0.0 };
    std::atomic<int> gnnTimingReferenceMicCount { kTrainingMicCount };
    std::atomic<float> lastAcousticNovelty { 0.0f };
    int gnnInferenceCountdown = 0;
    int weightVizCadenceCountdown = 0;
    bool weightVizHasPublished = false;
    int lastActiveMicCount = kNumMics;
    std::array<float, kNumMics> lastSchedulerMicX {};
    std::array<float, kNumMics> lastSchedulerMicY {};
    std::array<float, kNumMics> lastSchedulerMicZ {};
    std::array<float, kNumMics> lastSchedulerMicAzimuth {};
    float lastSchedulerListenerX = 0.0f;
    float lastSchedulerListenerY = 0.0f;
    float lastSchedulerListenerZ = 0.0f;
    bool schedulerGeometryInitialised = false;

    // Lock-free display diagnostics. Relative values are normalised only for
    // visualisation; DSP uses the corresponding positive GNN weights with row sums bounded to [0.5, 2.0].
    std::atomic<uint32_t> weightVizVersion { 0u };
    std::array<std::atomic<float>, kNumMics> weightVizRawMean {};
    std::array<std::atomic<float>, kNumMics> weightVizGNNRelativeMean {};
    std::array<std::atomic<float>, kNumMics> weightVizDelaunayRelativeMean {};
    std::array<std::atomic<float>, kNumMics> weightVizRelativeMean {};
    std::array<std::atomic<float>, kNumMics> weightVizRelativeLow {};
    std::array<std::atomic<float>, kNumMics> weightVizRelativeMid {};
    std::array<std::atomic<float>, kNumMics> weightVizRelativeHigh {};
    std::atomic<float> weightVizMeanRawSum { 0.0f };
    std::atomic<float> weightVizMeanAppliedSum { 0.0f };
    std::atomic<float> weightVizMaxRawWeight { 0.0f };
    std::atomic<float> weightVizMeanScale { 1.0f };
    std::atomic<float> weightVizMaxScale { 1.0f };
    std::atomic<float> weightVizLowClampFraction { 0.0f };
    std::atomic<float> weightVizHighClampFraction { 0.0f };
    std::atomic<float> weightVizGNNBinFraction { 0.0f };
    std::atomic<uint64_t> weightVizFrameCounter { 0u };
    std::atomic<bool> weightVizGnnMode { false };
    std::array<float, kNumMics> weightVizEmaRawMean {};
    std::array<float, kNumMics> weightVizEmaGNNRelativeMean {};
    std::array<float, kNumMics> weightVizEmaDelaunayRelativeMean {};
    std::array<float, kNumMics> weightVizEmaRelativeMean {};
    std::array<float, kNumMics> weightVizEmaRelativeLow {};
    std::array<float, kNumMics> weightVizEmaRelativeMid {};
    std::array<float, kNumMics> weightVizEmaRelativeHigh {};
    float weightVizEmaMeanRawSum = 0.0f;
    float weightVizEmaMeanAppliedSum = 0.0f;
    float weightVizEmaMeanScale = 1.0f;
    float weightVizEmaMaxScale = 1.0f;
    float weightVizEmaLowClampFraction = 0.0f;
    float weightVizEmaHighClampFraction = 0.0f;
    float weightVizEmaGNNBinFraction = 0.0f;
    bool weightVizEmaInitialised = false;

    // Orientation metadata can be edited live. Covariance cross-terms are
    // re-expressed into the new global yaw frame before recursive smoothing so
    // a metadata change does not leave stale intensity pointing in the old
    // direction.
    std::array<float, kNumMics> lastMicAzimuth {};
    bool micAzimuthStateInitialised = false;

    bool constrainGNNWeightSums(Eigen::MatrixXf& weights, int activeMicCount) noexcept;
    void publishWeightVisualization(const Eigen::MatrixXf& rawWeights,
                                    const Eigen::MatrixXf& gnnWeights,
                                    const Eigen::MatrixXf& delaunayWeights,
                                    const Eigen::MatrixXf& appliedWeights,
                                    bool gnnMode) noexcept;

    void processSTFTFrame(float listenerX,
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
                          float outputReservoirAvailableMs);

    bool forwardGNN(float qPosX,
                    float qPosY,
                    float qPosZ,
                    const std::array<float, kNumMics>& micX,
                    const std::array<float, kNumMics>& micY,
                    const std::array<float, kNumMics>& micZ,
                    Eigen::MatrixXf& outputWeights,
                    int activeMicCount,
                    const int* binIndices,
                    int binCount);

    void computeDelaunayWeights(float qPosX,
                                float qPosY,
                                const std::array<float, kNumMics>& micX,
                                const std::array<float, kNumMics>& micY,
                                Eigen::MatrixXf& outputWeights,
                                int activeMicCount) const noexcept;

    void applyTemporalWeightWindowForBins(const Eigen::MatrixXf& currentWeights,
                                          const int* binIndices,
                                          int binCount,
                                          int temporalWindow,
                                          Eigen::MatrixXf& outputWeights) noexcept;

    bool constrainGNNWeightSumsForBins(Eigen::MatrixXf& weights,
                                       int activeMicCount,
                                       const int* binIndices,
                                       int binCount) noexcept;

    int chooseGNNInferenceStride(double forwardEwmaMs, float reservoirFill01) const noexcept;
    double predictGNNForwardMs(int requestedBinCount, int activeBinCount, bool conservative, int activeMicCount) const noexcept;
    static bool gateWantsGNN(bool wasGNN, float binPeakEnergy, float enterThresholdEnergy) noexcept;
    static float advanceGateMix(float currentMix, bool targetGNN) noexcept;
    static bool shouldUseCoverageOnly(bool hasLastValidWeights,
                                      int inferenceCountdown,
                                      int coverageBinCount) noexcept;
    bool geometryRequiresImmediateGNN(float listenerX, float listenerY, float listenerZ,
                                      const std::array<float, kNumMics>& micX,
                                      const std::array<float, kNumMics>& micY,
                                      const std::array<float, kNumMics>& micZ,
                                      const std::array<float, kNumMics>& micAzimuth,
                                      int activeMicCount) noexcept;
    void updateGNNBlend(Eigen::MatrixXf& outputWeights) noexcept;
    void invalidateTemporalHistoryForBin(int bin) noexcept;
    bool acousticFeaturesChanged(const int* binIndices, int binCount, int activeMicCount,
                                 float& noveltyOut) const noexcept;
    void captureAcousticFeatureSnapshot(const int* binIndices, int binCount, int activeMicCount) noexcept;
    void resetWeightHistory() noexcept;

    void mlpApplyPrepared(const RowMatrixXf& inputStorage,
                          int inputColumns,
                          const MLP& layer,
                          bool useSoftplus,
                          RowMatrixXf& outputStorage) noexcept;

    void mlpApplyBatch(const RowMatrixXf& inputStorage,
                       int rows,
                       int inputColumns,
                       const MLP& layer,
                       bool useSoftplus,
                       RowMatrixXf& hiddenStorage,
                       RowMatrixXf& outputStorage) noexcept;

    void selectAnchor(float listenerX,
                      float listenerY,
                      float listenerZ,
                      const std::array<float, kNumMics>& micX,
                      const std::array<float, kNumMics>& micY,
                      const std::array<float, kNumMics>& micZ,
                      int activeMicCount,
                      bool exactNearestPrototype) noexcept;

    std::complex<float> getTransitionAnchorSpectrum(int channelWithinFOA,
                                                    int bin) const noexcept;

    static bool barycentric(const Vec2& p,
                            const Vec2& a,
                            const Vec2& b,
                            const Vec2& c,
                            std::array<float, 3>& w) noexcept;

    static int buildDelaunayTriangles(const std::array<Vec2, kNumMics>& points,
                                      int activeMicCount,
                                      std::array<std::array<int, 3>, 32>& triangles) noexcept;

    static Vec2 closestPointOnSegment(const Vec2& p,
                                      const Vec2& a,
                                      const Vec2& b,
                                      float& t) noexcept;

    SpatialParams interpolateLearnedParams(int bin, int activeMicCount) const noexcept;
    static SpatialParams blendSpatialParams(const SpatialParams& delaunay,
                                            const SpatialParams& learned,
                                            float learnedMix) noexcept;

    SpatialParams interpolatePaperDelaunayParams(int bin,
                                                  const Eigen::MatrixXf& weights,
                                                  int activeMicCount) const noexcept;
};
