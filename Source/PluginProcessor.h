// GNN4FOA v2
// Authors: Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci
// Image and Sound Processing Lab (ISPL), Politecnico di Milano
// Copyright (c) 2026 Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci

#pragma once

#include <JuceHeader.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "DiracGNNEngine.h"
#include "ManualFoaPolicy.h"

struct SpatialNode
{
    int id = 0;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float azimuth = 0.0f; // radians internally; microphone yaw rotates local FOA into the global frame
    bool isListener = false;
    juce::String name;
};

// GUI-only map annotations. They are serialized in the plugin state but are
// never published to the DSP, so walls/sources/sketches cannot affect either
// Delaunay or GNN rendering.
struct MapAnnotation
{
    enum class Type : int { wallSegment = 0, source = 1, dashedStroke = 2 };
    Type type = Type::wallSegment;
    std::vector<juce::Point<float>> points;
};

class GNNReconstructorAudioProcessor : public juce::AudioProcessor
{
public:
    static constexpr int kNumMics = 10; // maximum supported microphone arrays
    static constexpr int kModelBlockSize = 128;
    static constexpr double kModelSampleRate = DiracGNNEngine::kModelSampleRate;
    static constexpr const char* kReleaseVersion = "v2.0.0";
    static constexpr const char* kBuildFingerprint = "GNN4FOA-V2.0.0-ISPL";

    struct AudioSpatialSnapshot
    {
        std::array<float, kNumMics> micX {};
        std::array<float, kNumMics> micY {};
        std::array<float, kNumMics> micZ {};
        std::array<float, kNumMics> micAzimuth {};
        float listenerX = 0.0f;
        float listenerY = 0.0f;
        float listenerZ = 0.0f;
    };

    struct DiagnosticsSnapshot
    {
        double hostSampleRate = 0.0;
        double modelSampleRate = kModelSampleRate;
        double lastWorkerMs = 0.0;
        double maxWorkerMs = 0.0;
        uint64_t modelBlocks = 0;
        uint64_t deadlineMisses = 0;
        uint64_t inputFifoOverflows = 0;
        uint64_t outputFifoOverflows = 0;
        uint64_t outputStarvations = 0;
        uint64_t lateOutputFramesDropped = 0;
        uint64_t outputBacklogFramesTrimmed = 0;
        uint64_t outputTimelineDropDebt = 0;
        uint64_t invalidGNNFrames = 0;
        uint64_t reusedGNNFrames = 0;
        int lastGNNStage = 0;
        float lastMaxGNNWeight = 0.0f;
        float meanRawGNNWeightSum = 0.0f;
        float meanAppliedGNNWeightSum = 0.0f;
        float meanGNNWeightScale = 1.0f;
        float maxGNNWeightScale = 1.0f;
        float lowClampFraction = 0.0f;
        float highClampFraction = 0.0f;
        float gnnBinFraction = 0.0f;
        bool resamplingActive = false;
        int activeWorkerMode = -1;
        int hostInputFifoDepth = 0;
        int hostOutputFifoDepth = 0;
        double workerPrebufferMs = 0.0;
        double outputQueueMs = 0.0;
        double minOutputQueueMs = 0.0;
        double excessOutputQueueMs = 0.0;
        bool timelineCurrentlyHealthy = true;
        int inputChannels = 0; // physical flat-bus capacity: always 40
        int deliveredInputChannels = 0; // process-buffer channels visible to the host
        int configuredMicCount = 0;
        int activeMicCount = 0;
        int nodeCountMode = 5; // manual N, 1..10; never host-derived
        double lastGNNForwardMs = 0.0;
        double gnnForwardEwmaMs = 0.0;
        int gnnInferenceStride = 4;
        int gnnActiveBinCount = 0;
        uint64_t gnnInferenceCount = 0;
        uint64_t gnnCadenceHeldFrames = 0;
        uint64_t gnnPressureDeferredFrames = 0;
        uint64_t gnnGeometryForcedInferences = 0;
        uint64_t gnnAcousticForcedInferences = 0;
        uint64_t gnnGateEntries = 0;
        uint64_t gnnGateExits = 0;
        uint64_t gnnCoverageOnlyInferences = 0;
        uint64_t gnnFullInferences = 0;
        int gnnLastInferredBinCount = 0;
        int gnnCoverageBinCount = 0;
        double gnnPredictedForwardMs = 0.0;
        double gnnForwardPerBinEwmaMs = 0.0;
        float gnnAcousticNovelty = 0.0f;
        float outputReservoirFill = 1.0f;
        double startupGNNBenchmarkMs = 0.0;
        double adaptivePrebufferMs = 0.0;
    };

    GNNReconstructorAudioProcessor();
    ~GNNReconstructorAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
   #endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

    std::vector<SpatialNode> micArrays;
    SpatialNode listener;

    // Retained for backward-compatible session loading. The learned renderer
    // uses the complete graph; these UI-only custom edges do not affect DSP.
    std::vector<std::pair<int, int>> customEdges;

    // Persistent editor/map state. These fields are deliberately separate
    // from AudioSpatialSnapshot and therefore never enter the audio worker.
    std::vector<MapAnnotation> mapAnnotations;
    bool mapSceneLocked = false;
    float mapViewCenterX = 0.0f;
    float mapViewCenterY = 0.0f;
    float mapViewZoom = 1.0f;

    void publishSpatialStateFromGui() noexcept;
    DiagnosticsSnapshot getDiagnosticsSnapshot() const noexcept;
    DiracGNNEngine::WeightVisualizationSnapshot getWeightVisualizationSnapshot() const noexcept
    {
        return gnnEngine.getWeightVisualizationSnapshot();
    }
    int getActiveMicCount() const noexcept { return effectiveMicCount.load(std::memory_order_relaxed); }
    int getSceneMicCount() const noexcept;
    int getActiveInputChannels() const noexcept { return getActiveMicCount() * 4; }

    juce::String getBuildIdentity() const
    {
        return juce::String("GNN4FOA ") + kReleaseVersion + " | ISPL Politecnico di Milano";
    }

private:
    class InterleavedSpscFifo
    {
    public:
        void prepare(int channelsToUse, int capacityFramesToUse);
        void reset() noexcept;
        bool pushFrame(const float* frame) noexcept;
        bool popFrame(float* frame) noexcept;
        int discardOldestFrames(int maxFrames) noexcept;
        int getAvailableFrames() const noexcept;

    private:
        int channels = 0;
        int capacityFrames = 0;
        std::vector<float> data;
        std::atomic<uint64_t> writeCounter { 0 };
        std::atomic<uint64_t> readCounter { 0 };
    };

    class StreamingSincResampler
    {
    public:
        // Polyphase windowed-sinc SRC. All sin/cos work is done in prepare(),
        // never in the audio callback or the steady-state worker loop.
        static constexpr int kTaps = 48;
        static constexpr int kPhases = 1024;
        static constexpr int kHistorySize = 128;
        static constexpr int kMaxGeneratedFrames = 16;
        static constexpr int kMaxChannels = 40;

        void prepare(double sourceRateToUse, double destinationRateToUse, int channelsToUse) noexcept;
        void reset() noexcept;
        int processFrame(const float* inputFrame, float* outputInterleaved, int maxOutputFrames) noexcept;
        bool isBypassed() const noexcept { return bypass; }
        double getDelaySeconds() const noexcept;

    private:
        float getSample(int channel, int64_t absoluteIndex) const noexcept;
        float interpolate(int channel, double delayedTime) const noexcept;

        int channels = 0;
        double sourceRate = 1.0;
        double destinationRate = 1.0;
        double sourceStepPerOutput = 1.0;
        double nextOutputSourceTime = 0.0;
        double cutoff = 0.475;
        int64_t sourceIndex = -1;
        bool bypass = true;
        std::array<float, kMaxChannels * kHistorySize> history {};
        std::array<float, kPhases * kTaps> phaseTable {};
    };

    class ModelWorker final : public juce::Thread
    {
    public:
        explicit ModelWorker(GNNReconstructorAudioProcessor& ownerToUse)
            : juce::Thread("GNN 16k worker"), owner(ownerToUse) {}
        void run() override;
    private:
        GNNReconstructorAudioProcessor& owner;
    };

    DiracGNNEngine gnnEngine;
    std::unique_ptr<ModelWorker> modelWorker;
    juce::WaitableEvent workerWakeEvent;

    InterleavedSpscFifo hostInputFifo;    // audio -> worker, fixed 40 ch capacity; first 4*N active
    InterleavedSpscFifo hostOutputFifo;   // worker -> audio, 4 ch @ host Fs

    StreamingSincResampler inputResampler;
    StreamingSincResampler outputResampler;

    juce::AudioBuffer<float> workerInputBuffer;
    juce::AudioBuffer<float> workerOutputBuffer;

    double hostSampleRate = kModelSampleRate;
    int hostMaxBlockSize = 512;
    bool resamplingActive = false;
    int configuredInputChannels = 4;
    int configuredMicCount = 1;
    std::atomic<int> deliveredInputChannels { DiracGNNEngine::kNumInputChannels };
    std::atomic<int> effectiveMicCount { 1 };

    int safetyLatencyHostSamples = 0;
    int workerSafetyModelSamples = 8 * kModelBlockSize; // adaptive 64..512 ms reservoir at 16 kHz
    double startupGNNBenchmarkMs = 0.0;
    int workerModelInputFill = 0;

    std::array<float, DiracGNNEngine::kNumInputChannels> hostInputFrame {};
    std::array<float, DiracGNNEngine::kNumOutputChannels> modelOutputFrame {};
    std::array<float, DiracGNNEngine::kNumOutputChannels> hostOutputFrame {};
    std::array<float, StreamingSincResampler::kMaxGeneratedFrames * DiracGNNEngine::kNumInputChannels> inputResampleScratch {};
    std::array<float, StreamingSincResampler::kMaxGeneratedFrames * DiracGNNEngine::kNumOutputChannels> outputResampleScratch {};
    std::array<float, DiracGNNEngine::kNumOutputChannels> lastRenderedSample {};
    float starvationGain = 1.0f;

    std::array<std::atomic<float>, kNumMics> micXAtomic;
    std::array<std::atomic<float>, kNumMics> micYAtomic;
    std::array<std::atomic<float>, kNumMics> micZAtomic;
    std::array<std::atomic<float>, kNumMics> micAzimuthAtomic;
    std::atomic<float> listenerXAtomic { 0.0f };
    std::atomic<float> listenerYAtomic { 0.0f };
    std::atomic<float> listenerZAtomic { 0.0f };
    std::atomic<uint32_t> spatialStateVersion { 0u };

    std::atomic<float>* interpModeParam = nullptr;
    std::atomic<float>* energyThresholdParam = nullptr;
    std::atomic<float>* diracSmoothingParam = nullptr;
    std::atomic<float>* temporalWindowParam = nullptr;
    std::atomic<float>* nodeCountParam = nullptr;

    std::atomic<uint64_t> modelBlocksProcessed { 0 };
    std::atomic<uint64_t> deadlineMisses { 0 };
    std::atomic<uint64_t> inputFifoOverflows { 0 };
    std::atomic<uint64_t> outputFifoOverflows { 0 };
    std::atomic<uint64_t> outputStarvations { 0 };
    std::atomic<uint64_t> lateOutputFramesDropped { 0 };
    std::atomic<uint64_t> outputBacklogFramesTrimmed { 0 };
    std::atomic<uint64_t> outputTimelineDropDebt { 0 };
    std::atomic<int> minOutputFifoDepth { 0x7fffffff };
    std::atomic<double> lastWorkerMs { 0.0 };
    std::atomic<double> maxWorkerMs { 0.0 };
    std::atomic<float> outputReservoirFill { 1.0f };
    std::atomic<int> activeWorkerMode { -1 };

    juce::AudioProcessorValueTreeState::ParameterLayout createParameters();
    AudioSpatialSnapshot readSpatialSnapshot() const noexcept;
    void runModelWorker();
    void pushHostInputsToWorkerFifo(const juce::AudioBuffer<float>& buffer) noexcept;
    void renderHostOutput(juce::AudioBuffer<float>& buffer) noexcept;
    void trimExcessOutputBacklog() noexcept;
    void processOneModelBlockOnWorker();
    int calculateLatencySamples(double sampleRate) const noexcept;
    void resetDiagnostics() noexcept;

    int getSelectedMicCount() const noexcept;
    int graphIndexForNodeId(int nodeId) const noexcept;
    void restoreSpatialStateFromValueTree(const juce::ValueTree& state);
    void appendSpatialStateToValueTree(juce::ValueTree& state) const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GNNReconstructorAudioProcessor)
};
